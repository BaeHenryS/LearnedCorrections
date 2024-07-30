import numpy as np
import os
from scipy.signal import convolve
import burgers_1d
from phi.tf.flow import *

import tensorflow as tf
from tensorflow.keras import layers, models
import concurrent.futures

class DataLoader:
    def __init__(self, simulation_path, batch_size, k):
        self.simulation_path = simulation_path
        self.batch_size = batch_size
        self.k = k
        self.num_simulations, self.num_timesteps = self._count_simulations_and_timesteps()
        self.data = self._load_velocity_data()
        self.batches, self.batch_velocity_std = self._construct_batches()
        self.current_batch_index = 0
        self.current_epoch = 0

    def _count_simulations_and_timesteps(self):
        # Count the number of simulation directories
        simulation_dirs = [d for d in os.listdir(self.simulation_path) if os.path.isdir(os.path.join(self.simulation_path, d))]
        num_simulations = len(simulation_dirs)

        # Count the number of velocity files in the first simulation directory
        first_sim_dir = os.path.join(self.simulation_path, simulation_dirs[0])
        velocity_files = [f for f in os.listdir(first_sim_dir) if f.startswith('velocity') and f.endswith('.npz')]
        num_timesteps = len(velocity_files)
        print(f"Number of simulations: {num_simulations}, Number of timesteps: {num_timesteps}")
        return num_simulations, num_timesteps
    

    def _load_velocity_data(self):
        def load_data_with_progress(sim):
            result = self._load_simulation_data(sim)
            progress = (sim + 1) / self.num_simulations * 100  # Calculate progress
            print(f"Progress: {progress:.2f}%")
            return result

        # Specify max_workers. Adjust the number based on your system's capabilities
        #num_workers = 20  # Example: Set this to a higher number based on your system's capabilities

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Use list to preserve order, map through the simulations with their indices
            data = list(executor.map(load_data_with_progress, range(self.num_simulations)))
        return data

    def _load_simulation_data(self, sim):
        sim_dir = os.path.join(self.simulation_path, f'sim_{sim:06d}')
        if not os.path.exists(sim_dir):
            print(f"Simulation directory {sim_dir} not found, skipping to the next simulation.")
            return []  # Return empty list for missing simulations
        sim_data = []
        for timestep in range(self.num_timesteps):
            try:
                timestep_file = os.path.join(sim_dir, f'velocity_{timestep:06d}.npz')
                timestep_data = np.load(timestep_file)['data']
                # Include simulation identifier with the data
                sim_data.append((f'sim_{sim:06d}', timestep_data))
            except FileNotFoundError:
                print(f"File {timestep_file} not found, skipping this timestep.")
                continue  # Skip the current timestep if the file does not exist
        print(f"Loaded simulation {sim + 1}/{self.num_simulations}")
        return sim_data

    def _get_velocity_std(self, sim_num): 
        # Calculate the standard deviation of the velocity
        velocity_std = np.std(self.data[sim_num])
        return velocity_std 

    # def _construct_batches(self):
    #     input_data = []
    #     output_data = []
    #     sim_indices = []
    #     time_indices = []
    #     batch_velocity_std = []

    #     for sim_index, sim_data in enumerate(self.data):
    #         for i in range(len(sim_data) - self.k):
    #             input_data.append(sim_data[i])
    #             output_data.append(sim_data[i + 1: i + self.k + 1])
    #             sim_indices.append(sim_index)
    #             time_indices.append(i)

    #     # Create indices for shuffling
    #     indices = np.arange(len(input_data))
    #     np.random.shuffle(indices)

    #     # Shuffle data
    #     input_data = np.array(input_data)[indices]
    #     output_data = np.array(output_data)[indices]
    #     sim_indices = np.array(sim_indices)[indices]
    #     time_indices = np.array(time_indices)[indices]

    #     # Create batches
    #     num_batches = len(input_data) // self.batch_size
    #     batches = []
    #     for i in range(num_batches):
    #         batch_input = input_data[i * self.batch_size: (i + 1) * self.batch_size]
    #         batch_output = output_data[i * self.batch_size: (i + 1) * self.batch_size]
    #         batch_sim_indices = sim_indices[i * self.batch_size: (i + 1) * self.batch_size]
    #         batch_time_indices = time_indices[i * self.batch_size: (i + 1) * self.batch_size]
    #         batches.append((batch_input, batch_output, batch_sim_indices, batch_time_indices))

    #         # Calculate and store standard deviation of the velocity of the batch
    #         velocity_std = np.std(batch_input)
    #         batch_velocity_std.append(velocity_std)

    #     return batches, batch_velocity_std

    def _construct_batches(self):
        total_data_points = sum(len(sim_data) - self.k for sim_data in self.data)
        input_data = np.empty(total_data_points, dtype=object)  # Assuming sim_data[i] are objects or arrays
        output_data = np.empty(total_data_points, dtype=object)
        sim_indices = np.empty(total_data_points, dtype=int)
        time_indices = np.empty(total_data_points, dtype=int)

        index = 0
        for sim_index, sim_data in enumerate(self.data):
            for i in range(len(sim_data) - self.k):
                input_data[index] = sim_data[i]
                output_data[index] = sim_data[i + 1: i + self.k + 1]
                sim_indices[index] = sim_index
                time_indices[index] = i
                index += 1

        # Shuffle data
        indices = np.arange(total_data_points)
        np.random.shuffle(indices)
        input_data = input_data[indices]
        output_data = output_data[indices]
        sim_indices = sim_indices[indices]
        time_indices = time_indices[indices]

        # Create batches
        num_batches = total_data_points // self.batch_size
        batches = []
        batch_velocity_std = []
        for i in range(num_batches):
            print("Creating batch", i + 1, "of", num_batches)
            batch_start = i * self.batch_size
            batch_end = (i + 1) * self.batch_size
            batch_input = input_data[batch_start:batch_end]
            batch_output = output_data[batch_start:batch_end]
            batch_sim_indices = sim_indices[batch_start:batch_end]
            batch_time_indices = time_indices[batch_start:batch_end]
            batches.append((batch_input, batch_output, batch_sim_indices, batch_time_indices))

            # Calculate and store standard deviation of the velocity of the batch
            # Assuming batch_input is a NumPy array of velocities; adjust calculation as needed
            velocity_std = np.std(np.concatenate(batch_input))
            batch_velocity_std.append(velocity_std)

        print("Batches created")

        return batches, batch_velocity_std

    def get_batches(self):
        return self.batches, self.batch_velocity_std

    def next_batch(self):
        if self.current_batch_index >= len(self.batches):
            self.current_batch_index = 0
            self.current_epoch += 1

        batch = self.batches[self.current_batch_index]
        batch_std = self.batch_velocity_std[self.current_batch_index]
        self.current_batch_index += 1
        return batch, self.current_batch_index - 1, self.current_epoch, batch_std


simulation_path = './output'  # path to the directory containing the simulations
k = 3  # future timestep interval

LR = 1e-4

optimizer = tf.keras.optimizers.Adam(learning_rate=LR) 




import burgers_1d
from phi.flow import *
import tensorflow as tf

simulator = burgers_1d.Burgers_1d(resolution=128, output='../test_output')

def to_phiflow_format(data):
    # Check if data is already a phi tensor
    if isinstance(data, CenteredGrid):
        return data
    else:
        # Convert numpy.ndarray to TensorFlow tensor
        data_tf = tf.convert_to_tensor(data)
        # Then convert TensorFlow tensor to phi tensor
        grid = CenteredGrid(math.tensor(data_tf, spatial('x')), extrapolation.PERIODIC, x=int(128), bounds=Box['x', slice(0, 2 * np.pi)])
        return grid
def to_numpy_format(data):
    return data.values.numpy('x')

def to_tensor_format(data):
    return tf.convert_to_tensor(data.values.numpy('x'))


def to_model_format(data):
    return tf.expand_dims(tf.expand_dims(data, 0), -1)  # Reshape to (1, 128, 1)


# def transformation(velocity_function, output_data):
#     output_data = tf.reshape(output_data, [-1])
#     return convolve(velocity_function, output_data, mode='same')

def transformation(velocity_function, output_data):
    # Reshape velocity_function to (1, length, 1) and output_data to (filter_length, 1, 1)
    velocity_function = tf.reshape(velocity_function, [1, -1, 1])  # (1, 128, 1)
    output_data = tf.reshape(output_data, [-1, 1, 1])  # (8, 1, 1)

    # Perform the 1D convolution
    convolved_data = tf.nn.conv1d(velocity_function, output_data, stride=1, padding='SAME')

    # Squeeze the output to remove unnecessary dimensions, resulting in (128,)
    return tf.squeeze(convolved_data, axis=[0, 2])


# def training_step(input, output, simulationnumber, init_time, velocity_std, model, optimizer):
#     '''
#     Input Shape: (dim_x)
#     Output Shape: (k, dim_x)
#     '''
#     simulator.reset()


    

#     simulator.change_forcing_dir(os.path.join(simulation_path, f'sim_{simulationnumber:06d}', 'params.json'))
    

#     prediction, coefficients = [input], [0]

    
#     with tf.GradientTape() as tape:
#         for i in range(k):
#             prediction_flow = simulator.step(to_phiflow_format(prediction[-1]), t=init_time + i)
#             prediction_flow = to_numpy_format(prediction_flow)

#             prediction_flow_normalized = np.divide(prediction_flow, (velocity_std + 1e-6))

#             prediction += [prediction_flow]

#             model_input = tf.expand_dims(tf.expand_dims(prediction_flow_normalized, 0), -1)  # Reshape to (1, 128, 1)

#             model_output = model(model_input)
#             coefficients += model_output

#             correction = transformation(prediction[-1], model_output)
#             prediction[-1] = prediction[-1] + correction

#         # Calculate the Loss Function using MAE
#         final_prediction = tf.stack(prediction[1:]) 

#         loss = tf.reduce_mean(tf.abs(final_prediction - output))  # Use MAE for loss
#         # Calculate the Gradients
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#         return loss
    





def training_step(input, output, init_time, velocity_std, model, optimizer):
    '''
    Input Shape: (dim_x)
    Output Shape: (k, dim_x)
    '''
    prediction, coefficients = [input], [0]

    
    with tf.GradientTape() as tape:
        for i in range(k):
            prediction_flow = simulator.step(to_phiflow_format(prediction[-1]), t=init_time + i)
            prediction_flow = to_numpy_format(prediction_flow)

            prediction_flow_normalized = np.divide(prediction_flow, (velocity_std + 1e-6))

            prediction += [prediction_flow]

            model_input = tf.expand_dims(tf.expand_dims(prediction_flow_normalized, 0), -1)  # Reshape to (1, 128, 1)

            model_output = model(model_input)
            model_output *= velocity_std
            coefficients += model_output

            correction = transformation(prediction[-1], model_output)
            prediction[-1] = prediction[-1] + correction

        # Calculate the Loss Function using MAE

        final_prediction = tf.stack(prediction[1:]) 

        loss = tf.reduce_mean(tf.abs(final_prediction - output))  # Use MAE for loss
        # Calculate the Gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss



## TESTING

def training_step_spatial(input, output, init_time, velocity_std, model, optimizer):
    prediction = [input]
    # Ensure init_time is of type float32 or float64
    init_time = tf.cast(init_time, tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        for i in range(k):
            pred_last = prediction[-1]

            # Cast simulator.dt to float32 and perform arithmetic operation
            time_step = tf.cast(i, tf.float32) * tf.cast(simulator.dt, tf.float32)
            k1 = simulator.equation(to_phiflow_format(pred_last), t=int(init_time + time_step))
            k1 = to_numpy_format(k1)
            k1 += transformation(pred_last, model(to_model_format(pred_last / velocity_std)) * velocity_std)
            y_temp = pred_last + 0.5 * tf.cast(simulator.dt, tf.float32) * k1
            k2 = simulator.equation(to_phiflow_format(y_temp), t=int(init_time + (i + 0.5) * tf.cast(simulator.dt, tf.float32)))
            k2 = to_numpy_format(k2)
            k2 += transformation(y_temp, model(to_model_format(y_temp / velocity_std)) * velocity_std)

            y_temp = pred_last + 0.5 * tf.cast(simulator.dt, tf.float32) * k2
            k3 = simulator.equation(to_phiflow_format(y_temp), t=int(init_time + (i + 0.5) * tf.cast(simulator.dt, tf.float32)))
            k3 = to_numpy_format(k3)
            k3 += transformation(y_temp, model(to_model_format(y_temp / velocity_std)) * velocity_std)

            y_temp = pred_last + tf.cast(simulator.dt, tf.float32) * k3
            k4 = simulator.equation(to_phiflow_format(y_temp), t=int(init_time + (i + 1) * tf.cast(simulator.dt, tf.float32)))
            k4 = to_numpy_format(k4)
            k4 += transformation(y_temp, model(to_model_format(y_temp / velocity_std)) * velocity_std)

            prediction_flow = prediction[-1] + (tf.cast(simulator.dt, tf.float32) / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            prediction.append(prediction_flow)

        final_prediction = tf.stack(prediction[1:])
        loss = tf.reduce_mean(tf.abs(final_prediction - output))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print("Gradients:", gradients)
        return loss


# training_step_spatial = math.jit_compile(training_step_spatial)


# # Spatial Discretization Training Process - 
# def training_step_spatial(input, output, init_time, velocity_std, model, optimizer):
#     '''
#     Input Shape: (dim_x)
#     Output Shape: (k, dim_x)
#     '''
#     prediction, coefficients = [input], [0]

    
#     with tf.GradientTape() as tape:
#         for i in range(k):
            

#             ### RK4 INSIDE
#             k1 = simulator.equation(to_phiflow_format(prediction[-1]), t=init_time + i * simulator.dt)

#             print("????")
#             print(transformation(prediction[-1], model(to_model_format(prediction[-1]))))
#             print("????")
#             k1 = k1 + to_phiflow_format(transformation(prediction[-1], model(to_model_format(prediction[-1]))))
#             # k1 = to_numpy_format(k1)
#             print("!!!!!!!")
#             print(k1)
#             print("!!!!!!!")
#             y_temp = to_phiflow_format(prediction[-1]) + 0.5 * simulator.dt * k1

#             k2 = simulator.equation(y_temp, t=init_time + (i + 0.5) * simulator.dt)
#             k2 = k2 + to_phiflow_format(transformation(to_numpy_format(y_temp), model(to_model_format(to_numpy_format(y_temp)))))

#             y_temp = to_phiflow_format(prediction[-1]) + 0.5 * simulator.dt * k2
#             k3 = simulator.equation(y_temp, t=init_time + (i + 0.5) * simulator.dt)
#             k3 = k3 + to_phiflow_format(transformation(to_numpy_format(y_temp), model(to_model_format(to_numpy_format(y_temp)))))

#             y_temp = to_phiflow_format(prediction[-1]) + simulator.dt * k3
#             k4 = simulator.equation(y_temp, t=init_time + (i + 1) * simulator.dt)
#             k4 = k4 + to_phiflow_format(transformation(to_numpy_format(y_temp), model(to_model_format(to_numpy_format(y_temp)))))

#             k1 = to_numpy_format(k1)
#             k2 = to_numpy_format(k2)
#             k3 = to_numpy_format(k3)
#             k4 = to_numpy_format(k4)

#             prediction_flow = prediction[-1] + (simulator.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
#             #########

            
#             # prediction_flow = simulator.step_corrected(to_phiflow_format(prediction[-1]), t=init_time + i, model = model)
#             # prediction_flow = to_numpy_format(prediction_flow)

#             # prediction_flow_normalized = np.divide(prediction_flow, (velocity_std + 1e-6))

#             # prediction += [prediction_flow.values]
#             prediction += [prediction_flow]

#             # model_input = tf.expand_dims(tf.expand_dims(prediction_flow_normalized, 0), -1)  # Reshape to (1, 128, 1)

#             # model_output = model(model_input)
#             # model_output *= velocity_std
#             # coefficients += model_output

#             # correction = transformation(prediction[-1], model_output)
#             # prediction[-1] = prediction[-1] + correction

#         # Calculate the Loss Function using MAE

#         final_prediction = tf.stack(prediction[1:]) 

#         loss = tf.reduce_mean(tf.abs(final_prediction - output))  # Use MAE for loss
#         # Calculate the Gradients
#         gradients = tape.gradient(loss, model.trainable_variables)
#         print("Gradients:", gradients)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#         return loss



# def training_step_spatial(input, output, init_time, velocity_std, model, optimizer):
#     '''
#     Input Shape: (dim_x)
#     Output Shape: (k, dim_x)
#     '''
#     prediction = [input]
#     coefficients = [0]

#     with tf.GradientTape() as tape:
#         print("Initial input type and shape:", type(input), input.shape)

#         for i in range(k):
#             prediction_flow = simulator.step_corrected(to_phiflow_format(prediction[-1]), t=init_time + i, model=model)
#             print(f"Prediction flow {i} type and shape:", type(prediction_flow), prediction_flow.shape)

#             prediction_flow = to_tensor_format(prediction_flow)
#             print(f"Prediction flow {i} after conversion type and shape:", type(prediction_flow), prediction_flow.shape)

#             prediction.append(prediction_flow)

#         final_prediction = tf.stack(prediction[1:])
#         print("Final prediction type and shape:", type(final_prediction), final_prediction.shape)

#         loss = tf.reduce_mean(tf.abs(final_prediction - output))  # Use MAE for loss
#         print("Loss:", loss)

#     # Calculate the Gradients
#     gradients = tape.gradient(loss, model.trainable_variables)
#     for var, grad in zip(model.trainable_variables, gradients):
#         print(f"Variable: {var.name}, Gradient: {grad}")

#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#     return loss





# training_step = math.jit_compile(training_step)


batch_size = 32  # desired batch size

data_loader = DataLoader(simulation_path, batch_size, k)

num_coef = 8


# Convolutional Neural Network Model - 1-D
model = tf.keras.Sequential([
    # Input layer
    layers.InputLayer(input_shape=(128, 1)),
    
    # Convolutional layer 1
    layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    
    # Convolutional layer 2
    layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    
    # Convolutional layer 3
    layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    
    # Convolutional layer 4
    layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    
    # Flattening the output for the Dense layer
    layers.Flatten(),
    
    # Dense layer 1
    layers.Dense(256, activation='relu'),
    
    # Dense layer 2
    layers.Dense(128, activation='relu'),
    
    # Dense layer 3
    layers.Dense(64, activation='relu'),
    
    # Output layer
    layers.Dense(8)
])

# Compile the model (example for a regression problem)
model.compile(optimizer='adam')  # Use 'categorical_crossentropy' for classification

# # Summary of the model
# model.summary()






train_loss = tf.keras.metrics.Mean(name='train_loss')



# Define checkpoint directory
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)  # Create checkpoint directory if it doesn't exist


filepath = "./checkpoints/cp-{epoch:04d}.weights.h5"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,  # Adjusted file extension
    save_weights_only=True,
    verbose=1)
# Check if there are any checkpoints available and load the latest one
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

if latest_checkpoint:
    print(f"Loading weights from {latest_checkpoint}")
    model.load_weights(latest_checkpoint)
else:
    print("No checkpoints found, starting training from scratch.")


Epochs = 10

for epoch in range(Epochs):
    print(f"Epoch {epoch + 1}/{Epochs}")
    batches, batch_velocity_stds = data_loader.get_batches()
    for batch, velocity_std in zip(batches, batch_velocity_stds):
        input_batch, output_batch, sim_indices, time_indices = batch
        
        batch_losses = []
        
        for i in range(len(input_batch)):
            simulator.reset()
            simulator.change_forcing_dir(os.path.join(simulation_path, f'sim_{sim_indices[i]:06d}', 'params.json'))
            # Different Training Step for different configurations
            #loss = training_step(input_batch[i], output_batch[i], time_indices[i], velocity_std, model, optimizer)
            loss = training_step_spatial(input_batch[i], output_batch[i], time_indices[i], velocity_std, model, optimizer)
            batch_losses.append(loss)
            print("Batch", i + 1, "Loss:", loss.numpy())
        
        avg_batch_loss = tf.reduce_mean(batch_losses)
        print(f"Batch average loss: {avg_batch_loss:.8f}")
    
    print("End of epoch\n")
    
    # Save the model's weights at the end of each epoch
    model.save_weights(filepath.format(epoch=epoch))

# Optionally, save the final model
model.save('./nn_final.h5')