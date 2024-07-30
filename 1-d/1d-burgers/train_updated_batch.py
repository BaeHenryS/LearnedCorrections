import numpy as np
import os
from scipy.signal import convolve
import burgers_1d_batch
from phi.tf.flow import *

import tensorflow as tf
from tensorflow.keras import layers, models
import concurrent.futures

import random
import pickle
import json


simulation_path = './output'  # path to the directory containing the simulations
k = 5  # future timestep interval

LR = 1e-4
batch_size = 128

optimizer = tf.keras.optimizers.Adam(learning_rate=LR) 


simulator = burgers_1d_batch.Burgers_1d(resolution=128, output='../test_output', batch_size = batch_size)

def to_phiflow_format(data):
    # Check if data is already a phi tensor
    if isinstance(data, CenteredGrid):
        return data
    else:
        # Then convert TensorFlow tensor to phi tensor
        grid = CenteredGrid(math.tensor(data, spatial('x')), extrapolation.PERIODIC, x=int(128), bounds=Box['x', slice(0, 2 * np.pi)])
        return grid


def to_numpy_format(data):
    return data.values.numpy(order=('batch', 'x'))

# @tf.function
# def to_numpy_format(data):
#     def convert_to_numpy(data):
#         return data.values(order=('batch', 'x'))

    return tf.py_function(convert_to_numpy, [data], tf.float32)

def to_tensor_format(data):
    return tf.convert_to_tensor(data.values.numpy('x'))



def to_model_format(data):
    # Assuming data is of shape (32, 128), add a dimension at the end to get (32, 128, 1)
    return tf.expand_dims(data, -1)

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



class DataLoader:
    def __init__(self, simulation_path, k):
        self.simulation_path = simulation_path
        self.k = k
        self.num_simulations, self.num_timesteps = self._count_simulations_and_timesteps()
        self.data = None
        self.batched_data = None
        self.batched_std = None
        

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

            with concurrent.futures.ThreadPoolExecutor() as executor:
                data = list(executor.map(load_data_with_progress, range(self.num_simulations)))
            # Filter out empty arrays from each array in data before flattening
            return [element for array in data if array for element in array]                

    def _load_simulation_data(self, sim):
        sim_dir = os.path.join(self.simulation_path, f'sim_{sim:06d}')
        # Load params.json in the simulation directory
        params_file = os.path.join(sim_dir, 'params.json')
        if not os.path.exists(params_file):
            print(f"Params file {params_file} not found, skipping to the next simulation.")
            return []
        
        params_file = json.load(open(params_file))

        A_values = np.array(params_file["A_values"])
        ω_values = np.array(params_file["ω_values"])
        φ_values = np.array(params_file["φ_values"])
        l_values = np.array(params_file["l_values"])
        combined_array = np.column_stack((A_values, ω_values, φ_values, l_values))


        if not os.path.exists(sim_dir):
            print(f"Simulation directory {sim_dir} not found, skipping to the next simulation.")
            return []  # Return empty list for missing simulations
        sim_data = []
        for t in range(self.num_timesteps - self.k-1):
            try:
                velocity_data = np.stack([np.load(os.path.join(sim_dir, f'velocity_{(t + i):06d}.npz'))['data'] for i in range(0, self.k + 1)], axis=0)
                advection_diffusion_data = np.stack([np.load(os.path.join(sim_dir, f'advection_diffusion_{(t + i):06d}.npz'))['data'] for i in range(0, self.k+1)], axis=0)


                sim_data.append((sim, velocity_data, advection_diffusion_data, t, combined_array))

            except FileNotFoundError:
                print(f"File not found for simulation {sim}, timestep {t}, skipping to the next timestep.")


        print(f"Loaded simulation {sim + 1}/{self.num_simulations}")
        return sim_data
    


    def _compute_batch_std(self, batch_data):
        velocity_data_list = []
        advection_diffusion_data_list = []

        for sim, velocity_data, advection_diffusion_data, _, _ in batch_data:
            velocity_data_list.append(velocity_data)
            advection_diffusion_data_list.append(advection_diffusion_data)

        # Convert lists to numpy arrays
        velocity_data_array = np.array(velocity_data_list)
        advection_diffusion_data_array = np.array(advection_diffusion_data_list)

        # Check and handle NaNs
        if np.isnan(velocity_data_array).any() or np.isnan(advection_diffusion_data_array).any():
            print("Warning: NaN values found in the data. Ignoring NaNs in standard deviation computation.")
            velocity_std = np.nanstd(velocity_data_array.flatten())
            advection_diffusion_std = np.nanstd(advection_diffusion_data_array.flatten())
        else:
            velocity_std = np.std(velocity_data_array.flatten())
            advection_diffusion_std = np.std(advection_diffusion_data_array.flatten())

        return velocity_std, advection_diffusion_std

    def prepare_batches(self, batch_size):
        self.batched_data = []
        self.batched_std = []

        random.shuffle(self.data)  # Shuffle the data before creating batches
        total_batches = int(math.ceil(len(self.data) / batch_size))
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_data = self.data[start_idx:end_idx]

            
            self.batched_data.append(batch_data)
            self.batched_std.append(self._compute_batch_std(batch_data))

        return self.batched_data
    
    def save_data_as_pickle(self, data, filename):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
    
    def load_from_pickle(self, filename):
        with open(filename, 'rb') as file:
            self.data = pickle.load(file)
        return self.data
    
    
    def _remove_nans_from_data(self):
        cleaned_data = []
        for element in self.data:
            sim, velocity_data, advection_diffusion_data, t, combined_array = element
            # Check for NaNs in velocity_data and advection_diffusion_data
            if not (np.isnan(velocity_data).any() or np.isnan(advection_diffusion_data).any()):
                cleaned_data.append(element)
            else:
                print(f"Removing element with NaNs: Simulation {sim}, Timestep {t}")
        
        self.data = cleaned_data
        return cleaned_data




## TESTING
# import time 
# def training_step_spatial(velocity, advection_diffusion, init_time, std, model, optimizer):
#     velocity_std, advection_diffusion_std = std
#     input = velocity[0]
#     output = velocity[1:]
    
#     prediction = [input]
#     # Ensure init_time is of type float32 or float64
#     init_time = tf.cast(init_time, tf.float32)

#     # with tf.GradientTape(persistent=True) as tape:
#     with tf.GradientTape() as tape:
#         loss_advection_diffusion = 0

#         start = time.time()
#         for i in range(k):
#             pred_last = prediction[-1]

#             # Cast simulator.dt to float32 and perform arithmetic operation
#             time_step = tf.cast(i, tf.float32) * tf.cast(simulator.dt, tf.float32)
#             k1 = simulator.equation(to_phiflow_format(pred_last), t=int(init_time + time_step))
            
#             k1 = to_numpy_format(k1)
#             k1 += transformation(pred_last, model(to_model_format(pred_last / velocity_std)) * velocity_std)
#             y_temp = pred_last + 0.5 * tf.cast(simulator.dt, tf.float32) * k1
#             k2 = simulator.equation(to_phiflow_format(y_temp), t=int(init_time + (i + 0.5) * tf.cast(simulator.dt, tf.float32)))
#             k2 = to_numpy_format(k2)
#             k2 += transformation(y_temp, model(to_model_format(y_temp / velocity_std)) * velocity_std)

#             y_temp = pred_last + 0.5 * tf.cast(simulator.dt, tf.float32) * k2
#             k3 = simulator.equation(to_phiflow_format(y_temp), t=int(init_time + (i + 0.5) * tf.cast(simulator.dt, tf.float32)))
#             k3 = to_numpy_format(k3)
#             k3 += transformation(y_temp, model(to_model_format(y_temp / velocity_std)) * velocity_std)

#             y_temp = pred_last + tf.cast(simulator.dt, tf.float32) * k3
#             k4 = simulator.equation(to_phiflow_format(y_temp), t=int(init_time + (i + 1) * tf.cast(simulator.dt, tf.float32)))
#             k4 = to_numpy_format(k4)
#             k4 += transformation(y_temp, model(to_model_format(y_temp / velocity_std)) * velocity_std)

#             prediction_flow = prediction[-1] + (tf.cast(simulator.dt, tf.float32) / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
#             prediction.append(prediction_flow)
#             print("Hi")
#         end = time.time()
#         print("Time taken: ", end - start)

#         start = time.time()
#         final_prediction = tf.stack(prediction[1:])
#         loss = tf.reduce_mean(tf.abs(final_prediction - output))
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#         end = time.time()
#         print("Time taken: ", end - start)

#         return loss



def transformation_batch(velocity_functions, output_datas):
    """
    Apply a 1D convolution transformation to a batch of velocity functions with a batch of output data.
    
    Parameters:
    - velocity_functions: A batch of velocity functions, shape (batch_size, length, 1).
    - output_datas: A batch of output data for convolution filters, shape (batch_size, filter_length, 1, 1).
    
    Returns:
    - A batch of convolved data, each with shape (length,).
    """
    # Ensure velocity_functions is 4D for batched 1D convolution: (batch_size, width, in_channels, out_channels)
    velocity_functions = tf.reshape(velocity_functions, [velocity_functions.shape[0], -1, 1, 1])
    
    # Ensure output_datas is 4D for filters in batched 1D convolution: (batch_size, filter_width, in_channels, out_channels)
    output_datas = tf.reshape(output_datas, [output_datas.shape[0], -1, 1, 1])
    
    # Perform the batched 1D convolution
    convolved_datas = tf.nn.depthwise_conv2d(velocity_functions, output_datas, strides=[1, 1, 1, 1], padding='SAME')
    
    # Squeeze the output to remove unnecessary dimensions, resulting in a batch of shape (batch_size, length)
    return tf.squeeze(convolved_datas, axis=[2, 3])


# def to_phiflow_format_batch(datas):
#     """
#     Convert a batch of data to a batch of PhiFlow CenteredGrid objects and stack them.
    
#     Parameters:
#     - datas: A batch of data, shape (batch_size, length).
    
#     Returns:
#     - A stacked CenteredGrid object representing the batch.
#     """
#     # Assuming to_phiflow_format has been modified to return a tensor representation of CenteredGrid
#     # Use tf.map_fn with fn_output_signature to specify the expected output structure
#     grids = []
#     for data in datas:
#         grid = CenteredGrid(math.tensor(data, spatial('x'),), extrapolation.PERIODIC, x=int(128), bounds=Box['x', slice(0, 2 * np.pi)])
#         grids.append(grid)
    
#     # Assuming there's a way to convert the tensor representation back to CenteredGrid objects if necessary
#     # Stack the CenteredGrid objects to handle the batch dimension
#     # This step may need to be adjusted based on how CenteredGrid objects are represented as tensors
#     stacked_grid = phi.tf.flow.stack(grids, phi.tf.flow.batch('batch'))
#     return stacked_grid

def to_phiflow_format_batch(datas):
        return CenteredGrid(math.tensor(datas, phi.tf.flow.batch('batch'),spatial('x')), 
                        extrapolation.PERIODIC, 
                        x=128,
                        bounds=Box['x', slice(0, 2 * np.pi)])
import time
def training_step_spatial_batch(velocity, advection_diffusion, init_time, std, model, optimizer):

    velocity_std, advection_diffusion_std = std
    # input = velocity[0]
    # output = velocity[1:]
    

    input = velocity[:,0,:]
    output = velocity[:,1:,:]
    prediction = [input]
    # Ensure init_time is of type float32 or float64
    # with tf.GradientTape(persistent=True) as tape:

    with tf.GradientTape() as tape:
        loss_advection_diffusion = 0
        for i in range(k):
            
            pred_last = prediction[-1]

            # Assuming init_time is a 1D tensor or array with the same first dimension as prediction
            time_step = tf.cast(i, tf.float32) * tf.cast(simulator.dt, tf.float32)
            time_steps = init_time + time_step  # Element-wise addition for each element in init_time
            start = time.time()
            k1 = simulator.equation_batch(to_phiflow_format_batch(pred_last), t=tf.cast(time_steps, tf.int32))
            print(time.time() - start)
            #print(k1)
            k1 = to_numpy_format(k1)
            
            k1 += transformation_batch(pred_last, model(to_model_format(pred_last / velocity_std)) * velocity_std)
            # print(velocity_std)
            # print(k1)
            y_temp = pred_last + 0.5 * tf.cast(simulator.dt, tf.float32) * k1

            # Adjust time step for k2
            time_steps_k2 = init_time + (i + 0.5) * tf.cast(simulator.dt, tf.float32)
            k2 = simulator.equation_batch(to_phiflow_format_batch(y_temp), t=tf.cast(time_steps_k2, tf.int32))
            k2 = to_numpy_format(k2)
            k2 += transformation_batch(y_temp, model(to_model_format(y_temp / velocity_std)) * velocity_std)

            y_temp = pred_last + 0.5 * tf.cast(simulator.dt, tf.float32) * k2

            # Adjust time step for k3 (same as k2 since it's also a mid-step)
            k3 = simulator.equation_batch(to_phiflow_format_batch(y_temp), t=tf.cast(time_steps_k2, tf.int32))
            k3 = to_numpy_format(k3)
            k3 += transformation_batch(y_temp, model(to_model_format(y_temp / velocity_std)) * velocity_std)

            y_temp = pred_last + tf.cast(simulator.dt, tf.float32) * k3

            # Adjust time step for k4
            time_steps_k4 = init_time + (i + 1) * tf.cast(simulator.dt, tf.float32)
            k4 = simulator.equation_batch(to_phiflow_format_batch(y_temp), t=tf.cast(time_steps_k4, tf.int32))
            k4 = to_numpy_format(k4)
            k4 += transformation_batch(y_temp, model(to_model_format(y_temp / velocity_std)) * velocity_std)

            prediction_flow = prediction[-1] + (tf.cast(simulator.dt, tf.float32) / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        
        final_prediction = tf.stack(prediction[1:])
        final_prediction = tf.transpose(final_prediction, perm=[1, 0, 2])

        # print(final_prediction.shape)
        # print(output.shape)       
        #loss = tf.reduce_mean(tf.abs(final_prediction - output))
        loss = tf.reduce_sum(tf.square(final_prediction - output))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


        return loss


# training_step_spatial_batch = math.jit_compile(training_step_spatial_batch)



  # desired batch size

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
model.compile(optimizer='adam')  

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



data_loader = DataLoader(simulation_path, k)

#data_loader._load_velocity_data()
#Only for k = 5
data_loader.load_from_pickle('data.pkl')
data_loader._remove_nans_from_data()
print("Data loaded")

Epochs = 10


for epoch in range(Epochs):
    print(f"Epoch {epoch + 1}/{Epochs}")

    data = data_loader.prepare_batches(batch_size)

    batch_losses = []
    print(f"Number of batches: {len(data)}")
    # Ensure the losses directory exists
    losses_dir = './losses/'
    if not os.path.exists(losses_dir):
        os.makedirs(losses_dir)

    # Ensure the models directory exists
    models_dir = './models/'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    print(f"Number of batches: {len(data)}")
    for i, batch in enumerate(data):
        batch_std = data_loader.batched_std[i]
        simulator.set_batch_params([item[4] for item in batch])

        input_1 = np.stack([item[1] for item in batch], axis=0)
        input_2 = np.stack([item[2] for item in batch], axis=0)

        loss = training_step_spatial_batch(input_1, input_2, [item[3] for item in batch], batch_std, model, optimizer)  
        batch_losses.append(loss.numpy())  
        print(loss.numpy())
        if (i + 1) % 100 == 0:  # Every 100 batches
            # Save loss profiles
            batch_losses_serializable = [float(loss) for loss in batch_losses]
            with open(f'{losses_dir}loss_profile_epoch_{epoch+1}_batch_{i+1}.json', 'w') as f:
                json.dump(batch_losses_serializable, f)
            
            # Save model weights
            model.save_weights(f'{models_dir}model_weights_epoch_{epoch+1}_batch_{i+1}.weights.h5')

            print(f"Saved loss profile and model weights at batch {i + 1}")


    # Save the model's weights at the end of each epoch
    model.save_weights(filepath.format(epoch=epoch))

model.save_weights('./nn_final.h5')   

        
    
