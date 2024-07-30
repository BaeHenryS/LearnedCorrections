import numpy as np
import os
from scipy.signal import convolve
import burgers_1d
from phi.tf.flow import *

import tensorflow as tf
from tensorflow.keras import layers, models
import concurrent.futures

import random
import pickle


simulation_path = './output'  # path to the directory containing the simulations
k = 5  # future timestep interval

LR = 1e-4

optimizer = tf.keras.optimizers.Adam(learning_rate=LR) 


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


class DataLoader:
    def __init__(self, simulation_path, k):
        self.simulation_path = simulation_path
        self.k = k
        self.num_simulations, self.num_timesteps = self._count_simulations_and_timesteps()
        # Changed to load data later. 
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

        # Specify max_workers. Adjust the number based on your system's capabilities
        #num_workers = 20  # Example: Set this to a higher number based on your system's capabilities

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Use list to preserve order, map through the simulations with their indices
            data = list(executor.map(load_data_with_progress, range(self.num_simulations)))
        return [element for array in data for element in array]

    def _load_simulation_data(self, sim):
        sim_dir = os.path.join(self.simulation_path, f'sim_{sim:06d}')
        if not os.path.exists(sim_dir):
            print(f"Simulation directory {sim_dir} not found, skipping to the next simulation.")
            return []  # Return empty list for missing simulations
        sim_data = []
        for t in range(self.num_timesteps - self.k-1):
            try:
                velocity_data = np.stack([np.load(os.path.join(sim_dir, f'velocity_{(t + i):06d}.npz'))['data'] for i in range(0, self.k + 1)], axis=0)
                advection_diffusion_data = np.stack([np.load(os.path.join(sim_dir, f'advection_diffusion_{(t + i):06d}.npz'))['data'] for i in range(0, self.k+1)], axis=0)


                sim_data.append((sim, velocity_data, advection_diffusion_data, t))

            except FileNotFoundError:
                print(f"File not found for simulation {sim}, timestep {t}, skipping to the next timestep.")


        print(f"Loaded simulation {sim + 1}/{self.num_simulations}")
        return sim_data
    
    def _compute_batch_std(self, batch_data):

        velocity_data_list = []
        advection_diffusion_data_list = []

        for sim, velocity_data, advection_diffusion_data, timestep in batch_data:
            velocity_data_list.append(velocity_data)
            advection_diffusion_data_list.append(advection_diffusion_data)

        # Convert lists to numpy arrays
        velocity_data_array = np.array(velocity_data_list)
        advection_diffusion_data_array = np.array(advection_diffusion_data_list)


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
    

    def load_from_pickle(self, filename):
        self.data = np.load(filename, allow_pickle=True)
        
    




## TESTING

def training_step_spatial(velocity, advection_diffusion, init_time, std, model, optimizer):
    velocity_std, advection_diffusion_std = std
    input = velocity[0]
    output = velocity[1:]
    
    prediction = [input]
    # Ensure init_time is of type float32 or float64
    init_time = tf.cast(init_time, tf.float32)

    # with tf.GradientTape(persistent=True) as tape:
    with tf.GradientTape() as tape:
        loss_advection_diffusion = 0
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


        return loss


# training_step_spatial = math.jit_compile(training_step_spatial)



batch_size = 32  # desired batch size

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
print("Data loaded")

Epochs = 10

for epoch in range(Epochs):
    print(f"Epoch {epoch + 1}/{Epochs}")

    data = data_loader.prepare_batches(batch_size)

    batch_losses = []
    for i, batch in enumerate(data):
        batch_std = data_loader.batched_std[i]

        for j in range(len(batch)):
            simulator.reset()
            simulator.change_forcing_dir(os.path.join(simulation_path, f'sim_{batch[j][0]:06d}', 'params.json'))
            loss = training_step_spatial(batch[j][1], batch[j][2], batch[j][3], batch_std, model, optimizer)
            
            batch_losses.append(loss)

        avg_batch_loss = tf.reduce_mean(batch_losses)
        print(f"Batch average loss: {avg_batch_loss:.8f}")

    print("End of epoch\n")

    # Save the model's weights at the end of each epoch
    model.save_weights(filepath.format(epoch=epoch))

model.save_weights('./nn_final.h5')   

        
    
