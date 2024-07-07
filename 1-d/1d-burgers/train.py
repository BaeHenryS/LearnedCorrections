import numpy as np
import os
from scipy.signal import convolve
import burgers_1d
from phi.flow import *

import tensorflow as tf
from tensorflow.keras import layers, models


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

        return num_simulations, num_timesteps

    def _load_velocity_data(self):
        data = []
        for sim in range(self.num_simulations):
            sim_data = []
            sim_dir = os.path.join(self.simulation_path, f'sim_{sim:06d}')
            for timestep in range(self.num_timesteps):
                file_path = os.path.join(sim_dir, f'velocity_{timestep:06d}.npz')
                with np.load(file_path) as npz_file:
                    sim_data.append(npz_file['data'])
            data.append(sim_data)
        return data

    def _get_velocity_std(self, sim_num): 
        # Calculate the standard deviation of the velocity
        velocity_std = np.std(self.data[sim_num])
        return velocity_std 

    def _construct_batches(self):
        input_data = []
        output_data = []
        sim_indices = []
        time_indices = []
        batch_velocity_std = []

        for sim_index, sim_data in enumerate(self.data):
            for i in range(len(sim_data) - self.k):
                input_data.append(sim_data[i])
                output_data.append(sim_data[i + 1: i + self.k + 1])
                sim_indices.append(sim_index)
                time_indices.append(i)

        # Create indices for shuffling
        indices = np.arange(len(input_data))
        np.random.shuffle(indices)

        # Shuffle data
        input_data = np.array(input_data)[indices]
        output_data = np.array(output_data)[indices]
        sim_indices = np.array(sim_indices)[indices]
        time_indices = np.array(time_indices)[indices]

        # Create batches
        num_batches = len(input_data) // self.batch_size
        batches = []
        for i in range(num_batches):
            batch_input = input_data[i * self.batch_size: (i + 1) * self.batch_size]
            batch_output = output_data[i * self.batch_size: (i + 1) * self.batch_size]
            batch_sim_indices = sim_indices[i * self.batch_size: (i + 1) * self.batch_size]
            batch_time_indices = time_indices[i * self.batch_size: (i + 1) * self.batch_size]
            batches.append((batch_input, batch_output, batch_sim_indices, batch_time_indices))

            # Calculate and store standard deviation of the velocity of the batch
            velocity_std = np.std(batch_input)
            batch_velocity_std.append(velocity_std)

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
k = 5  # future timestep interval

LR = 1e-4

optimizer = tf.keras.optimizers.Adam(learning_rate=LR) 




import burgers_1d
from phi.flow import *
import tensorflow as tf

simulator = burgers_1d.Burgers_1d()

def to_phiflow_format(data):
    return CenteredGrid(math.tensor(data, spatial('x')), extrapolation.PERIODIC, x=int(128), bounds=Box['x', slice(0, 2 * np.pi)])

def to_numpy_format(data):
    return data.values.numpy('x')

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


def training_step(input, output, simulationnumber, init_time, velocity_std, model, optimizer):
    '''
    Input Shape: (dim_x)
    Output Shape: (k, dim_x)
    '''
    simulator.reset()


    

    simulator.change_forcing_dir(os.path.join(simulation_path, f'sim_{simulationnumber:06d}', 'params.json'))
    

    prediction, coefficients = [input], [0]

    with tf.GradientTape() as tape:
        for i in range(k):
            prediction_flow = simulator.step(to_phiflow_format(prediction[-1]), t=init_time + i)
            prediction_flow = to_numpy_format(prediction_flow)

            prediction_flow_normalized = np.divide(prediction_flow, (velocity_std + 1e-6))
    
            prediction += [prediction_flow]

            model_input = tf.expand_dims(tf.expand_dims(prediction_flow_normalized, 0), -1)  # Reshape to (1, 128, 1)
            # Get shape of model_input

            model_output = model(model_input)
            coefficients += model_output

            correction = transformation(prediction[-1], model_output)
            #prediction[-1] = prediction[-1] + tf.reshape(model_output, [-1])
            prediction[-1] = prediction[-1] + correction

        # Calculate the Loss Function
        final_prediction = tf.stack(prediction[1:]) 
    
        loss = tf.reduce_sum(tf.square(final_prediction - output))
        # Calculate the Gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss
    






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

# Define checkpoint file path
checkpoint_path = os.path.join(checkpoint_dir, "cp-{epoch:04d}.ckpt")

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq='epoch')  # Save weights, every epoch


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
            loss = training_step(input_batch[i], output_batch[i], sim_indices[i], time_indices[i], velocity_std, model, optimizer)
            batch_losses.append(loss)
        
        avg_batch_loss = tf.reduce_mean(batch_losses)
        print(f"Batch average loss: {avg_batch_loss:.8f}")
    
    print("End of epoch\n")
    
    # Save the model's weights at the end of each epoch
    model.save_weights(checkpoint_path.format(epoch=epoch))

# Optionally, save the final model
model.save('./nn_final.h5')