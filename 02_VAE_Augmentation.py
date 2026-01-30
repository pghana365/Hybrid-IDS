import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

#Loading Preprocessed Data
input_file = 'processed_data.npz'
output_file = 'balanced_dataset.npz'

print("Loading data artifacts...")

if not os.path.exists(input_file):
    raise FileNotFoundError("Error: Preprocessed file not found. Run the first notebook again.")

data = np.load(input_file)
X_train = data['X_train']
y_train = data['y_train']

# Filter for Attacks
# here we only want to train the VAE on the Attack class (Label 1) so it learns to generate them.
print("Separating attack samples...")
X_attacks = X_train[y_train == 1]
X_normal = X_train[y_train == 0]

print(f"Attack samples: {X_attacks.shape[0]}")
print(f"Normal samples: {X_normal.shape[0]}")

# Build the VAE Model
# Configuring parameters
input_dim = X_train.shape[1]  # Should be 10 features
latent_dim = 2  # Compressing down to 2 variables
batch_size = 64
epochs = 50

# Encoder: Compresses input to Mean and Variance
encoder_inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(16, activation="relu")(encoder_inputs)
x = layers.Dense(8, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

# Sampling Layer: The "Reparameterization Trick" needed for VAEs
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder: Reconstructs the data from the latent space
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8, activation="relu")(latent_inputs)
x = layers.Dense(16, activation="relu")(x)
decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x) # Output is 0-1 (matching our scaled data)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# Custom VAE Class to handle the loss calculation
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Loss Function: Reconstruction + KL Divergence
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= input_dim
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
            
        # Backprop
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss}

# Training 
print("Starting training on Attack data...")
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

history = vae.fit(X_attacks, epochs=epochs, batch_size=batch_size, verbose=1)

#Data Augmentation
# Calculate how many samples needed to balance the dataset (1:1 ratio)
needed = len(X_normal) - len(X_attacks)
print(f"Generating {needed} synthetic attacks...")

# Generate random points in latent space and decode them
z_sample = np.random.normal(size=(needed, latent_dim))
X_synthetic = decoder.predict(z_sample)

# Merging and Saving the data
print("Merging real and synthetic data...")
X_balanced = np.concatenate([X_train, X_synthetic])
# Assign Label 1 to the new synthetic attacks
y_balanced = np.concatenate([y_train, np.ones(needed)])

np.savez(output_file, X_balanced=X_balanced, y_balanced=y_balanced)

print(f"Success. Balanced dataset saved to {output_file}")
print(f"Final Normal count: {sum(y_balanced==0)}")
print(f"Final Attack count: {sum(y_balanced==1)}")
