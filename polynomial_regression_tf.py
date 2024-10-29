import os
import numpy as np
import tensorflow as tf
from datetime import datetime  # Import datetime for timestamps

class CubicPolynomialModel:
    def __init__(self, num_points=1000, sigma=0.1, learning_rate=0.01, batch_size=32):
        # Generate synthetic data
        self.x_values, self.y_values = self.generate_data(num_points, sigma)
        self.batch_size = batch_size

        # Define TensorFlow variables for the cubic polynomial coefficients (a, b, c, d)
        self.a = tf.Variable(initial_value=tf.random.normal([]), dtype=tf.float32, name='a')
        self.b = tf.Variable(initial_value=tf.random.normal([]), dtype=tf.float32, name='b')
        self.c = tf.Variable(initial_value=tf.random.normal([]), dtype=tf.float32, name='c')
        self.d = tf.Variable(initial_value=tf.random.normal([]), dtype=tf.float32, name='d')

        # Optimizer for training
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    def generate_data(self, num_points, sigma):
        """Generate synthetic cubic data with Gaussian noise."""
        x = np.linspace(start=-1, stop=1, num=num_points).astype(np.float32)
        # Define true cubic polynomial: y = 2x^3 - x^2 + x - 5 + noise
        y_true = 2 * (x ** 3) - (x ** 2) + x - 5
        noise = np.random.normal(0, sigma, size=num_points).astype(np.float32)
        y_observed = y_true + noise
        return x, y_observed

    def cubic_model(self, x):
        """TensorFlow model: y = ax^3 + bx^2 + cx + d"""
        return self.a * x ** 3 + self.b * x ** 2 + self.c * x + self.d

    def loss_fn(self, y_true, y_pred):
        """Mean Squared Error (MSE) loss function."""
        return tf.reduce_mean(tf.square(y_true - y_pred))

    @tf.function
    def train_step(self, x_batch, y_batch):
        """Perform one training step on a batch of data."""
        with tf.GradientTape() as tape:
            y_pred = self.cubic_model(x_batch)
            loss = self.loss_fn(y_batch, y_pred)
        gradients = tape.gradient(loss, [self.a, self.b, self.c, self.d])
        self.optimizer.apply_gradients(zip(gradients, [self.a, self.b, self.c, self.d]))
        return loss

    def ensure_directory_exists(self, file_path):
        """Ensure the directory exists for a given file path."""
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def get_timestamp(self):
        """Return the current timestamp in a readable format."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def train(self, n_epochs=1000, log_file="training_log.txt"):
        """Training loop with batch processing and logging to a file."""
        # Ensure the directory for the log file exists
        self.ensure_directory_exists(log_file)

        dataset = tf.data.Dataset.from_tensor_slices((self.x_values, self.y_values))
        dataset = dataset.shuffle(buffer_size=1024).batch(self.batch_size)

        # Open the log file in append mode
        with open(log_file, 'a') as file:
            file.write(f"\nTraining started at {self.get_timestamp()}\n")
            file.write("Epoch\tLoss\n")

            for epoch in range(n_epochs):
                for x_batch, y_batch in dataset:
                    loss_value = self.train_step(x_batch, y_batch)

                # Log the loss at every 100 epochs
                if epoch % 100 == 0:
                    log_message = f"Epoch {epoch}, Loss: {loss_value.numpy()}\n"
                    print(log_message.strip())  # Still print to console
                    file.write(f"{self.get_timestamp()}\t{epoch}\t{loss_value.numpy()}\n")

            file.write(f"Training ended at {self.get_timestamp()}\n")

    def save_learned_parameters(self, param_file="learned_params.txt"):
        """Save the learned parameters a, b, c, and d to a file."""
        # Ensure the directory for the parameter file exists
        self.ensure_directory_exists(param_file)

        with open(param_file, 'a') as file:  # Use append mode ('a')
            file.write(f"\nLearned parameters at {self.get_timestamp()} (New Run):\n")
            file.write(f"a = {self.a.numpy()}\n")
            file.write(f"b = {self.b.numpy()}\n")
            file.write(f"c = {self.c.numpy()}\n")
            file.write(f"d = {self.d.numpy()}\n")

        print(f"Learned parameters appended to {param_file}")
