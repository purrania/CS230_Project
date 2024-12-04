# Import necessary libraries
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar100, cifar10, fashion_mnist
import tensorflow_datasets as tfds

GROWTH_INTERVAL = 5
GROWTH_STAGES = 5
EPOCHS = GROWTH_INTERVAL * GROWTH_STAGES

# Load CIFAR-10 dataset
#dataset= "cifar-10"
#dataset = "cifar-100"
dataset = "cifar-100"
if dataset == "cifar-10":
    size = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
elif dataset == "cifar-100":
    size = 100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
elif dataset == "fashion_mnist":
    size = 10
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

else:
    raise ValueError("Invalid dataset name. Choose from 'cifar-10', 'cifar-100', or 'fashion_mnist'.")
# for sanity-checking/bug-cheking, can increase n for faster checking
n = 1
x_train = x_train[:len(x_train)//n]
y_train = y_train[:len(y_train)//n]


# Normalize the data

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to categorical (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, size)
y_test = tf.keras.utils.to_categorical(y_test, size)

input_shape = (32,32,3)
if dataset == "fashion_mnist": # because it is grayscale
  input_shape = (28, 28, 1)

# Define a simple model for CIFAR-10 (starting point)

def create_base_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(20, activation='relu'),
        layers.Dense(size, activation='softmax')
    ])

    return model

# Build the base model
base_model = create_base_model()
base_model.summary()

# Compile the model
base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def custom_loss_with_regularization(base_loss, model, lambda_depth=0.01, lambda_width=0.01):
    """
    Custom loss function that includes regularization terms for depth and width.

    Args:
    - base_loss: TensorFlow/Keras loss function (e.g., categorical_crossentropy).
    - model: Keras model whose structure is being regularized.
    - lambda_depth: Weight for the depth regularization term.
    - lambda_width: Weight for the width regularization term.

    Returns:
    - A loss function that combines the base loss with depth and width regularization.
    """
    def loss_function(y_true, y_pred):
        # Base task-specific loss
        task_loss = base_loss(y_true, y_pred)

        # Depth Regularization: Penalize the total number of layers
        depth = len([layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)])
        depth_regularization = lambda_depth * depth

        # Width Regularization: Penalize the total number of neurons in the model
        width = sum(layer.units for layer in model.layers if isinstance(layer, tf.keras.layers.Dense))
        width_regularization = lambda_width * width

        # Combine all loss terms
        total_loss = task_loss + depth_regularization + width_regularization
        return total_loss

    return loss_function

# Function to check if a layer needs pruning based on various metrics
def needs_pruning(model, x_train, layer_idx, gradient_threshold=0.01, activation_threshold=0.1, weight_threshold=0.1):
    """
    Determines if a layer is prunable based on gradient, activation, and weight metrics.
    """
    layer = model.layers[layer_idx]

    # Ensure x_train is a Tensor
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)

    # 1. Check gradients
    gradients = get_layer_gradients(model, x_train)
    if np.max(np.abs(gradients[layer_idx])) < gradient_threshold:
        return True  # Prune if gradients are too small

    # 2. Check activations
    activations = get_layer_activations(model, x_train)
    if np.mean(np.abs(activations[layer_idx])) < activation_threshold:
        return True  # Prune if activation is too small

    # 3. Check weights (magnitude of weights)
    weights = layer.get_weights()[0]  # Weights for dense layers
    if np.mean(np.abs(weights)) < weight_threshold:
        return True  # Prune if weights are too small

    return False

# Helper function to get gradients for each layer
def get_layer_gradients(model, x_train):
    """
    Calculates gradients for each layer of the model with respect to a batch of inputs.
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_train)  # Watch the input tensor
        predictions = model(x_train, training=True)
    gradients = tape.gradient(predictions, model.trainable_variables)
    return gradients

# Helper function to get activations for each layer
def get_layer_activations(model, x_train):
    """
    Calculates activations for each layer of the model for a batch of inputs.
    """
    activations = []
    for layer in model.layers:
        activations.append(layer(x_train, training=False).numpy())  # Extract activations
    return activations

# Function to rebuild the model after pruning a layer
def rebuild_model_after_pruning(model, layer_idx_to_remove):
    """
    Rebuilds the model after pruning a layer. It removes the specified layer.
    """
    new_model = tf.keras.Sequential()

    input_layer = model.layers[0]  # Preserve the input layer
    new_model.add(input_layer)

    # Add layers from the original model except the one to be pruned
    for i, layer in enumerate(model.layers[1:]):  # Start from 1 to skip input layer
        if i != layer_idx_to_remove:
            new_model.add(layer)

    # After pruning, ensure the output layer has the correct number of units (e.g., 10 for CIFAR-10)
    if isinstance(new_model.layers[-1], tf.keras.layers.Dense):
        new_model.add(tf.keras.layers.Dense(10, activation='softmax'))  # Recreate output layer with 10 units

    # Ensure the model is compiled after changes
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return new_model

# Example of pruning during training
def prune_model(model, x_train, y_train, epochs=10, pruning_interval=5):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train the model for one epoch
        model.fit(x_train, y_train, epochs=1)

        # After every 'pruning_interval' epochs, check for pruning
        if (epoch + 1) % pruning_interval == 0:
            print(f"Pruning after epoch {epoch + 1}...")

            for layer_idx in range(len(model.layers)):
                if len(model.layers)-1 > layer_idx and isinstance(model.layers[layer_idx], tf.keras.layers.Dense):
                    if needs_pruning(model, x_train[:32], layer_idx):  # Check for pruning condition
                        print(f"Layer {layer_idx} is prunable.")

                        # Rebuild the model after pruning the layer
                        model = rebuild_model_after_pruning(model, layer_idx)
        print(model.summary())
        print("-" * 50)
    return model

# Function to compute activation variance for a given layer
def compute_activation_variance(model, layer_idx, x_data):
    # Create a temporary model to extract activations
    temp_model = tf.keras.Model(inputs=model.input, outputs=model.layers[layer_idx].output)
    activations = temp_model(x_data)
    variance = tf.reduce_mean(tf.math.reduce_variance(activations, axis=0))  # Variance of activations across neurons
    return variance.numpy()

# Function to compute gradient flow (simplified version)
def compute_gradient_flow(model, x_data, y_data):
    with tf.GradientTape() as tape:
        predictions = model(x_data)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_data, predictions))
    grads = tape.gradient(loss, model.trainable_variables)
    grad_norms = [tf.reduce_mean(tf.abs(grad)) for grad in grads if grad is not None]
    return grad_norms

# Function to check if a layer needs growth
def needs_growth(model, x_data, layer_idx, threshold=0.05):
    # Get variance and gradient flow for the layer
    activation_variance = compute_activation_variance(model, layer_idx, x_data)
    grad_flow = compute_gradient_flow(model, x_data, y_train)

    # If variance or gradient flow is below threshold, it's a sign the layer needs growth
    if activation_variance < threshold or np.mean(grad_flow) < threshold:
        return True
    return False

# Function to add neurons to a target layer (example for Dense layer)
def add_neurons_to_layer(model, layer_idx, num_neurons=size):
    # Get the current layer and add neurons
    layer = model.layers[layer_idx]
    if isinstance(layer, layers.Dense):
        new_units = layer.units + num_neurons
        new_layer = layers.Dense(new_units, activation='relu', name=f'new_{layer.name}')
        model.layers[layer_idx] = new_layer
    elif isinstance(layer, layers.Conv2D):
        new_filters = layer.filters + num_neurons
        new_layer = layers.Conv2D(new_filters, layer.kernel_size, activation='relu', padding='same')
        model.layers[layer_idx] = new_layer
    return model

# Train the model
base_history = base_model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))
# Evaluate on test data
test_loss, test_acc = base_model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

from sklearn.metrics import confusion_matrix
import seaborn as sns
#model.evaluate(x_test, y_test)
y_pred = base_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis = 1)
print(y_pred_classes.shape, y_true.shape)
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=None, cmap='Blues', xticklabels=False, yticklabels=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for CIFAR-100 Base Model')
plt.show()

# Plot training & validation accuracy
plt.plot(base_history.history['accuracy'], label='Training accuracy')
plt.plot(base_history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Function to add neurons to the last dense layer
def add_neurons_to_dense_layer(model, num_new_neurons=size):
    # Get the current dense layers
    dense_layers = [layer for layer in model.layers if isinstance(layer, layers.Dense)]

    if len(dense_layers) == 0:
        raise ValueError("No Dense layers found in the model.")

    last_dense_layer = dense_layers[-1]
    new_units = last_dense_layer.units + num_new_neurons

    # Create a new dense layer with additional neurons
    new_dense_layer = layers.Dense(new_units, activation='relu')

    # Rebuild the model with the new layer
    new_model = models.Sequential()
    for layer in model.layers[:-1]:  # Keep all layers except the last Dense layer
        new_model.add(layer)

    new_model.add(new_dense_layer)  # Add the new dense layer

    # Add the final output layer
    new_model.add(tf.keras.layers.Dense(size, activation='softmax'))

    return new_model

# Training function with growth
def train_with_growth(model, x_train, y_train, x_test, y_test, epochs=10, growth_interval=3, num_new_neurons=size):
    full_history = {"loss": [], "accuracy": []}
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train the model for one epoch
        history = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
        for key, values in history.history.items():
            if key not in full_history:
                full_history[key] = []
            full_history[key].extend(values)
        # After every 'growth_interval' epochs, check for growth
        if (epoch + 1) % growth_interval == 0 and epoch != epochs-1:
            print(f"Growing model after epoch {epoch + 1}...")
            model = add_neurons_to_dense_layer(model, num_new_neurons)  # Add neurons to the last dense layer
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Re-compile after growth
        print("-" * 50)
    return model, full_history

# Build and compile the base model
activation_model = create_base_model()
activation_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(activation_model.summary())
# Train the model with growth

activation_model, activation_history = train_with_growth(activation_model, x_train, y_train, x_test, y_test, epochs=EPOCHS, growth_interval=GROWTH_INTERVAL, num_new_neurons=4)

# Final evaluation
test_loss, test_acc = activation_model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

y_pred = activation_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis = 1)
print(y_pred_classes.shape, y_true.shape)
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=None, cmap='Blues', xticklabels=False, yticklabels=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for CIFAR-100 Growing Model')
plt.show()

plt.plot(activation_history['accuracy'], label='Training accuracy')
plt.plot(activation_history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

def reset_keras_model(model):
    """
    Creates a new Keras model with the same architecture as the given model, but without any trained weights.

    Parameters:
        model (tf.keras.Model): The trained Keras model.

    Returns:
        tf.keras.Model: A new model with the same architecture but untrained (weights reinitialized).
    """
    # Clone the model to get the same architecture
    untrained_model = tf.keras.models.clone_model(model)
    # Compile the cloned model with the same configuration
    untrained_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Reinitialize the weights
    for layer in untrained_model.layers:
        if hasattr(layer, 'kernel_initializer') and layer.kernel_initializer is not None:
            layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
        if hasattr(layer, 'bias_initializer') and layer.bias_initializer is not None:
            layer.bias.assign(layer.bias_initializer(layer.bias.shape))
    return untrained_model

reset_activation_model = reset_keras_model(activation_model)
# Train the model
reset_activation_history = reset_activation_model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))
# Evaluate on test data
test_loss, test_acc = reset_activation_model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

plt.plot(reset_activation_history.history['accuracy'], label='Training accuracy')
plt.plot(reset_activation_history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

y_pred = reset_activation_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis = 1)
print(y_pred_classes.shape, y_true.shape)
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=None, cmap='Blues', xticklabels=False, yticklabels=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for CIFAR-100 Mature Model')
plt.show()

def replace_neurons_in_dense_layer(model, layer_index, num_new_neurons):
    # Get the current dense layers
    dense_layers = [layer for layer in model.layers if isinstance(layer, layers.Dense)]

    if len(dense_layers) == 0:
        raise ValueError("No Dense layers found in the model.")

    last_dense_layer = dense_layers[-1]
    new_units = last_dense_layer.units + num_new_neurons

    # Create a new dense layer with additional neurons
    new_dense_layer = layers.Dense(new_units, activation='relu')

    # Rebuild the model with the new layer
    new_model = models.Sequential()
    i = 0
    for layer in model.layers[:-1]:  # Keep all layers except the last Dense layer
        if i == layer_index:
            new_model.add(layers.Dense(num_new_neurons, activation='relu'))
        else:
            new_model.add(layer)

    new_model.add(new_dense_layer)  # Add the new dense layer

    # Add the final output layer
    new_model.add(tf.keras.layers.Dense(10, activation='softmax'))
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return new_model

print(new_activation_model.summary())
#activation_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
new_activation_model, new_activation_history = train_with_growth(new_activation_model, x_train, y_train, x_test, y_test, epochs=EPOCHS, growth_interval=GROWTH_INTERVAL, num_new_neurons=4)

test_loss, test_acc = reset_activation_model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

import pandas as pd
from google.colab import files
def save_history_to_csv_colab(history, file_name):

    # Convert the history dictionary to a pandas DataFrame
    try:
        history_df = pd.DataFrame(history.history)
    except:
        history_df = pd.DataFrame(history)

    # Save the DataFrame to a CSV file
    history_df.to_csv(file_name, index=False)

    # Download the file to the local machine
    files.download(file_name)
    print(f"Training history saved and ready to download as {file_name}")

save_history_to_csv_colab(base_history, dataset + "_base_history.csv")
save_history_to_csv_colab(activation_history, dataset +  "_activation_history.csv")
save_history_to_csv_colab(reset_activation_history, dataset + "_reset_activation_history.csv")
