import numpy as np
import tensorflow as tf
from sys import stdout
from PIL import Image
import os

import sys

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Layer

class MinMax(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._flat_op = Flatten()

    def call(self, x):
        x_flat = self._flat_op(x)
        x_shape = tf.shape(x_flat)

        grouped_x = tf.reshape(
            x_flat,
            tf.concat([x_shape[:-1], (-1, 2)], -1))

        min_x = tf.reduce_min(grouped_x, axis=-1, keepdims=True)
        max_x = tf.reduce_max(grouped_x, axis=-1, keepdims=True)

        sorted_x = tf.reshape(
            tf.concat([min_x, max_x], axis=-1),
            tf.shape(x))

        return sorted_x

    def lipschitz(self):
        return 1.

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} INTERNAL_LAYER_SIZES model_weights_csv_dir output_file\n");
    sys.exit(1)

INTERNAL_LAYER_SIZES=eval(sys.argv[1])

csv_loc=sys.argv[2]+"/"

output_file=sys.argv[3]

print(f"Running with internal layer dimensions: {INTERNAL_LAYER_SIZES}")



def mprint(string):
    print(string, end="")


def printlist(floatlist):
        count=0
        n=len(floatlist)
        for num in floatlist:
            mprint(f"{num:.5f}")
            count=count+1
            if count<n:
                mprint(",")

    
# Define the model architecture
inputs = Input((28, 28))
z = Flatten()(inputs)
for size in INTERNAL_LAYER_SIZES:
    z = Dense(size, activation='relu')(z)
outputs = Dense(10)(z)
model = Model(inputs, outputs)

# define a second copy to hold the original model
inputs = Input((28, 28))
z = Flatten()(inputs)
for size in INTERNAL_LAYER_SIZES:
    z = Dense(size, activation='relu')(z)
outputs = Dense(10)(z)
model_orig = Model(inputs, outputs)

# % ls model_weights_csv 
# layer_0_biases.csv	layer_1_biases.csv
# layer_0_weights.csv	layer_1_weights.csv

print("Building zero-bias gloro model from saved weights...")



dense_weights = []
dense_biases = []
dense_zero_biases = []
i=0
# always one extra iteration than INTERNAL_LAYER_SIZES length
while i<=len(INTERNAL_LAYER_SIZES):
    dense_weights.append(np.loadtxt(csv_loc+f"layer_{i}_weights.csv", delimiter=","))
    dense_biases.append(np.loadtxt(csv_loc+f"layer_{i}_biases.csv", delimiter=","))
    dense_zero_biases.append(np.zeros_like(dense_biases[i]))

    model.layers[i+2].set_weights([dense_weights[i], dense_zero_biases[i]])
    model_orig.layers[i+2].set_weights([dense_weights[i], dense_biases[i]])
    i=i+1

# evaluate hte resulting model
print("Evaluating the resulting zero-bias gloro model...")

# turn off SSL cert checking :(
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Step 2: Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# save original test data
x_test_orig = x_test

# Normalize pixel values to [0, 1]
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoded format
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Step 3: Compile the model
model.compile(optimizer='adam', 
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Step 3: Compile the model
model_orig.compile(optimizer='adam', 
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Step 4: Evaluate the model on the test dataset
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

loss_orig, accuracy_orig = model_orig.evaluate(x_test, y_test, verbose=2)

print(f"Test Loss (zero bias model): {loss:.4f}")
print(f"Test Accuracy (zero bias model): {accuracy:.4f}")

print(f"Test Loss (original model): {loss_orig:.4f}")
print(f"Test Accuracy (original model): {accuracy_orig:.4f}")


print("Generating output vectors from the test (evaluation) data...")

outputs = model.predict(x_test)
n=len(outputs)

#print(f"We have {n} test outputs we could try the certifier on.")
#print("How many do you want?")
#user_input = int(input(f"Enter a number between 0 and {n}: "))
user_input=n

# Check if input is in the range
if 0 <= user_input <= n:


    # Create a directory to save the images
    output_dir = "mnist_images"
    os.makedirs(output_dir, exist_ok=True)
    saved_stdout=sys.stdout
    with open(output_file,'w') as f:
        sys.stdout=f
    
        for i in range(user_input):
            test_output = outputs[i].tolist()
            printlist(test_output)
            #mprint(" ")
            #mprint(epsilon)
            mprint("\n")

            # Get the image data
            image_array = x_test_orig[i]
    
            # Convert the image array to a PIL Image object
            image = Image.fromarray(image_array)
    
            # Save the image to a file
            output_path = os.path.join(output_dir, f"mnist_image_{i}.png")
            image.save(output_path)
    sys.stdout=saved_stdout
else:
    print("Invalid number entered. No outputs for you!")
        

