import numpy as np
import tensorflow as tf
from sys import stdout
from PIL import Image
import json
import os

import sys

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} INTERNAL_LAYER_SIZES certifier_results.json model_weights_csv_dir\n");
    sys.exit(1)

INTERNAL_LAYER_SIZES=eval(sys.argv[1])

json_results_file=sys.argv[2]

csv_loc=sys.argv[3]+"/"

print(f"Running with internal layer dimensions: {INTERNAL_LAYER_SIZES}")


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense

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
    i=i+1

# evaluate hte resulting model
print("Evaluating the resulting zero-bias gloro model...")

# turn off SSL cert checking :(
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Step 2: Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoded format
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Step 3: Compile the model
model.compile(optimizer='adam', 
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])


# Step 4: Evaluate the model on the test dataset
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)


print(f"Test Loss (zero bias model): {loss:.4f}")
print(f"Test Accuracy (zero bias model): {accuracy:.4f}")


print("Generating output vectors from the test (evaluation) data...")

outputs = model.predict(x_test)

predicted_classes = np.argmax(outputs, axis=1)
true_classes = np.argmax(y_test, axis=1)

correct_classifications = predicted_classes == true_classes

robustness=[]
with open(json_results_file, 'r') as f:
    robustness = json.load(f)

print("Evaluating Verified Certified Robust Accuracy...\n")
i=0 # robustness index
j=0 # correct_classifications index
count_robust_and_correct=0
count_robust=0
count_correct=0
# the first item in this list is the Lipschitz bounds; others may be debug messages etc.
assert len(robustness) >= len(correct_classifications)+1
robustness=robustness[1:]
assert len(robustness) >= len(correct_classifications)
n=len(robustness)
while i<n:
    r = robustness[i]
    if "certified" in r:
        robust = r["certified"]
        correct = correct_classifications[j]
        if robust and correct:
            count_robust_and_correct=count_robust_and_correct+1
        if robust:
            count_robust=count_robust+1
        if correct:
            count_correct=count_correct+1
        if i%1000==0:
            print(f"...done {i} of {n} evaluations...\n");
        j=j+1
    i=i+1

assert j==10000
assert i>=10000

print(f"Proportion robust: {float(count_robust)/float(10000)}")
print(f"Proportion correct: {float(count_correct)/float(10000)}")
print(f"Proportion robust and correct: {float(count_robust_and_correct)/float(10000)}")

