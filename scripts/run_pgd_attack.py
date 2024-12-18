# Derived from examples/adversarial_training_data_augmentation.py
# of the ART repo: https://github.com/Trusted-AI/adversarial-robustness-toolbox
# License: MIT
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import sys
import keras
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input, BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier
from art.data_generators import KerasDataGenerator
from art.defences.trainer import AdversarialTrainer
from art.utils import load_dataset

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} INTERNAL_LAYER_SIZES model_weights_csv_dir\n");
    sys.exit(1)

INTERNAL_LAYER_SIZES=eval(sys.argv[1])

csv_loc=sys.argv[2]+"/"

print(f"Running with internal layer dimensions: {INTERNAL_LAYER_SIZES}")

# Define the model architecture
inputs = Input((28, 28))
z = Flatten()(inputs)
for size in INTERNAL_LAYER_SIZES:
    z = Dense(size, activation='relu')(z)
outputs = Dense(10)(z)
model = Model(inputs, outputs)

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





# Step 3: Compile the model
model.compile(optimizer='adam', 
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Step 2: Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_test = x_test.astype('float32') / 255.0


# Convert labels to one-hot encoded format
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

labels_true = np.argmax(y_test, axis=1)


# Build a Keras image augmentation object and wrap it in ART
batch_size = 50

classifier = KerasClassifier(model, clip_values=(0.0,1.0), use_logits=False)
model.summary()

x_test_pred = np.argmax(classifier.predict(x_test), axis=1)
nb_correct_pred = np.sum(x_test_pred == np.argmax(y_test, axis=1))

pgd = ProjectedGradientDescent(classifier, norm=2, eps=0.3, eps_step=0.01, max_iter=100, num_random_init=True)

# Create some adversarial samples for evaluation
x_test_pgd = pgd.generate(x_test,y_test)

# Evaluate the model on the adversarial samples
labels_pgd = np.argmax(model.predict(x_test_pgd), axis=1)

print(f"Model accuracy: {nb_correct_pred/x_test.shape[0] * 100}")

print("Accuracy on PGD adversarial samples: %.2f%%" % (np.sum(labels_pgd == labels_true) / x_test.shape[0] * 100))
