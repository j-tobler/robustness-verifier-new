# Derived from examples/adversarial_training_data_augmentation.py
# of the ART repo: https://github.com/Trusted-AI/adversarial-robustness-toolbox
# License: MIT
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# artibtrary precision math
from mpmath import mp, mpf, sqrt

import os
import json
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

if len(sys.argv) != 6 and len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} INTERNAL_LAYER_SIZES model_weights_csv_dir epsilon [certifier_results.json disagree_output_dir]\n");
    sys.exit(1)

INTERNAL_LAYER_SIZES=eval(sys.argv[1])

csv_loc=sys.argv[2]+"/"

epsilon=float(sys.argv[3])

json_results_file=None
disagree_output_dir=None
if len(sys.argv) == 6:
    json_results_file=sys.argv[4]
    disagree_output_dir=sys.argv[5]+"/"

    if os.path.exists(disagree_output_dir):
        raise FileExistsError(f"The directory '{disagree_output_dir}' already exists.")


print(f"Running with internal layer dimensions: {INTERNAL_LAYER_SIZES}")

print(f"Running PGD attack with epsilon: {epsilon}")

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

conservative_epsilon=epsilon#-0.0000001

pgd = ProjectedGradientDescent(classifier, norm=2, eps=conservative_epsilon, eps_step=0.01, max_iter=1000, num_random_init=True)

# Create some adversarial samples for evaluation
x_test_pgd = pgd.generate(x_test,y_test)

# Evaluate the model on the adversarial samples
predict_pgd = model.predict(x_test_pgd)
labels_pgd = np.argmax(predict_pgd, axis=1)

n=labels_pgd.shape[0]
assert labels_pgd.shape[0] == x_test.shape[0]

if disagree_output_dir is not None:
    os.makedirs(disagree_output_dir)

robustness_log=[]

if json_results_file is not None:
    with open(json_results_file, 'r') as f:
        robustness_log = json.load(f)

robustness = [d for d in robustness_log if "certified" in d]

assert len(robustness)==n or (robustness==[] and json_results_file is None)

disagree=0
false_positive=0
max_fp_norm=-1.0
min_fp_norm=-1.0

max_disagree_norm=-1.0
min_disagree_norm=-1.0

unsound=0

def vector_to_mph(v):
    return list(map(mpf, v.tolist()[0]))

def l2_norm_mph(vector1, vector2):
    return sqrt(sum((x - y)**2 for x, y in zip(vector1, vector2)))

i=0
while i<n:
    if labels_pgd[i] != x_test_pred[i]:
        # calculate the norm using arbitrary precision arithmetic to make sure it really is a valid attack
        x_pgd_mph = vector_to_mph(x_test_pgd[i])
        x_mph = vector_to_mph(x_test[i])        
        l2_norm = l2_norm_mph(x_pgd_mph, x_mph)
        if (l2_norm > epsilon):
            if false_positive == 0:
                max_fp_norm=l2_norm
                min_fp_norm=l2_norm
            else:
                if max_fp_norm < l2_norm:
                    max_fp_norm=l2_norm
                if min_fp_norm > l2_norm:
                    min_fp_norm=l2_norm
            false_positive=false_positive+1
        else:
            if disagree == 0:
                max_disagree_norm=l2_norm
                min_disagree_norm=l2_norm
            else:
                if max_disagree_norm < l2_norm:
                    max_disagree_norm=l2_norm
                if min_disagree_norm > l2_norm:
                    min_disagree_norm=l2_norm
            # found a successful attack
            disagree=disagree+1
            
            
            if robustness!=[]:

                r = robustness[i]
                robust = r["certified"]

                if robust:
                    # found an attack when the certifier said the output was robust!
                    unsound=unsound+1
                    x=x_test[i]
                    x_pgd=x_test_pgd[i]
                    lab_y=x_test_pred[i]                
                    y_pgd=predict_pgd[i]
                    lab_y_pgd=np.argmax(y_pgd, axis=0)
                    input_path = os.path.join(disagree_output_dir, f"input_{i}.npy")
                    np.savetxt(disagree_output_dir+f"/unsound_{i}_x.csv", x, delimiter=',', fmt='%f')
                    np.savetxt(disagree_output_dir+f"/unsound_{i}_x_pgd.csv", x_pgd, delimiter=',', fmt='%f')
                    with open(disagree_output_dir+f"/unsound_{i}_summary.txt", "w") as f:
                        f.write(f"L2 Norm    : {l2_norm}\n")
                        f.write(f"Y label    : {lab_y}\n")                    
                        f.write(f"Y PGD      : {y_pgd}\n")
                        f.write(f"Y PGD label: {lab_y_pgd}\n")
    i=i+1

agree=n-disagree
print(f"Model accuracy: {nb_correct_pred/x_test.shape[0] * 100}")

assert(agree >= np.sum(labels_pgd == labels_true))

print("Accuracy on PGD adversarial samples: %.2f%%" % (agree / n * 100))
if disagree > 0:
    print(f"Norms of non-false-positive vectors that cause classification changes: min: {min_disagree_norm}; max: {max_disagree_norm}")

print(f"False positives in PGD attack: {false_positive}")
if false_positive > 0:
    print(f"Norms of false positive vectors: min: {min_fp_norm}; max: {max_fp_norm}")

print(f"Number of PGD attacks succeeding against certified robust outputs: {unsound}")
