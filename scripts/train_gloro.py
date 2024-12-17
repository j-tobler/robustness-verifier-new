# This code was derived from the file tools/training/train_gloro.py from the gloro github repository
# Copyright (c) 2021 Klas Leino
# License: MIT; See https://github.com/klasleino/gloro/blob/a218dcdaaa41951411b0d520581e96e7401967d7/LICENSE
#
# Contributors (beyond those who contributed to the "gloro" project): Hira Syeda, Toby Murray
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from gloro.utils import print_if_verbose
from utils import get_data
from utils import get_optimizer
from gloro.models import GloroNet
from gloro.layers import Dense
from gloro.layers import Flatten
from gloro.layers import Input
from gloro.training import losses
from tensorflow.keras import backend as K
from gloro.training.callbacks import EpsilonScheduler
from gloro.training.callbacks import LrScheduler
from gloro.training.callbacks import TradesScheduler
from gloro.training.metrics import rejection_rate
from sklearn.metrics import confusion_matrix


def train_gloro(
        dataset,
        epsilon,
        epsilon_schedule='fixed',
        loss='crossentropy',
        augmentation='standard',
        epochs=None,
        batch_size=None,
        optimizer='adam',
        lr=0.001,
        lr_schedule='fixed',
        trades_schedule=None,
        verbose=False,
        INTERNAL_LAYER_SIZES=[64]
):
    _print = print_if_verbose(verbose)

    # Load data and set up data pipeline.
    _print('loading data...')

    train, test, metadata = get_data(dataset, batch_size, augmentation)

    # Create the model.
    _print('creating model...')

    inputs = Input((28, 28))
    z = Flatten()(inputs)
    for size in INTERNAL_LAYER_SIZES:
        z = Dense(size, activation='relu')(z)
    outputs = Dense(10)(z)

    g = GloroNet(inputs, outputs, epsilon)

    if verbose:
        g.summary()

    # Compile and train the model.
    _print('compiling model...')

    g.compile(
        # loss=losses.get(loss),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=get_optimizer(optimizer, lr),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        # metrics=[rejection_rate]
    )

    print('training model...')
    g.fit(
        train,
        epochs=epochs,
        validation_data=test,
        callbacks=[
                      EpsilonScheduler(epsilon_schedule),
                      LrScheduler(lr_schedule),
                  ] + ([TradesScheduler(trades_schedule)] if trades_schedule else []),
    )

    print('model training done.')

    return g


def script(
        dataset,
        epsilon,
        epsilon_schedule='fixed',
        loss='crossentropy',
        augmentation='standard',
        epochs=100,
        batch_size=128,
        optimizer='adam',
        lr=1e-3,
        #lr_schedule='decay_to_0.000001',
        lr_schedule='fixed',
        trades_schedule=None,
        plot_learning_curve=False,
        plot_confusion_matrix=False,
        INTERNAL_LAYER_SIZES=[64],
):

    g = train_gloro(
        dataset,
        epsilon,
        epsilon_schedule=epsilon_schedule,
        loss=loss,
        augmentation=augmentation,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        lr=lr,
        lr_schedule=lr_schedule,
        trades_schedule=trades_schedule,
        INTERNAL_LAYER_SIZES=INTERNAL_LAYER_SIZES
    )

    print('getting model accuracy numbers...')
    # Access the training accuracy
    final_training_accuracy = g.history.history['sparse_categorical_accuracy'][-1]

    # Access the validation accuracy (if validation_data was provided)
    final_validation_accuracy = g.history.history['val_sparse_categorical_accuracy'][-1]    

    print(f'model training accuracy: {final_training_accuracy}; validation accuracy: {final_validation_accuracy}')
    
    if plot_learning_curve:
        print('plotting learning curve...')
        history = g.history.history
    
        # learning curve
        # accuracy
        acc = history['sparse_categorical_accuracy']
        val_acc = history['val_sparse_categorical_accuracy']

        # loss
        loss = history['loss']
        val_loss = history['val_loss']

        epochs = range(1, len(acc) + 1)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(epochs, acc, 'r', label='Training Accuracy')
        ax1.plot(epochs, val_acc, 'b', label='Validation Accuracy')
        ax1.set_title('Training and Validation Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')

        ax2.plot(epochs, loss, 'r', label='Training Loss')
        ax2.plot(epochs, val_loss, 'b', label='Validation Loss')
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')

        plt.tight_layout()
        plt.show()
        fig.savefig('learning_curve.png')

        print('learning curve plotted.')

    print('saving model summary...')
    g.summary()
    g.save("model.keras")
    
    saved_stdout = sys.stdout    
    with open('gloro.summary', 'w') as sys.stdout:
        g.summary()
    sys.stdout=saved_stdout
    
    print('model summary saved.')

    print('extracting and saving model weights and biases...')
    # Extract weights and biases
    weights_and_biases = [
        (layer.get_weights()[0], layer.get_weights()[1])
        for layer in g.layers if len(layer.get_weights()) > 0]
            
    lipschitz_constants = [layer.lipschitz() for layer in g.layers if isinstance(layer,Dense)]
    
    for i, (weights, biases) in enumerate(weights_and_biases):
        np.savez(f"layer_{i}_weights_biases.npz", weights=weights, biases=biases)

    # Create a directory to save the files if it does not exist
    save_dir = 'model_weights_csv'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Loop through each layer, extract weights and biases, and save them
    for i, (weights, biases) in enumerate(weights_and_biases):
        np.savetxt(os.path.join(save_dir, f'layer_{i}_weights.csv'), weights, delimiter=',', fmt='%f')
        np.savetxt(os.path.join(save_dir, f'layer_{i}_biases.csv'), biases, delimiter=',', fmt='%f')

    # Loop through each layer, extract weights and biases, and save them
    for i, c in enumerate(lipschitz_constants):
        np.savetxt(os.path.join(save_dir, f'layer_{i}_lipschitz.csv'), [c], delimiter=',', fmt='%f')
        
    print('model weights and biases extracted.')

    
    print("evaluating the model against testing dataset...")
    train, ds_test, metadata = get_data(dataset, batch_size, augmentation)

    eval_results = g.evaluate(ds_test)

    eval_accuracy = eval_results[1]
    print(f"model evaluation accuracy: {eval_accuracy}.")


    print("SUMMARY")
    print(f"lr_schedule: {lr_schedule}")
    print(f"epsilon: {epsilon}")
    print(f"dense layer sizes: {INTERNAL_LAYER_SIZES}")
    print(f"accuracy: {eval_accuracy}")
    print(f"lipschitz constants: {lipschitz_constants}")
    
    # At the end of your script
    K.clear_session()

import sys

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} epsilon INTERNAL_LAYER_SIZES epochs\n");
    sys.exit(1)

epsilon=float(sys.argv[1])
    
internal_layers=eval(sys.argv[2])

epochs=int(sys.argv[3])

print(f"Running with internal layer dimensions: {internal_layers}")

# using values from Table B.1 in Leino et al. 2021 for epsilon_schedule, lr_schedule, lr, loss, and augmentation.
# we use custom values for:
#   batch_size -- Leino use 256 but this gives much worse Lipschitz bounds. Smaller batch size seems to do some kind of quasi-regularisation
#   epochs     -- Leino use 500 eophcs but, perhaps because we use a far simpler neural net, optimal value for epochs seems to be about 3.
#              -- larger values results in bigger Lipschitz bounds, as the model fits the data better
script(
    dataset='mnist',
    epsilon=epsilon,
    #epsilon_schedule='[0.01]-log-[50%:1.1]',
    epsilon_schedule='fixed',
    batch_size=32,
    lr=1e-3,
    #lr_schedule='decay_after_half_to_0.000001',
    lr_schedule='decay_to_0.000001',
    epochs=epochs, 
    #loss='sparse_trades_kl.1.5',
    loss='crossentropy',    
    augmentation='none',
    INTERNAL_LAYER_SIZES=internal_layers)
