import utils
import numpy as np
import tensorflow as tf

def build_mnist_model(Input, Flatten, Dense, input_size=28, internal_layer_sizes=[]):
    """set input_size to something smaller if the model is downsampled"""
    inputs = Input((input_size, input_size))
    z = Flatten()(inputs)
    for size in internal_layer_sizes:
        z = Dense(size, activation='relu')(z)
    outputs = Dense(10)(z)
    return (inputs, outputs)

def load_and_set_weights(csv_loc, internal_layer_sizes, model):
    """model should already be built. This will compile it too"""
    dense_weights = []
    dense_biases = []
    dense_zero_biases = []
    i=0
    # always one extra iteration than internal_layer_sizes length
    while i<=len(internal_layer_sizes):
        dense_weights.append(np.loadtxt(csv_loc+f"layer_{i}_weights.csv", delimiter=","))
        dense_biases.append(np.loadtxt(csv_loc+f"layer_{i}_biases.csv", delimiter=","))
        dense_zero_biases.append(np.zeros_like(dense_biases[i]))

        model.layers[i+2].set_weights([dense_weights[i], dense_zero_biases[i]])
        i=i+1
        
    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
                  metrics=['accuracy'])


    
def load_mnist_gloro_data(batch_size=256, augmentation='none', new_input_size=None):
    """set new_input_size to resize the dataset. Returns a pair (train, test)"""
    train, test, metadata = utils.get_data('mnist', batch_size, augmentation)

    
    if new_input_size:
        def resize(image, label):
            image = tf.image.resize(image, [new_input_size, new_input_size])  
            return image, label
        train = train.map(resize)
        test = test.map(resize)
        
    return (train, test)
    
def load_mnist_test_data(new_input_size=None):
    """set new_input_size to resize the test dataset. Returns a pair (x_test, y_test)"""
    # turn off SSL cert checking :(
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_test = x_test.astype('float32') / 255.0

    if new_input_size:
        x_test = tf.image.resize(x_test[..., tf.newaxis], [new_input_size, new_input_size]).numpy()

    # Convert labels to one-hot encoded format
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return (x_test, y_test)
