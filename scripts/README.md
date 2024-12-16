Scripts to empirically evaluate the robustness certifier.

The top-level script is the shell script: `doit_verified_robust_gloro.sh`.

That script:
1. Trains an Gloro MNIST model (using `train_gloro.py`)
2. Manually constructs a "zero bias" version of the model and uses that to produce output vectors for all MNIST test points (using `zero_bias_saved_model.py`)
3. Saves the resulting "zero bias" model's weights in a format the certifier can understand (using `make_certifier_format.py`)
4. Runs the certifier over the output vectors produced by step 2, using the weights produced by step 3
5. Measures the resulting "zero bias" model's (using `test_verified_certified_robust_accuracy.py`):
  * accuracy: the proportion of MNIST test points correctly classified
  * robustness: the proportion of the MNIST test points that the cerfifier says are robust
  * verified robust accuracy (VRA): the proportion of MNIST test points that are both accurately classified and certified robust

The top-level script takes arguments to specify parameters to use to define the model architecture and how to train it, as well as the evaluation epsilon value used by the certifier when certifying robustness.