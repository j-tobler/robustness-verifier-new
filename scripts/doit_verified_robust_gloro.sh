# train a gloro net

if [ $# -ne 7 ]; then
    echo "Usage $0 gloro_epsilon INTERNAL_LAYER_SIZES eval_epsilon robustness_certifier_binary GRAM_ITERATIONS epochs batch_size"
    echo ""
    echo "Optimal values based on testing for eval_epsilon=0.3 so far suggest: "
    echo "  gloro_epsilon=0.45"
    echo "  INTERNAL_LAYER_SIZES=[]"
    echo "  GRAM_ITERATIONS=1 (well, it can't be anything else for performance reasons)"
    echo "  epochs=3"
    echo "  batch_size=32"
    exit 1
fi

EPSILON=$1
INTERNAL_LAYER_SIZES=$2
EVAL_EPSILON=$3
CERTIFIER=$4
GRAM_ITERATIONS=$5
EPOCHS=$6
BATCH_SIZE=$7

if ! [[ $GRAM_ITERATIONS =~ ^[0-9]+$ ]] || [ "$GRAM_ITERATIONS" -le 0 ]; then
    echo "GRAM_ITERATIONS should be positive"
    exit 1
fi

if ! [[ $EPOCHS =~ ^[0-9]+$ ]] || [ "$GRAM_ITERATIONS" -le 0 ]; then
    echo "EPOCHS should be positive"
    exit 1
fi

if ! [[ $BATCH_SIZE =~ ^[0-9]+$ ]] || [ "$GRAM_ITERATIONS" -le 0 ]; then
    echo "BATCH_SIZE should be positive"
    exit 1
fi

if [ ! -x ${CERTIFIER} ]; then
    echo "${CERTIFIER} doesn't exist or is not executable"
    exit 1
fi

PYTHON=python3

# clean out any old temporary model weights etc.
rm -rf model_weights_csv

MODEL_WEIGHTS_DIR="model_weights_epsilon_${EPSILON}_${INTERNAL_LAYER_SIZES}_${EPOCHS}"
MODEL_OUTPUTS="all_mnist_outputs_epsilon_${EPSILON}_${INTERNAL_LAYER_SIZES}_${EPOCHS}.txt"
NEURAL_NET_FILE="neural_net_mnist_epsilon_${EPSILON}_${INTERNAL_LAYER_SIZES}_${EPOCHS}.txt"
MODEL_OUTPUTS_EVAL="all_mnist_outputs_epsilon_${EPSILON}_${INTERNAL_LAYER_SIZES}_${EPOCHS}_eval_${EVAL_EPSILON}.txt"
RESULTS_JSON="results_epsilon_${EPSILON}_${INTERNAL_LAYER_SIZES}_${EPOCHS}_eval_${EVAL_EPSILON}.json"

# clean out old results of this script
rm -rf "${MODEL_WEIGHTS_DIR}" "${MODEL_OUTPUTS}" "${NEURAL_NET_FILE}" "${MODEL_OUTPUTS_EVAL}" "${RESULTS_JSON}"


# train the gloro model
${PYTHON} train_gloro.py $EPSILON "$INTERNAL_LAYER_SIZES" $EPOCHS

if [ ! -d model_weights_csv ]; then
    echo "Training gloro model failed or results not successfully saved to model_weights_csv/ dir"
    exit 1
fi

# save the weights
mv model_weights_csv "$MODEL_WEIGHTS_DIR"
# make the outputs from the zero-bias model
${PYTHON} zero_bias_saved_model.py "$INTERNAL_LAYER_SIZES" "$MODEL_WEIGHTS_DIR" "$MODEL_OUTPUTS"
# make the neural net in a form the certifier can understand
${PYTHON} make_certifier_format.py "$INTERNAL_LAYER_SIZES" "$MODEL_WEIGHTS_DIR" > "$NEURAL_NET_FILE"
# add the epsilon to each model output for the certifier to certify against
sed "s/$/ ${EVAL_EPSILON}/" "$MODEL_OUTPUTS" > "$MODEL_OUTPUTS_EVAL"


echo "Running the certifier. This may take a while and there is no progress output..."
cat "$MODEL_OUTPUTS_EVAL" | ${CERTIFIER} "$NEURAL_NET_FILE" "$GRAM_ITERATIONS" > "$RESULTS_JSON"

${PYTHON} test_verified_certified_robust_accuracy.py "$INTERNAL_LAYER_SIZES" "$RESULTS_JSON" "$MODEL_WEIGHTS_DIR"

echo "All done."
echo "Model weights saved in: ${MODEL_WEIGHTS_DIR}"
echo "Model outputs saved in: ${MODEL_OUTPUTS}"
echo "Neural network (for certifier) saved in: ${NEURAL_NET_FILE}"
echo "Model outputs for evaluation saved in: ${MODEL_OUTPUTS_EVAL}"
echo "Certified robustness results saved in: ${RESULTS_JSON}"


