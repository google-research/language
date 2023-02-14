This directory contains scripts to train a T5-based cross-attention classifier.
The codebase relies upon the [t5x repository](https://github.com/google-research/t5x).
Follow instructions from that library to define a task, run fine-tuning, and
generate scores at inference time.

To generate training examples, you can run `gen_training_examples.py`.
This script can also run on the validation set to generate an evaluation set to efficiently
evaluate model performance during fine-tuning.

To generate predictions, you should first run `gen_inference_inputs.py`. Then, generate scores following the inference instructions from the `t5x` library with `--gin.infer.mode="'score'"`.
You can determine a threshold by running `determine_threshold.py` on the validation
set.
Then, you can run `filter_predictions.py` to filter a set of retrieved documents based on the cross-attention classifier.
