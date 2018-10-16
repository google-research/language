# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs baseline models on the pattern dataset with tf.Estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import string

from absl import app
from absl import flags

from language.labs.memory import synthetic_dataset
from language.labs.memory.baseline_models import rnn_attention
from language.labs.memory.baseline_models import rnn_fast_weights
from language.labs.memory.baseline_models import vanilla_lstm
from language.labs.memory.baseline_models import vanilla_rnn
from language.labs.memory.differentiable_plasticity import rnn_differentiable_plasticity
from language.labs.memory.explicit_mem import rnn_explicit_mem
from language.labs.memory.model_utils import hamming_loss
from language.labs.memory.model_utils import write_flags_to_file


import tensorflow as tf

# Dataset params
flags.DEFINE_integer("num_examples", 100000, "Number of examples to train on.")
flags.DEFINE_integer("num_sets_per_sequence", 3,
                     "Number of patterns or kv pairs to remember in one"
                     " sequence.")
flags.DEFINE_integer("num_patterns_store", 2,
                     "Number of patterns we have to selectively remember.")
flags.DEFINE_integer("pattern_size", 50, "Dimensionality of each pattern.")
flags.DEFINE_bool("selective_task", None, "True if we want to remember a"
                  " subset of patterns in the sequence.")
flags.DEFINE_string("task_name", None, "'pattern' for pattern completion"
                    " task or 'symbolic' for symbolic character task.")

# Model params
flags.DEFINE_string(
    "model_name", None, "Name of the model to test.")
flags.DEFINE_integer("hidden_size", 50, "Number of hidden units in the RNN.")

# Learning params
flags.DEFINE_float("learning_rate", 3e-4, "Learning rate.")
flags.DEFINE_integer("lr_decay_step", 1000,
                     "How often (in iters) to decay the learning rate.")
flags.DEFINE_float("lr_decay_rate", .99, "Rate at which learning rate decays.")
flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum norm to clip gradients.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("num_epochs", 10, "Number of training epochs.")

# Logging / evaluation params
flags.DEFINE_integer("num_eval_steps", None,
                     "Number of steps to take during evaluation.")
flags.DEFINE_integer("save_checkpoints_steps", 5000,
                     "Number of steps between checkpoint saves.")
flags.DEFINE_string("experiment_logdir", "",
                    "Directory to write hparam settings.")

# Fast weight model-specific params
flags.DEFINE_integer(
    "fast_steps", 1, "Number of inner loop iterations"
    " where we apply fast weights.")
flags.DEFINE_float("fast_decay_rate", .95,
                   "Decay rate (lambda) for fast weights update.")
flags.DEFINE_float("fast_lr", .5,
                   "Learning rate (eta) for fast weights update.")

# Differentiable plasticity model-specific params
flags.DEFINE_bool(
    "use_oja", None, "True if we update memory with Oja's rule,"
    " False if we use Hebb's rule.")
flags.DEFINE_bool(
    "update_mem_with_prev_timestep", None, "True if we update memory with"
    " Hebb's rule, using a dot-product with the hidden state from the previous"
    " timestep.")
flags.DEFINE_bool("learn_fast_lr", None, "True if the fast lr is learnable.")
flags.DEFINE_bool("learn_plasticity_coeffs", None, "True if the plasticity"
                  " coefficients alpha are learnable.")

# Selective variant model-specific params
flags.DEFINE_string("fast_lr_learning_rule", None,
                    "'input_dependent', 'input_independent', or 'none'"
                    " depending on whether the fast_lr should be learned, and"
                    " whether it is learned as a function of the input at each"
                    " timestep.")
flags.DEFINE_string("fast_lr_activation", None, "'sigmoid' or 'clip'"
                    " or 'none' specifying how fast lr is constrained"
                    " to [0,1].")

# Required params
flags.mark_flag_as_required("selective_task")
flags.mark_flag_as_required("task_name")
flags.mark_flag_as_required("model_name")

FLAGS = flags.FLAGS
SYMBOLIC_VOCAB_SIZE = len(string.ascii_lowercase + "0123456789?")


def model_function(features, labels, mode):
  """A generic model, satisfying the tf.Estimator API.

  Args:
    features: Dictionary of input tensors.
      Key "seqs":
        For the pattern task: <tf.float32>[batch_size, seq_len, pattern_size]
        Sequences of patterns, where `seq_len` is the length of a sequence
        (including the query pattern) and `pattern_size` is the dimensionality
        of each pattern.
        For the symbolic task: <tf.int32>[batch_size, seq_len]
        Sequences of encoded kv pairs.
      Key "targets": same as `labels` arg.
    labels:
        For the pattern task: <tf.int32>[batch_size, pattern_size]
        The correct pattern to retrieve for the degraded query.
        For the symbolic task: <tf.int32>[batch_size]
        The correct value to retrieve for the given key.
    mode: One of the keys from tf.estimator.ModeKeys

  Returns:
    final_out: retrieved pattern for the degraded query.

  Raises:
    NameError: If FLAGS.model_name is not one of "vanilla_rnn", "vanilla_lstm",
    "rnn_attention", "rnn_fast_weights", "rnn_differentiable_plasticity," or if
    FLAGS.task_name is not "pattern" or "symbolic".
  """
  seqs = features["seqs"]

  if FLAGS.task_name == "symbolic":
    # seqs: <tf.int32>[FLAGS.batch_size, len(seqs), SYMBOLIC_VOCAB_SIZE]
    #   where len(seqs) == 2 * FLAGS.num_sets_per_sequence as each set is a
    #   key-value pair to put into memory.
    # labels: <tf.int32>[FLAGS.batch_size, SYMBOLIC_VOCAB_SIZE]
    seqs = tf.one_hot(seqs, SYMBOLIC_VOCAB_SIZE)
    labels = tf.one_hot(labels, SYMBOLIC_VOCAB_SIZE)

  # Forward pass through model
  # preds: <float32>[FLAGS.batch_size, FLAGS.pattern_size]
  if FLAGS.model_name == "vanilla_rnn":
    preds = vanilla_rnn(seqs, hidden_size=FLAGS.hidden_size)
  elif FLAGS.model_name == "vanilla_lstm":
    preds = vanilla_lstm(seqs, hidden_size=FLAGS.hidden_size)
  elif FLAGS.model_name == "rnn_attention":
    preds = rnn_attention(
        seqs, batch_size=FLAGS.batch_size, hidden_size=FLAGS.hidden_size)
  elif FLAGS.model_name == "rnn_fast_weights":
    preds = rnn_fast_weights(
        seqs,
        batch_size=FLAGS.batch_size,
        hidden_size=FLAGS.hidden_size,
        fast_steps=FLAGS.fast_steps,
        fast_decay_rate=FLAGS.fast_decay_rate,
        fast_lr=FLAGS.fast_lr,
    )
  elif FLAGS.model_name == "rnn_differentiable_plasticity":
    preds = rnn_differentiable_plasticity(
        seqs,
        batch_size=FLAGS.batch_size,
        hidden_size=FLAGS.hidden_size,
        fast_steps=FLAGS.fast_steps,
        fast_lr_fixed=FLAGS.fast_lr,
        use_oja_rule=FLAGS.use_oja,
        update_mem_with_prev_timestep=FLAGS.update_mem_with_prev_timestep,
        learn_fast_lr=FLAGS.learn_fast_lr,
        learn_plasticity_coeffs=FLAGS.learn_plasticity_coeffs,
    )
  elif FLAGS.model_name == "rnn_explicit_mem":
    preds = rnn_explicit_mem(
        seqs, batch_size=FLAGS.batch_size, hidden_size=FLAGS.hidden_size)
  else:
    raise NameError("Model %s not found" % FLAGS.model_name)

  # Calculate loss based on the last output (retrieved for the query)
  if FLAGS.task_name == "pattern":
    loss = tf.losses.mean_squared_error(preds, labels)
  elif FLAGS.task_name == "symbolic":
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels))
  else:
    raise NameError("Task name %s not found" % FLAGS.task_name)

  lr = tf.train.exponential_decay(
      FLAGS.learning_rate,
      tf.train.get_global_step(),
      FLAGS.lr_decay_step,
      FLAGS.lr_decay_rate,
      staircase=True)
  tf.summary.scalar("lr", lr)

  optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  grads, variables = zip(
      *optimizer.compute_gradients(loss, tf.trainable_variables()))
  grads, _ = tf.clip_by_global_norm(grads, FLAGS.max_gradient_norm)
  train_op = optimizer.apply_gradients(
      zip(grads, variables), global_step=tf.train.get_global_step())

  logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=1000)

  if FLAGS.task_name == "pattern":
    eval_metric_ops = {
        "hamming": hamming_loss(preds, labels, sign=True),
        # Naive baseline where we just guess the query
        "naive_guess_query": tf.metrics.mean_squared_error(
            seqs[:, -1, :], tf.cast(labels, seqs.dtype))
    }
  elif FLAGS.task_name == "symbolic":
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(predictions=tf.argmax(preds, 1),
                                        labels=tf.argmax(labels, 1))
    }

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      predictions=preds,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops,
      training_hooks=[logging_hook],
  )


def input_function(is_train):
  """Pulls in pattern dataset.

  Args:
    is_train: A boolean indicating whether we are constructing the dataset for
    the train set or the eval set.

  Returns:
    dataset: A tf.data.Dataset object containing the features and labels.

  Raises:
    NameError: If FLAGS.task_name is not "pattern" or "symbolic".
  """
  if FLAGS.task_name == "pattern":
    dataset = synthetic_dataset.get_pattern_dataset(
        FLAGS.num_examples,
        FLAGS.num_sets_per_sequence,
        FLAGS.pattern_size,
        selective=FLAGS.selective_task,
        num_patterns_store=FLAGS.num_patterns_store)
  elif FLAGS.task_name == "symbolic":
    dataset = synthetic_dataset.get_symbolic_dataset(
        is_train,
        FLAGS.num_examples,
        FLAGS.num_sets_per_sequence)
  else:
    raise NameError("Task %s not found" % FLAGS.task_name)

  dataset = (
      dataset.repeat(FLAGS.num_epochs).shuffle(buffer_size=1000)
      .batch(FLAGS.batch_size, drop_remainder=True))

  # Estimator expects a tuple.
  dataset = dataset.map(lambda d: (d, d["targets"]))

  return dataset


def experiment_function(run_config, hparams):
  """An experiment function satisfying the tf.estimator API.

  Args:
    run_config: A learn_running.EstimatorConfig object.
    hparams: Unused set of hyperparams.

  Returns:
    experiment: A tf.contrib.learn.Experiment object.
  """
  del hparams

  train_input_fn = partial(input_function, is_train=True)
  eval_input_fn = partial(input_function, is_train=False)

  estimator = tf.estimator.Estimator(
      model_fn=model_function,
      config=run_config,
      model_dir=run_config.model_dir)

  experiment = tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      eval_steps=FLAGS.num_eval_steps,
  )

  return experiment


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.experiment_logdir:
    write_flags_to_file(FLAGS, FLAGS.experiment_logdir + "/hparams.txt")

  # TODO(djweiss): Finish third-party Estimator code here.


if __name__ == "__main__":
  app.run(main)
