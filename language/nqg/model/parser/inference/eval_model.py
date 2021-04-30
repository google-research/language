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
"""Binary to evaluate model.

This binary can also be configured to run alongside a training job
and poll for new model checkpoints, writing eval metrics (e.g. for TensorBoard).

This binary also supports evaluations for settings such as NQG-T5, where
predictions from T5 are used when NQG does not produce an output. Such
'fallback' predictions can be supplied via the `--fallback_predictions` flag.
"""

import os
import time

from absl import app
from absl import flags

from language.nqg.model.parser import config_utils
from language.nqg.model.parser.data import tokenization_utils
from language.nqg.model.parser.inference import inference_wrapper
from language.nqg.model.parser.inference.targets import target_grammar
from language.nqg.model.qcfg import qcfg_file

from language.nqg.tasks import tsv_utils

import tensorflow as tf

from official.nlp.bert import configs

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_integer("limit", 0,
                     "Index of example to begin processing (Ignored if 0).")

flags.DEFINE_integer("offset", 0,
                     "Index of example to end processing (Ignored if 0).")

flags.DEFINE_bool("verbose", True, "Whether to print debug output.")

flags.DEFINE_string("model_dir", "", "Model directory.")

flags.DEFINE_bool("poll", False, "Whether to poll.")

flags.DEFINE_bool("write", False, "Whether to write metrics to model_dir.")

flags.DEFINE_string("subdir", "eval_test",
                    "Sub-directory of model_dir for writing metrics.")

flags.DEFINE_string("checkpoint", "", "Checkpoint prefix, or None for latest.")

flags.DEFINE_string("config", "", "Config file.")

flags.DEFINE_string("bert_dir", "",
                    "Directory for BERT, including vocab and config.")

flags.DEFINE_string("rules", "", "QCFG rules txt file.")

flags.DEFINE_string("fallback_predictions", "",
                    "Optional fallback predictions txt file.")

flags.DEFINE_string("target_grammar", "", "Optional target CFG.")


def compute_metrics(wrapper, examples):
  """Compute accuracy on examples."""
  # Initialize stats.
  num_examples = 0
  num_nqg_correct = 0
  num_nqg_predictions = 0
  num_fallback_correct = 0
  num_hybrid_correct = 0

  fallback_predictions = None
  if FLAGS.fallback_predictions:
    fallback_predictions = []
    with tf.io.gfile.GFile(FLAGS.fallback_predictions, "r") as predictions_file:
      for line in predictions_file:
        fallback_predictions.append(line.rstrip())

  for idx, example in enumerate(examples):
    if FLAGS.offset and idx < FLAGS.offset:
      continue
    if FLAGS.limit and idx >= FLAGS.limit:
      break
    if FLAGS.verbose:
      print("Processing example %s: %s" % (idx, example[0]))

    num_examples += 1

    source = example[0]
    gold_target = example[1]

    nqg_prediction, _ = wrapper.get_output(source)
    if nqg_prediction:
      num_nqg_predictions += 1

    if nqg_prediction == gold_target:
      num_nqg_correct += 1
    else:
      if FLAGS.verbose:
        print("nqg incorrect (gold vs. predicted):\n%s\n%s\n" %
              (gold_target, nqg_prediction))

    fallback_prediction = (
        fallback_predictions[idx] if fallback_predictions else None)
    if fallback_prediction == gold_target:
      num_fallback_correct += 1
    else:
      if FLAGS.verbose:
        print("fallback incorrect (gold vs. predicted):\n%s\n%s\n" %
              (gold_target, fallback_prediction))

    hybrid_prediction = nqg_prediction or fallback_prediction
    if hybrid_prediction == gold_target:
      num_hybrid_correct += 1
      if FLAGS.verbose:
        print("hybrid correct.")
    else:
      if FLAGS.verbose:
        print("hybrid incorrect.")

  metrics_dict = {
      "nqg_accuracy": float(num_nqg_correct) / float(num_examples),
      "fallback_accuracy": float(num_fallback_correct) / float(num_examples),
      "hybrid_accuracy": float(num_hybrid_correct) / float(num_examples),
      "nqg_coverage": float(num_nqg_predictions) / float(num_examples),
      "nqg_precision": float(num_nqg_correct) / float(num_nqg_predictions),
  }

  if FLAGS.verbose:
    print("num_examples: %s" % num_examples)
    print("num_nqg_correct: %s" % num_nqg_correct)
    print("num_nqg_predictions: %s" % num_nqg_predictions)
    print("num_fallback_correct: %s" % num_fallback_correct)
    print("num_hybrid_correct: %s" % num_hybrid_correct)
    print("metrics_dict: %s" % metrics_dict)

  return metrics_dict


def get_summary_writer():
  if not FLAGS.write:
    return None
  return tf.summary.create_file_writer(
      os.path.join(FLAGS.model_dir, FLAGS.subdir))


def write_metric(writer, name, metric, step):
  with writer.as_default():
    tf.summary.scalar(name, metric, step=step)


def get_checkpoint():
  """Return checkpoint path and step, or (None, None)."""
  if FLAGS.checkpoint:
    checkpoint = os.path.join(FLAGS.model_dir, FLAGS.checkpoint)
  else:
    checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
  # TODO(petershaw): Consider less hacky way to get current step.
  step = None
  if checkpoint is not None:
    step = int(checkpoint.split("-")[-2])
  print("Using checkpoint %s at step %s" % (checkpoint, step))
  return checkpoint, step


def get_inference_wrapper(config):
  """Construct and return InferenceWrapper."""
  rules = qcfg_file.read_rules(FLAGS.rules)

  tokenizer = tokenization_utils.get_tokenizer(
      os.path.join(FLAGS.bert_dir, "vocab.txt"))
  bert_config = configs.BertConfig.from_json_file(
      os.path.join(FLAGS.bert_dir, "bert_config.json"))

  target_grammar_rules = None
  if FLAGS.target_grammar:
    target_grammar_rules = target_grammar.load_rules_from_file(
        FLAGS.target_grammar)

  wrapper = inference_wrapper.InferenceWrapper(tokenizer, rules, config,
                                               bert_config,
                                               target_grammar_rules)

  return wrapper


def run_inference(writer, wrapper, examples, checkpoint, step=None):
  """Run inference."""
  wrapper.restore_checkpoint(checkpoint)
  metrics_dict = compute_metrics(wrapper, examples)
  for metric_name, metric_value in metrics_dict.items():
    print("%s at %s: %s" % (metric_name, step, metric_value))
    if FLAGS.write:
      write_metric(writer, metric_name, metric_value, step)


def main(unused_argv):
  config = config_utils.json_file_to_dict(FLAGS.config)
  wrapper = get_inference_wrapper(config)
  examples = tsv_utils.read_tsv(FLAGS.input)
  writer = get_summary_writer()

  if FLAGS.poll:
    last_checkpoint = None
    while True:
      checkpoint, step = get_checkpoint()
      if checkpoint == last_checkpoint:
        print("Waiting for new checkpoint...\nLast checkpoint: %s" %
              last_checkpoint)
      else:
        run_inference(writer, wrapper, examples, checkpoint, step=step)
        last_checkpoint = checkpoint
      if step and step >= config["training_steps"]:
        # Stop eval job after completing eval for last training step.
        break
      time.sleep(10)
  else:
    checkpoint, _ = get_checkpoint()
    run_inference(writer, wrapper, examples, checkpoint)


if __name__ == "__main__":
  app.run(main)
