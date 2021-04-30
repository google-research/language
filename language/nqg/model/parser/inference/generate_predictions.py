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
"""Binary to generate predicted targets given input txt file of sources.

An input txt file of sources can be generated from a TSV file using
the `nqg/tasks/strip_targets.py` script.

This binary also supports evaluations for settings such as NQG-T5, where
predictions from T5 are used when NQG does not produce an output. Such
'fallback' predictions can be supplied via the `--fallback_predictions` flag.
"""

import os

from absl import app
from absl import flags

from language.nqg.model.parser import config_utils
from language.nqg.model.parser.data import tokenization_utils
from language.nqg.model.parser.inference import inference_wrapper
from language.nqg.model.parser.inference.targets import target_grammar
from language.nqg.model.qcfg import qcfg_file

import tensorflow as tf

from official.nlp.bert import configs

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input txt file for sources.")

flags.DEFINE_string("output", "", "Output txt file for predicted targets.")

flags.DEFINE_bool("verbose", True, "Whether to print debug output.")

flags.DEFINE_string("model_dir", "", "Model directory.")

flags.DEFINE_string("checkpoint", "", "Checkpoint prefix, or None for latest.")

flags.DEFINE_string("config", "", "Config file.")

flags.DEFINE_string(
    "bert_dir", "",
    "Directory for BERT vocab, config, and (optionally) checkpoint.")

flags.DEFINE_string("rules", "", "QCFG rules txt file.")

flags.DEFINE_string("fallback_predictions", "",
                    "Optional fallback predictions txt file.")

flags.DEFINE_string("target_grammar", "", "Optional target CFG.")


def get_checkpoint():
  if FLAGS.checkpoint:
    return os.path.join(FLAGS.model_dir, FLAGS.checkpoint)
  else:
    return tf.train.latest_checkpoint(FLAGS.model_dir)


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

  # Restore checkpoint.
  checkpoint = get_checkpoint()
  print("Loading from checkpoint: %s" % checkpoint)
  wrapper.restore_checkpoint(checkpoint)

  return wrapper


def get_predicted_target(wrapper, source, fallback_prediction):
  nqg_prediction, _ = wrapper.get_output(source)
  if nqg_prediction is None:
    return fallback_prediction
  else:
    return nqg_prediction


def get_fallback_predictions(sources):
  """Return List of fallback predictions or List of `None` if not provided."""
  if FLAGS.fallback_predictions:
    fallback_predictions = []
    with tf.io.gfile.GFile(FLAGS.fallback_predictions, "r") as predictions_file:
      for line in predictions_file:
        fallback_predictions.append(line.rstrip())
    if len(sources) != len(fallback_predictions):
      raise ValueError(
          "Number of inputs != number of fallback predictions: %s vs. %s." %
          (len(sources), len(fallback_predictions)))
    return fallback_predictions
  else:
    return [None] * len(sources)


def main(unused_argv):
  config = config_utils.json_file_to_dict(FLAGS.config)
  wrapper = get_inference_wrapper(config)

  sources = []
  with tf.io.gfile.GFile(FLAGS.input, "r") as input_file:
    for line in input_file:
      sources.append(line.rstrip())

  fallback_predictions = get_fallback_predictions(sources)
  with tf.io.gfile.GFile(FLAGS.output, "w") as output_file:
    for source, fallback_prediction in zip(sources, fallback_predictions):
      predicted_target = get_predicted_target(wrapper, source,
                                              fallback_prediction)
      output_file.write("%s\n" % predicted_target)


if __name__ == "__main__":
  app.run(main)
