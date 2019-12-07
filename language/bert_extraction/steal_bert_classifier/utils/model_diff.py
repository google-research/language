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
"""This file compares the difference between parameters of two models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bert import modeling
from bert import tokenization

from bert_extraction.steal_bert_classifier.models import run_classifier

import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Other parameters

flags.DEFINE_string(
    "init_checkpoint1", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("bert_config_file1", None,
                    "BERT config file for the first model.")

flags.DEFINE_string(
    "init_checkpoint2", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("bert_config_file2", None,
                    "BERT config file for the second model.")

flags.DEFINE_string("diff_type", "euclidean",
                    "Type of difference function to be used.")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "sst-2": run_classifier.SST2Processor,
      "mnli": run_classifier.MnliProcessor
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint1)
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint2)

  bert_config1 = modeling.BertConfig.from_json_file(FLAGS.bert_config_file1)
  bert_config2 = modeling.BertConfig.from_json_file(FLAGS.bert_config_file2)

  if FLAGS.max_seq_length > bert_config1.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config1.max_position_embeddings))

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  input_ids = tf.placeholder(dtype=tf.int32, shape=(None, FLAGS.max_seq_length))
  input_mask = tf.placeholder(
      dtype=tf.int32, shape=(None, FLAGS.max_seq_length))
  segment_ids = tf.placeholder(
      dtype=tf.int32, shape=(None, FLAGS.max_seq_length))
  label_ids = tf.placeholder(dtype=tf.int32, shape=(None,))
  num_labels = len(processor.get_labels())

  with tf.variable_scope("model1"):
    run_classifier.create_model(
        bert_config1,
        False,
        input_ids,
        input_mask,
        segment_ids,
        label_ids,
        num_labels,
        use_one_hot_embeddings=False)
  vars1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model1")

  with tf.variable_scope("model2"):
    run_classifier.create_model(
        bert_config2,
        False,
        input_ids,
        input_mask,
        segment_ids,
        label_ids,
        num_labels,
        use_one_hot_embeddings=False)
  vars2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model2")

  tf.train.init_from_checkpoint(
      FLAGS.init_checkpoint1,
      {"%s" % v.name[v.name.index("/") + 1:].split(":")[0]: v for v in vars1})

  tf.train.init_from_checkpoint(
      FLAGS.init_checkpoint2,
      {"%s" % v.name[v.name.index("/") + 1:].split(":")[0]: v for v in vars2})

  def abs_diff(var_name):
    with tf.variable_scope("model1", reuse=True):
      var1 = tf.get_variable(var_name)

    with tf.variable_scope("model2", reuse=True):
      var2 = tf.get_variable(var_name)

    return tf.math.abs(tf.math.subtract(var1, var2))

  def sq_diff(var_name):
    with tf.variable_scope("model1", reuse=True):
      var1 = tf.get_variable(var_name)

    with tf.variable_scope("model2", reuse=True):
      var2 = tf.get_variable(var_name)

    return tf.math.subtract(var1, var2) * tf.math.subtract(var1, var2)

  total_diff = 0.0
  total_params = 0

  bert_diff = 0.0
  bert_params = 0

  classifier_diff = 0.0
  classifier_params = 0

  for var in vars1:
    if FLAGS.diff_type == "euclidean":
      var_diff = tf.reduce_sum(
          sq_diff(var.name[var.name.index("/") + 1:var.name.index(":")]))
    else:
      var_diff = tf.reduce_sum(
          abs_diff(var.name[var.name.index("/") + 1:var.name.index(":")]))

    var_params = 1
    shape = var.get_shape()
    for dim in shape:
      var_params *= dim

    total_diff += var_diff
    total_params += var_params

    # Setup for BERT parameters
    if "bert" in var.name:
      bert_diff += var_diff
      bert_params += var_params
    else:
      classifier_diff += var_diff
      classifier_params += var_params

  if FLAGS.diff_type == "euclidean":
    total_diff = tf.sqrt(total_diff)
    bert_diff = tf.sqrt(bert_diff)
    classifier_diff = tf.sqrt(classifier_diff)
  else:
    total_diff = total_diff / tf.cast(total_params, tf.float32)
    bert_diff = bert_diff / tf.cast(bert_params, tf.float32)
    classifier_diff = classifier_diff / tf.cast(classifier_params, tf.float32)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  tf.logging.info("average diff in all params = %.8f", sess.run(total_diff))
  tf.logging.info("average diff in bert params = %.8f", sess.run(bert_diff))
  tf.logging.info("average diff in classifier params = %.8f",
                  sess.run(classifier_diff))

  return


if __name__ == "__main__":
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file1")
  flags.mark_flag_as_required("bert_config_file2")
  flags.mark_flag_as_required("init_checkpoint1")
  flags.mark_flag_as_required("init_checkpoint2")
  tf.app.run()
