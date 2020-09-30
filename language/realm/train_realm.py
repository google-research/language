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
"""Run REALM pre-training."""
import functools
import os

from absl import app
from absl import flags

from language.common.utils import experiment_utils
from language.realm import model
import tensorflow.compat.v1 as tf


flags.DEFINE_string("bert_hub_module_handle", None,
                    "Handle for the BERT TF-Hub module.")

flags.DEFINE_string("vocab_path", None, "Path to vocabulary file.")

flags.DEFINE_boolean("do_lower_case", True, "Whether to lowercase text.")

flags.DEFINE_string("embedder_hub_module_handle", None,
                    "Hub module for embedding queries and candidates.")

flags.DEFINE_integer("query_seq_len", None,
                     "Maximum sequence length of the query text.")

flags.DEFINE_integer("candidate_seq_len", None,
                     "Maximum sequence length of a candidate text.")

flags.DEFINE_integer("max_masks", None,
                     "Maximum number of tokens that can be masked out.")

flags.DEFINE_float("learning_rate", 3e-5, "Learning rate.")

flags.DEFINE_integer("num_input_threads", 12, "Num threads for input reading.")

flags.DEFINE_integer("num_candidates", None,
                     "Number of candidate texts considered by the model.")

flags.DEFINE_list("train_preprocessing_servers", None,
                  "Training data generation servers.")

flags.DEFINE_list("eval_preprocessing_servers", None,
                  "Evaluation data generation servers.")

flags.DEFINE_boolean("share_embedders", True,
                     "Whether we use the same embedders for queries and docs")

flags.DEFINE_boolean("separate_candidate_segments", True,
                     "Whether titles and bodies have separate segment IDs.")

FLAGS = flags.FLAGS


def main(_):
  params = dict(
      batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      bert_hub_module_handle=FLAGS.bert_hub_module_handle,
      embedder_hub_module_handle=FLAGS.embedder_hub_module_handle,
      vocab_path=FLAGS.vocab_path,
      do_lower_case=FLAGS.do_lower_case,
      query_seq_len=FLAGS.query_seq_len,
      candidate_seq_len=FLAGS.candidate_seq_len,
      max_masks=FLAGS.max_masks,
      learning_rate=FLAGS.learning_rate,
      num_input_threads=FLAGS.num_input_threads,
      num_candidates=FLAGS.num_candidates,
      num_train_steps=FLAGS.num_train_steps,
      train_preprocessing_servers=FLAGS.train_preprocessing_servers,
      eval_preprocessing_servers=FLAGS.eval_preprocessing_servers,
      share_embedders=FLAGS.share_embedders,
      separate_candidate_segments=FLAGS.separate_candidate_segments)

  experiment_utils.run_experiment(
      model_fn=model.model_fn,
      train_input_fn=functools.partial(model.input_fn, is_train=True),
      eval_input_fn=functools.partial(model.input_fn, is_train=False),
      params=params,
      params_fname="estimator_params.json",
      exporters=model.get_exporters(params))

  # Write a "done" file from the trainer. As in experiment_utils, we currently
  # use 'use_tpu' as a proxy for whether this is a train or eval node.
  #
  # We could also use the 'type' field in the 'task' of the TF_CONFIG
  # environment variable, but we would generally like to get away from TF_CONFIG
  # in the future.
  #
  # This file is checked for existence by refresh_doc_embeds.
  if experiment_utils.FLAGS.use_tpu:
    model_dir = experiment_utils.EstimatorSettings.from_flags().model_dir
    training_done_filename = os.path.join(model_dir, "TRAINING_DONE")
    with tf.gfile.GFile(training_done_filename, "w") as f:
      f.write("done")


if __name__ == "__main__":
  app.run(main)
