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
"""Entry point for Inverse Cloze Task (ICT) pre-training."""
import functools

from absl import app
from absl import flags
from language.common.utils import experiment_utils
from language.orqa.models import ict_model
import tensorflow.compat.v1 as tf

flags.DEFINE_string("bert_hub_module_path",
                    "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
                    "Path to the BERT TF-Hub module.")
flags.DEFINE_integer("query_seq_len", 64, "Query sequence length.")
flags.DEFINE_integer("block_seq_len", 288, "Document sequence length.")
flags.DEFINE_integer("projection_size", 128, "Projection size.")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate.")
flags.DEFINE_integer("num_block_records", 13353718, "Number of block records.")
flags.DEFINE_string("examples_path", None, "Input examples path")
flags.DEFINE_integer("num_input_threads", 12, "Num threads for input reading.")
flags.DEFINE_float("mask_rate", 0.9, "Mask rate.")

FLAGS = flags.FLAGS


def main(_):
  params = dict(
      batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      bert_hub_module_path=FLAGS.bert_hub_module_path,
      query_seq_len=FLAGS.query_seq_len,
      block_seq_len=FLAGS.block_seq_len,
      projection_size=FLAGS.projection_size,
      learning_rate=FLAGS.learning_rate,
      examples_path=FLAGS.examples_path,
      mask_rate=FLAGS.mask_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_block_records=FLAGS.num_block_records,
      num_input_threads=FLAGS.num_input_threads)
  experiment_utils.run_experiment(
      model_fn=ict_model.model_fn,
      train_input_fn=functools.partial(ict_model.input_fn, is_train=True),
      eval_input_fn=functools.partial(ict_model.input_fn, is_train=False),
      exporters=ict_model.exporter(),
      params=params)

if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
