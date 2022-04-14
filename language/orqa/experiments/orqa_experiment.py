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
"""Entry point for Open-Retrieval Question Answering (ORQA) fine-tuning."""
import functools

from absl import app
from absl import flags
from language.common.utils import experiment_utils
from language.orqa.models import orqa_model
import tensorflow.compat.v1 as tf

flags.DEFINE_integer("retriever_beam_size", 5000,
                     "Retriever beam size.")
flags.DEFINE_integer("reader_beam_size", 5, "Reader beam size.")
flags.DEFINE_float("learning_rate", 1e-5, "Initial learning rate.")
flags.DEFINE_integer("span_hidden_size", 256, "Span hidden size.")
flags.DEFINE_integer("max_span_width", 10, "Maximum span width.")
flags.DEFINE_integer("num_block_records", 13353718, "Number of block records.")
flags.DEFINE_integer("query_seq_len", 64, "Query sequence length.")
flags.DEFINE_integer("block_seq_len", 288, "Document sequence length.")
flags.DEFINE_integer("reader_seq_len", 288 + 32, "Reader sequence length.")
flags.DEFINE_string("reader_module_path",
                    "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
                    "Path to the reader TF-Hub module.")
flags.DEFINE_string("retriever_module_path", None,
                    "Path to the retriever TF-Hub module.")
flags.DEFINE_string("data_root", None, "Data root.")
flags.DEFINE_string("block_records_path", None, "Block records path.")
flags.DEFINE_string("dataset_name", None, "Name of dataset.")

FLAGS = flags.FLAGS


def main(_):
  params = dict(
      data_root=FLAGS.data_root,
      batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      query_seq_len=FLAGS.query_seq_len,
      block_seq_len=FLAGS.block_seq_len,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      retriever_module_path=FLAGS.retriever_module_path,
      reader_module_path=FLAGS.reader_module_path,
      retriever_beam_size=FLAGS.retriever_beam_size,
      reader_beam_size=FLAGS.reader_beam_size,
      reader_seq_len=FLAGS.reader_seq_len,
      span_hidden_size=FLAGS.span_hidden_size,
      max_span_width=FLAGS.max_span_width,
      block_records_path=FLAGS.block_records_path,
      num_block_records=FLAGS.num_block_records)

  train_input_fn = functools.partial(orqa_model.input_fn,
                                     name=FLAGS.dataset_name,
                                     is_train=True)
  eval_input_fn = functools.partial(orqa_model.input_fn,
                                    name=FLAGS.dataset_name,
                                    is_train=False)

  experiment_utils.run_experiment(
      model_fn=orqa_model.model_fn,
      params=params,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      exporters=orqa_model.exporter(),
      params_fname="params.json")


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
