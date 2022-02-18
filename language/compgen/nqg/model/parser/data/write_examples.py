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
"""Write tf.Example protos for model training.

This requires a dataset tsv file and a set of QCFG rules as input.
"""

import os

from absl import app
from absl import flags

from language.compgen.nqg.model.parser import config_utils
from language.compgen.nqg.model.parser.data import example_converter
from language.compgen.nqg.model.parser.data import tokenization_utils
from language.compgen.nqg.model.qcfg import qcfg_file

from language.compgen.nqg.tasks import tsv_utils

import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_string("output", "", "Output TF example file.")

flags.DEFINE_string("bert_dir", "", "Directory for BERT, including vocab file.")

flags.DEFINE_string("config", "", "Config file.")

flags.DEFINE_string("rules", "", "Input rules file.")

flags.DEFINE_integer("offset", 0, "Start index for examples to process.")

flags.DEFINE_integer("limit", 0, "End index for examples to process if >0.")


def main(unused_argv):
  config = config_utils.json_file_to_dict(FLAGS.config)
  examples = tsv_utils.read_tsv(FLAGS.input)
  rules = qcfg_file.read_rules(FLAGS.rules)
  tokenizer = tokenization_utils.get_tokenizer(
      os.path.join(FLAGS.bert_dir, "vocab.txt"))
  converter = example_converter.ExampleConverter(rules, tokenizer, config)

  total_written = 0
  writer = tf.io.TFRecordWriter(FLAGS.output)
  for idx, example in enumerate(examples):
    if FLAGS.offset and idx < FLAGS.offset:
      continue
    if FLAGS.limit and idx >= FLAGS.limit:
      break
    print("Processing example %s." % idx)

    tf_example = converter.convert(example)
    writer.write(tf_example.SerializeToString())
    total_written += 1

  converter.print_max_sizes()
  print("Wrote %d examples." % total_written)


if __name__ == "__main__":
  app.run(main)
