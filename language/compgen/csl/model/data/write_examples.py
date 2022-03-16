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

from absl import app
from absl import flags
from language.compgen.csl.common import json_utils
from language.compgen.csl.model.data import example_converter
from language.compgen.csl.qcfg import qcfg_file
from language.compgen.nqg.tasks import tsv_utils
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_string("output", "", "Output TF example file.")

flags.DEFINE_string("config", "", "Config file.")

flags.DEFINE_string("rules", "", "Input rules file.")

flags.DEFINE_integer("offset", 0, "Start index for examples to process.")

flags.DEFINE_integer("limit", 0, "End index for examples to process if >0.")

flags.DEFINE_bool("verbose", False, "Whether to print debug output.")


def main(unused_argv):
  config = json_utils.json_file_to_dict(FLAGS.config)
  examples = tsv_utils.read_tsv(FLAGS.input)
  rules = qcfg_file.read_rules(FLAGS.rules)
  converter = example_converter.ExampleConverter(rules, config)

  slice_start = FLAGS.offset
  slice_end = FLAGS.limit if FLAGS.limit else None
  examples = examples[slice(slice_start, slice_end)]

  num_examples = 0
  writer = tf.io.TFRecordWriter(FLAGS.output)
  for idx, example in enumerate(examples):
    tf_example = converter.convert(example)
    writer.write(tf_example.SerializeToString())
    num_examples += 1
    if FLAGS.verbose:
      print("Processing example %s." % idx)
      print("(%s, %s)" % (example[0], example[1]))

  converter.print_max_sizes()
  print("Wrote %d examples." % num_examples)


if __name__ == "__main__":
  app.run(main)
