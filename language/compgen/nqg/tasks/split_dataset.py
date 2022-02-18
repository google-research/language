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
"""Split tsv dataset file based on predefined sets of example ids."""

import json
import os

from absl import app
from absl import flags

from language.compgen.nqg.tasks import tsv_utils

from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_string("split", "", "Json split file.")

flags.DEFINE_string("output_dir", "", "Output directory for dataset files.")


def load_splits():
  """Reads a JSON file containing split IDs.

  Returns:
    A dictionary where keys are a split name (e.g. `train` or `test`) and values
    are a list of integer example IDs.
  """
  with gfile.GFile(FLAGS.split, "r") as reader:
    text = reader.read()
  splits = json.loads(text)
  return splits


def main(unused_argv):
  splits = load_splits()
  examples = tsv_utils.read_tsv(FLAGS.input)
  example_id_to_example = {
      example_id: example for example_id, example in enumerate(examples)
  }

  for split, split_ids in splits.items():
    examples = []
    for split_id in split_ids:
      examples.append(example_id_to_example[split_id])
    filename = os.path.join(FLAGS.output_dir, "%s.tsv" % split)
    tsv_utils.write_tsv(examples, filename)


if __name__ == "__main__":
  app.run(main)
