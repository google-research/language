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
"""Replace T5 SPM OOV character with `<`.

Certain punctuation characters are mapped to the OOV symbol in T5's
sentence-piece model. For Spider, this appears to only affect the `<` symbol,
so it can be deterministically recovered by running this script.

An alternative is to preprocess dataset to avoid OOV symbols for T5.
"""

from absl import app
from absl import flags

from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input txt file.")

flags.DEFINE_string("output", "", "Output txt file.")


def main(unused_argv):
  with gfile.Open(FLAGS.output, "w") as output_file:
    with gfile.Open(FLAGS.input, "r") as input_file:
      for line in input_file:
        pred = line.replace(" ⁇ ", "<")
        if line != pred:
          print("Original: %s" % line)
          print("New: %s" % pred)
        output_file.write("%s" % pred)


if __name__ == "__main__":
  app.run(main)
