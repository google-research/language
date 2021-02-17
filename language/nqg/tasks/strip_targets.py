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
"""Strip targets from a tsv file and write as newline-separated txt.

This file can be useful as input to generate predictions (e.g. for evaluation).
"""

from absl import app
from absl import flags

from language.nqg.tasks import tsv_utils

from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_string("output", "", "Output txt file.")

flags.DEFINE_string("prefix", "", "Optional prefix to prepend to source.")


def main(unused_argv):
  examples = tsv_utils.read_tsv(FLAGS.input)
  with gfile.GFile(FLAGS.output, "w") as txt_file:
    for example in examples:
      txt_file.write("%s%s\n" % (FLAGS.prefix, example[0]))


if __name__ == "__main__":
  app.run(main)
