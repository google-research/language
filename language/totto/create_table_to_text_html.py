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
# Lint as: python3
"""Creates HTML for annotated examples for easy visualization."""
import json
import os

from absl import app
from absl import flags

from language.totto import table_to_text_html_utils
import six

flags.DEFINE_string("input_path", None, "Input json file.")

flags.DEFINE_string("output_dir", None, "Output directory.")

flags.DEFINE_integer("examples_to_visualize", 100,
                     "Number of examples to visualize.")

FLAGS = flags.FLAGS


def main(_):
  input_path = FLAGS.input_path
  output_dir = FLAGS.output_dir
  with open(input_path, "r", encoding="utf-8") as input_file:
    index = 0
    for line in input_file:
      line = six.ensure_text(line, "utf-8")
      json_example = json.loads(line)
      html_str = table_to_text_html_utils.get_example_html(json_example)
      output_path = os.path.join(output_dir, "example-" + str(index) + ".html")
      with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(html_str + "\n")
      index += 1
      if index % 100 == 0:
        print("Processed %d examples" % index)
      if index >= FLAGS.examples_to_visualize:
        break

  print("Num examples processed: %d" % index)


if __name__ == "__main__":
  flags.mark_flags_as_required(["input_path", "output_dir"])
  app.run(main)
