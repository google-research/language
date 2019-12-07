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
"""Sentencize the raw wikitext103."""

import tensorflow as tf

app = tf.compat.v1.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string("wiki103_raw", None,
                    "Path to raw wikitext103 train corpus.")
flags.DEFINE_string("output_path", None,
                    "Path to output the processed dataset.")

FLAGS = flags.FLAGS


def main(_):
  with open(FLAGS.wiki103_raw, "r") as f:
    data = f.read().strip().split("\n")

  data = [x.split(" . ") for x in data if x.strip() and x.strip()[0] != "="]

  sentences = []
  for para in data:
    for sent in para:
      sentences.append(sent + ".")
  data = "\n".join(sentences)

  data = data.replace(" @.@ ", ".").replace(" @-@ ", "-").replace(" ,", ",")
  data = data.replace(" \'", "\'").replace(" )", ")").replace("( ", "(")
  data = data.replace(" ;", ";")

  data = "\n".join([x for x in data.split("\n") if len(x.split()) > 3])

  logging.info("length = %d", len(data.split("\n")))

  with open(FLAGS.output_path, "w") as f:
    f.write(data)


if __name__ == "__main__":
  app.run(main)
