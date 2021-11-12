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
r"""Converts MTOP data from TSV to JSONL.

Example JSON output (formatted into multiple lines here for clarity):
{
  "hashed_id": "en-train-0",
  "orig_query": "Has Angelika Kratzer video messaged me?",
  "input_str": "Has Angelika Kratzer video messaged me ?",
  "output_str": "[IN:GET_MESSAGE [SL:CONTACT Angelika Kratzer ] ... ",
  "lang": "en",
  "domain": "messaging"
}
"""
import collections
import json
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("infile", None, "Input TSV filename")
flags.DEFINE_string("outfile", None, "Output JSONL filename")

SPLIT_NAMES = {
    "train.txt": "train",
    "eval.txt": "dev",
    "test.txt": "test",
}

MTopEx = collections.namedtuple(
    "MTopEx",
    ["exid", "intent", "args", "query", "domain", "locale", "output", "tokens"])


def tsv_to_jsonl(infile, outfile, split):
  """Converts TSV file to JSONL.

  Args:
    infile: Input TSV filename.
    outfile: Output JSONL filename.
    split: The split name (e.g., "train" or "dev"). Used in example IDs.
  """
  count = 0
  with tf.io.gfile.GFile(infile) as fin:
    with tf.io.gfile.GFile(outfile, "w") as fout:
      for i, line in enumerate(fin):
        fields = line.rstrip("\n").split("\t")
        ex = MTopEx(*fields)
        lang = ex.locale.split("_")[0]  # en_XX --> en
        unique_id = "{}-{}-{}".format(lang, split, i)
        entry = {
            "hashed_id": unique_id,
            "orig_query": ex.query,
            "input_str": " ".join(json.loads(ex.tokens)["tokens"]),
            "output_str": ex.output,
            "lang": lang,
            "domain": ex.domain,
        }
        print(json.dumps(entry), file=fout)
        count += 1
  logging.info("Converted %d examples to %s", count, outfile)


def main(_):
  filename = os.path.basename(FLAGS.infile)
  split = SPLIT_NAMES[filename]
  tsv_to_jsonl(FLAGS.infile, FLAGS.outfile, split)


if __name__ == "__main__":
  app.run(main)
