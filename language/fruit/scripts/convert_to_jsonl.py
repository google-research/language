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
r"""Converts a Wikipedia XML dump to JSONL.

There are two reasons we do this:
  1. To filter down the dump to articles and redirects.
  2. JSONL is much more amenable to parallel processing.
"""
import bz2
import json
import os

from absl import app
from absl import flags
from absl import logging
from language.fruit import wiki_utils
import tqdm

flags.DEFINE_string("input_xml", None, "Input compressed XML file.")
flags.DEFINE_string("output_jsonl", None, "Ouput JSONL file.")

FLAGS = flags.FLAGS


def main(_):
  openers = {
      ".xml": open,
      ".bz2": bz2.open,
  }
  filetype = os.path.splitext(FLAGS.input_xml)[-1]
  opener = openers.get(filetype, open)
  logging.info("processing file: %s", FLAGS.input_xml)
  logging.info("filetype: %s", filetype)
  with opener(FLAGS.input_xml, "r") as xml_file, \
      open(FLAGS.output_jsonl, "w") as jsonl_file:
    for page in tqdm.tqdm(wiki_utils.generate_pages(xml_file)):
      page_json = json.dumps(page, ensure_ascii=False)
      print(page_json, file=jsonl_file)


if __name__ == "__main__":
  app.run(main)
