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
# coding=utf-8
"""Script to extract plain text and hyperlinked mentions from wiki dump."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bz2
import fnmatch
import json
import os
import re
import urllib.parse
from absl import flags
import tensorflow.compat.v1 as tf
from tqdm import tqdm

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("base_dir", None, "Wiki dump from HotpotQA")

flags.DEFINE_string("output_file", None,
                    "Output file as jsonl to store passages.")

flags.DEFINE_boolean("debug", False,
                     "If true, only preprocess a small number of passages.")


def _process_item(item):
  """Process single wiki abstract."""
  paragraph = "".join(item["text_with_links"])
  context = ""
  current_mention = None
  all_mentions = []
  pos = 0
  for is_, sent_offsets in enumerate(item["charoffset_with_links"]):
    for st, en in sent_offsets:
      token = paragraph[st:en]
      if token.startswith("<a href="):
        link = re.match('<a href=\"(.*)\">', token)
        if link is None:
          print("Could not identify %s" % token)
          continue
        current_mention = [link.groups()[0], pos, en]
      elif token == "</a>":
        if current_mention is None:
          print("Loose end of hyperlink", paragraph[st - 20:en + 10])
          continue
        my_text = context[current_mention[1]:].rstrip()
        all_mentions.append({
            "kb_id": current_mention[0],
            "start": current_mention[1],
            "text": my_text,
            "sent_id": is_,
        })
        current_mention = None
      else:
        context += token + " "
        pos += len(token) + 1
  context = context.rstrip()
  m = re.search(re.escape(item["title"]), context[:100], re.IGNORECASE)
  if m is not None:
    all_mentions = [{
        "kb_id": urllib.parse.quote(item["title"]),
        "start": m.start(),
        "text": context[m.start():m.end()],
        "sent_id": 0,
    }] + all_mentions
  return {
      "id": item["id"],
      "title": item["title"],
      "url": item["url"],
      "kb_id": urllib.parse.quote(item["title"]),
      "context": context,
      "mentions": all_mentions,
  }


def _process_file(filename, kbid2id, num_mentions, fo):
  with bz2.BZ2File(filename, "r") as f:
    for line in f:
      processed_item = _process_item(json.loads(line.strip()))
      assert processed_item["kb_id"] not in kbid2id
      kbid2id[processed_item["kb_id"]] = (processed_item["id"],
                                          processed_item["title"])
      num_mentions[0] += len(processed_item["mentions"])
      fo.write(json.dumps(processed_item) + "\n")


def main(_):
  all_files = []
  for root, _, filenames in os.walk(FLAGS.base_dir):
    for filename in fnmatch.filter(filenames, "*.bz2"):
      all_files.append(os.path.join(root, filename))
  print("Found %d matching files" % len(all_files))

  kbid2id = {}
  num_mentions = [0]
  with open(FLAGS.output_file, "w") as fo:
    for ii in tqdm(range(len(all_files))):
      _process_file(all_files[ii], kbid2id, num_mentions, fo)
      if FLAGS.debug and ii == 100:
        break
  print("total %d articles %d mentions" % (len(kbid2id), num_mentions[0]))


if __name__ == "__main__":
  flags.mark_flag_as_required("in_file")
  flags.mark_flag_as_required("out_file")
  tf.app.run()
