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
"""Identifies overlapping sentences in the Europarl training parallel corpora.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

from absl import app
from absl import flags
from absl import logging

from tensorflow import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("input_data_dir", "", "Input data directory.")

flags.DEFINE_string("output_data_dir", "", "Overlap data directory.")

_ZEROSHOT_DIRECTIONS = [
    ("de", "es"),
    ("de", "fr"),
    ("es", "fr"),
]

_ZEROSHOT_PIVOTS = [
    ("en",),
    ("en",),
]

_OVERLAP_DIRECTIONS = [
    (src, pvt, tgt)
    for (src, tgt), pivots in zip(_ZEROSHOT_DIRECTIONS, _ZEROSHOT_PIVOTS)
    for pvt in pivots
]

random.seed(42)


def _parse_lines(path):
  """Parses lines from UNCorpus dataset."""
  lines = []
  with gfile.GFile(path) as fp:
    for line in fp:
      lines.append(line.strip())
  return lines


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Identify overlaps.
  overlaps = {}
  for src, pvt, tgt in _OVERLAP_DIRECTIONS:
    logging.info("Processing %s-%s-%s...", src, pvt, tgt)
    # Load src-pvt corpus.
    src_pvt_fname = "europarl-v7.{src_lang}-{tgt_lang}.{tgt_lang}".format(
        src_lang=src, tgt_lang=pvt)
    src_pvt_lines = _parse_lines(
      os.path.join(FLAGS.input_data_dir, src_pvt_fname))
    # Load pvt-tgt corpus.
    tgt_pvt_fname = "europarl-v7.{src_lang}-{tgt_lang}.{src_lang}".format(
        src_lang=pvt, tgt_lang=tgt)
    tgt_pvt_lines = _parse_lines(
      os.path.join(FLAGS.input_data_dir, tgt_pvt_fname))
    # Identify overlapping lines and randomly split between src-pvt and pvt-tgt.
    overlap_in_pvt = list(set(src_pvt_lines) & set(tgt_pvt_lines))
    random.shuffle(overlap_in_pvt)
    remove_src_pvt = set(overlap_in_pvt[:len(overlap_in_pvt)//2])
    remove_tgt_pvt = set(overlap_in_pvt[len(overlap_in_pvt)//2:])
    # Save overlaps.
    overlaps[(src, pvt)] = overlaps.get((src, pvt), set()) | remove_src_pvt
    overlaps[(tgt, pvt)] = overlaps.get((tgt, pvt), set()) | remove_tgt_pvt

  for src, pvt in overlaps:
    # Write overlapping lines.
    logging.info("Writing remove.(%s-%s).%s...", src, pvt, pvt)
    for s, t in [(src, pvt), (pvt, src)]:
      fname = "remove.{src}-{tgt}.{lang}".format(src=s, tgt=t, lang=pvt)
      with gfile.GFile(os.path.join(FLAGS.output_data_dir, fname), "w") as fp:
        for line in overlaps[(src, pvt)]:
          fp.write(line + "\n")

  logging.info("Done.")


if __name__ == "__main__":
  app.run(main)
