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
r"""Script to pre-process WebNLG human evaluation data for PARENT correlation.

Usage:
  python preprocess_webnlg.py <data_dir> <output_file>

Download the data from https://gitlab.com/shimorina/webnlg-human-evaluation
and place it in <data_dir>.
The processed data will be store in JSON format in <output_file>.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import io
import json
import os
import sys

from collections import defaultdict

data_dir = sys.argv[1]
out_file = sys.argv[2]

with open(os.path.join(data_dir, "all_data_final_averaged.csv")) as f:
  csv_reader = csv.reader(f, delimiter=",")
  next(csv_reader)
  mr_to_team = defaultdict(dict)
  for row in csv_reader:
    row = [entry.decode("utf-8") for entry in row]
    if row[2] == "webnlg": continue
    mr_to_team[row[1]][row[2]] = {
        "pred": row[3],
        "bleu": row[6],
        "meteor": row[7],
        "ter": row[8],
        "fluency": row[11],
        "grammar": row[12],
        "semantics": row[13],
        }

def _read_file(filename):
  items = []
  with io.open(filename) as f:
    for line in f:
      items.append(line.strip())
  return items

mrs = _read_file(os.path.join(data_dir, "MRs.txt"))
mr_to_index = {m: i for i, m in enumerate(mrs)}
ref0 = _read_file(os.path.join(data_dir, "gold-sample-reference0.lex"))
ref1 = _read_file(os.path.join(data_dir, "gold-sample-reference1.lex"))
ref2 = _read_file(os.path.join(data_dir, "gold-sample-reference2.lex"))


def _parse_mr(mr):
  return [item.split(" | ") for item in mr.split("<br>")]

output = []
for mr, teams in mr_to_team.iteritems():
  table = _parse_mr(mr)
  ref_i = mr_to_index[mr]
  references = [ref for ref in [ref0[ref_i], ref1[ref_i], ref2[ref_i]] if ref]
  out = {
      "table": table,
      "references": references,
      }
  for team, scores in teams.iteritems():
    for key, val in scores.iteritems():
      out[team + "-" + key] = val
  output.append(out)

json.dump(output, open(out_file, "w"))
