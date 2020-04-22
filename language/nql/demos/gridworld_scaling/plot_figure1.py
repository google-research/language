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
"""Public version of code to reproduce experiments in a paper.

Paper is 'Scalable Neural Methods for Reasoning With a Symbolic Knowledge Base',
ICLR 2020 https://openreview.net/forum?id=BJlguT4YPr
"""
from __future__ import print_function
import os
import sys

# make sure matplotlib works without display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

DATA_STEM = sys.argv[1]
DATA_DIR = sys.argv[2]


def qps_data(filename, squarex=False):
  """Load queries per second data from output produced by scaling_eval.py.

  Args:
    filename: string name of file
    squarex: if true, square the x coordinates in the plot.

  Returns:
    x list of values of the x-axis variable (whatever was swept through).
      If squarex = True then these values are squared.
    y list of queries per second
  """
  x = []
  y = []
  for k, line in enumerate(open(DATA_STEM + "_" + filename)):
    if k > 0:  # skip header line
      parts = line.strip().split("\t")
      x.append(float(parts[0]))
      y.append(float(parts[1]))
  if squarex:
    x = [z * z for z in x]
  return x, y


# draw leftmost plot

plt.figure(1, figsize=(14, 4.5))
plt.subplot(131)
plt.title("Queries/sec (4 relations)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Num Entities")
x1, y1 = qps_data("reified_kb_vary_n.tsv", squarex=True)
x2, y2 = qps_data("late_mix_vary_n.tsv", squarex=True)
x3, y3 = qps_data("naive_vary_n.tsv", squarex=True)
plt.plot(x1, y1, color="r", label="reified")
plt.plot(x2, y2, color="b", label="late mixing")
plt.plot(x3, y3, color="k", label="naive")

# these statistics were discussed in text of the paper

print(("speedup of mix vs reified as you vary grid size n",
       [a / b for (a, b) in zip(y2, y1)]))
print(("speedup of mix vs naive you vary grid size n",
       [a / b for (a, b) in zip(y2, y3)]))

# draw middle plot

plt.subplot(132)
plt.title("Queries/sec (10k entities)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Num Relations")
x1, y1 = qps_data("reified_kb_vary_extra_rels.tsv", squarex=False)
x2, y2 = qps_data("late_mix_vary_extra_rels.tsv", squarex=False)
x3, y3 = qps_data("naive_vary_extra_rels.tsv", squarex=False)
plt.plot(x1, y1, color="r", label="reified")
plt.plot(x2, y2, color="b", label="late mixing")
plt.plot(x3, y3, color="k", label="naive")

# these statistics were discussed in text of the paper

print("speedup of reified vs mix as you vary #rels",
      [a / b for (a, b) in zip(y1, y2)])
print("speedup of reified vs naive you vary #rels",
      [a / b for (a, b) in zip(y1, y3)])

# draw rightmost plot

y2 = [a / b for (a, b) in zip(y1, y2)]
y3 = [a / b for (a, b) in zip(y1, y3)]

plt.subplot(133)
plt.title("Reified KB speedup")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Num Relations")
plt.grid(True)
plt.plot(x2, y2, color="b", label="vs late mixing")
plt.plot(x3, y3, color="k", label="vs naive")
plt.legend()

outfile = os.path.join(DATA_DIR, "figure1.png")

print("saving to", outfile)
plt.draw()
plt.savefig(outfile, format="png")
