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
"""Computes BLEURT (https://arxiv.org/abs/2004.04696) for ToTTo dataset.

To handle multiple references, scores are computed for each reference separately
and then averaged.
"""
import glob
import io
import os

from absl import app
from absl import flags

from bleurt import score
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "reference_path", None, "Text file containing references, one per line. "
    "Multiple references should be in different files named as: "
    "'<reference_path>0', '<reference_path>1', '<reference_path>2'.")

flags.DEFINE_string("generation_path", None,
                    "Text file containing generations, one per line.")


def _text_reader(text_file):
  """Returns list of lines from the text file.

  Performs lowercasing and white-space tokenization on each line before
  returning. Slightly different than the one in totto_parent_eval.py.

  Args:
    text_file: String filename.
  """
  texts = []
  with io.open(text_file, encoding="utf-8") as f:
    for line in f:
      line = line.strip().lower()
      texts.append(line)
  return texts


def _text_reference_reader(text_file):
  """Returns list of references in the multi-reference setting.

  Performs lowercasing on each line before returning. Note this file
  is slightly different than the one in totto_parent_eval.py due to how
  we create the list of references.

  Args:
    text_file: String filename.
  """
  single_reference_exists = os.path.isfile(text_file)

  # Check for multi-references.
  multi_reference_paths = glob.glob(text_file + "[0-9]")

  # Either the file should exist or it should correspond to multiple reference
  # files but not both.
  assert ((single_reference_exists or multi_reference_paths) and
          (not single_reference_exists or not multi_reference_paths))

  if single_reference_exists:
    references = _text_reader(text_file)
    return [references]
  else:
    multi_reference_paths = glob.glob(text_file + "[0-9]")
    assert len(multi_reference_paths) == 3
    references0 = _text_reader(multi_reference_paths[0])
    references1 = _text_reader(multi_reference_paths[1])
    references2 = _text_reader(multi_reference_paths[2])

    assert len(references0) == len(references1)
    assert len(references0) == len(references2)
    return [references0, references1, references2]


def main(_):
  multi_references = (_text_reference_reader(FLAGS.reference_path))
  generations = _text_reader(FLAGS.generation_path)

  # FLAGS.bleurt_checkpoint is defined in the BLEURT library. Importing the
  # BLEURT scoring module automatically imports the flags.
  scorer = score.BleurtScorer(FLAGS.bleurt_checkpoint)
  multi_bleurt_scores = []

  for references in multi_references:
    assert len(references) == len(generations)

    # Maximize parallelism.
    bleurt_scores = scorer.score(references=references, candidates=generations)
    multi_bleurt_scores.append(bleurt_scores)

  if len(multi_references) == 1:
    avg_bleurt_score = np.mean(multi_bleurt_scores[0])
  else:
    assert len(multi_references) == 3
    avg_bleurt_scores = []
    for i in range(len(generations)):
      # All examples have atleast two references but some do not have three.
      assert multi_references[0][i] and multi_references[1][i]
      r2 = multi_references[2][i]
      if r2:
        # Take average over 3 references.
        score_i = (multi_bleurt_scores[0][i] + multi_bleurt_scores[1][i] +
                   multi_bleurt_scores[2][i]) / 3
      else:
        print("only two refs")
        # Take average over two references.
        score_i = (multi_bleurt_scores[0][i] + multi_bleurt_scores[1][i]) / 2
      avg_bleurt_scores.append(score_i)
    avg_bleurt_score = np.mean(avg_bleurt_scores)

  print("Evaluated %d examples." % len(generations))
  print("Average BLEURT score = %.4f" % avg_bleurt_score)


if __name__ == "__main__":
  flags.mark_flags_as_required([
      "reference_path",
      "generation_path",
      "bleurt_checkpoint",
  ])
  app.run(main)
