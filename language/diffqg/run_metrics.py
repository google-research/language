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
"""Binary to read annotations and compute metrics over the predictions."""

import json
from typing import Sequence

from absl import app
from absl import flags
from language.diffqg import annotation
from language.diffqg import metrics


_GOLD_ANNOTATIONS = flags.DEFINE_string(
    "gold_annotations",
    None,
    "Path to the final jsonl file containing the human annotations.",
    required=True,
)

_PRED_ANNOTATIONS = flags.DEFINE_string(
    "predicted_annotations",
    None,
    "Path to predicted annotations.",
    required=True,
)

_OUTPUT_SCORES = flags.DEFINE_string(
    "output_scores",
    "",
    (
        "Path to write the individual scored annotations. If blank, they won't"
        " be written, and only aggregated metrics will be printed."
    ),
)

_OUTPUT_METRICS = flags.DEFINE_string(
    "output_metrics",
    "",
    (
        "Path to write aggregated metrics for the predictions. If blank, they"
        " won't be written but will still be printed."
    ),
)

_BLEURT_CHECKPOINT = flags.DEFINE_string(
    "bleurt_checkpoint",
    "",
    (
        "Path to the BLEURT checkpoint. If not specified, BLEURT metrics will"
        " not be computed, which will make metric computation faster"
    ),
)

_RUN_QSIM = flags.DEFINE_bool(
    "run_qsim",
    False,
    (
        "Whether or not to run query similarity metrics. Skipping them will"
        " make metric computation faster."
    ),
)

_QSIM_MODEL = flags.DEFINE_string(
    "qsim_model_name",
    "cross-encoder/quora-roberta-large",
    (
        "Name of the query similarity model to load from hugging face. This"
        " should not be changed during official metrics computation."
    ),
)

_QSIM_THRESHOLD = flags.DEFINE_float(
    "qsim_threshold",
    0.5,
    (
        "Float value to determine if a query is a duplicate or not. This should"
        " not be changed in official metrics computation."
    ),
)

_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    1,
    (
        "Number of examples to run at once. Increasing this will make the"
        " metrics computation faster but more memory intensive."
    ),
)

_NUM_BATCHES = flags.DEFINE_integer(
    "num_batches", 0, "Number of batches to run. At 0, will run all batches."
)

_SUBSETS = {
    "all": lambda _: True,
    "tp": lambda s: s.score.label == metrics.Label.TRUE_POSITIVE,
    "pos": lambda s: s.score.label.is_positive(),
    "human": lambda s: s.paired_annotation.is_edited,
}


def main(unused_argv: Sequence[str]) -> None:
  paired_annotations = annotation.make_paired_annotations(
      _GOLD_ANNOTATIONS.value, _PRED_ANNOTATIONS.value
  )

  bleurt_checkpoint = (
      _BLEURT_CHECKPOINT.value if _BLEURT_CHECKPOINT.value else None
  )
  if not bleurt_checkpoint:
    print("Not running BLEURT as no checkpoint was provided.")
  qsim_model_name = (
      _QSIM_MODEL.value if (_QSIM_MODEL.value and _RUN_QSIM.value) else None
  )
  if not qsim_model_name:
    print("Not running QSIM as no model was provided or flag was set to False.")

  # We need to consume the list a few times to calculate filtered metrics as
  # well as write.
  scored_annotations = list(
      metrics.score_annotations(
          paired_annotations,
          qsim_model_name,
          bleurt_checkpoint,
          _BATCH_SIZE.value,
          _QSIM_THRESHOLD.value,
          _NUM_BATCHES.value,
      )
  )
  aggregated_metrics = metrics.calculate_metrics(scored_annotations, _SUBSETS)
  print(aggregated_metrics)

  if _OUTPUT_SCORES.value:
    print(f"Writing scored examples to {_OUTPUT_SCORES.value}")
    with open(_OUTPUT_SCORES.value, "wt") as fw:
      for anno in scored_annotations:
        fw.write(f"{anno}\n")
  if _OUTPUT_METRICS.value:
    print(f"Writing metrics to {_OUTPUT_METRICS.value}")
    with open(_OUTPUT_METRICS.value, "wt") as fw:
      fw.write(json.dumps(aggregated_metrics))


if __name__ == "__main__":
  app.run(main)
