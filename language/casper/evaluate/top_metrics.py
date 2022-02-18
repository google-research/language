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
r"""Metrics for TOP and MTOP parses."""
from typing import Dict, List

from language.casper.utils import top_utils


def _safe_divide(x, y):
  return x / y if y != 0 else 0.0


def top_metrics(targets: List[str], predictions: List[str]) -> Dict[str, float]:
  """Returns eval metrics for TOP and MTOP datasets."""
  num_correct = 0
  num_total = 0
  num_invalid = 0

  num_intent_correct = 0
  num_frame_correct = 0

  for target, predicted in zip(targets, predictions):
    if target == predicted:
      num_correct += 1
    num_total += 1

    target_lf = top_utils.deserialize_top(target)
    predicted_lf = top_utils.deserialize_top(predicted)

    assert target_lf is not None
    if not predicted_lf:
      num_invalid += 1
      continue

    target_frame = top_utils.get_frame_top(target_lf)
    predicted_frame = top_utils.get_frame_top(predicted_lf)
    target_intent = target_frame.split("-")[0]
    predicted_intent = predicted_frame.split("-")[0]

    num_intent_correct += int(predicted_intent == target_intent)
    num_frame_correct += int(predicted_frame == target_frame)

  return dict(
      num_total=num_total,
      full_accuracy=_safe_divide(num_correct, num_total),
      intent_accuracy=_safe_divide(num_intent_correct, num_total),
      intent_arg_accuracy=_safe_divide(num_frame_correct, num_total),
      invalid_predictions=_safe_divide(num_invalid, num_total))
