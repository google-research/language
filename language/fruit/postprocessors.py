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
"""SeqIO postprocessors for wikidiff tasks."""

import json

from language.fruit import tf_utils
import tensorflow as tf


@tf.autograph.experimental.do_not_convert
def postprocess_wikidiff(
    output,
    vocabulary,
    normalize_fn,
    is_target=False,
    example=None,
):
  """Applies normalization to outputs."""
  del is_target
  inputs = tf_utils.maybe_decode(
      vocabulary.decode_tf(example["inputs"]).numpy())
  targets = tf_utils.maybe_decode(output)
  normalized_inputs, normalized_targets = normalize_fn(inputs, targets)
  results = {
      "inputs":
          inputs,
      "targets":
          targets,
      "normalized_inputs":
          normalized_inputs,
      "normalized_targets":
          normalized_targets,
      "generatable_surfaces":
          json.loads(tf_utils.maybe_decode(example["generatable_surfaces"])),
  }
  return results
