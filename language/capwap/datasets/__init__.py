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
"""General dataset functions and definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.capwap.utils import image_utils
from language.capwap.utils import text_utils
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator


class DatasetTypes(object):
  """An Enum for dataset types."""
  NONE = -1
  REFERENCE = 0
  VQA = 1
  GUIDED = 2


def footprint(params):
  """Union of dataset features for multi-tasking.

  Combining datasets is only possible if each has the same input features
  footprint, so here we define a default features dictionary with dummy inputs
  that will be overwritten as necessary by client datasets.

  Args:
    params: Dictionary of model settings.

  Returns:
    features: Dictionary of TF tensors.
  """
  question_length = params.get("question_length", 1)
  answer_length = params.get("answer_length", 1)
  caption_length = params.get("caption_length", 1)
  condition_length = params.get("condition_length", 1)
  num_image_regions = params.get("num_image_regions", 1)
  image_feature_size = params.get("image_feature_size", 1)
  features = dict(
      input_type=DatasetTypes.NONE,
      image_id=np.int64(0),
      question_id=np.int64(-1),
      question_inputs=text_utils.TextInputs(
          token_ids=np.zeros([question_length], dtype=np.int32),
          mask=np.zeros([question_length], dtype=np.int32),
          segment_ids=np.zeros([question_length], dtype=np.int32),
          positions=np.zeros([question_length], dtype=np.int32)),
      answer_inputs=text_utils.TextInputs(
          token_ids=np.zeros([answer_length], dtype=np.int32),
          mask=np.zeros([answer_length], dtype=np.int32),
          segment_ids=np.zeros([answer_length], dtype=np.int32),
          positions=np.zeros([answer_length], dtype=np.int32)),
      answer_outputs=text_utils.TextOutputs(
          token_ids=np.zeros([answer_length - 1], dtype=np.int32),
          mask=np.zeros([answer_length - 1], dtype=np.int32)),
      token_inputs=text_utils.TextInputs(
          token_ids=np.zeros([caption_length - 1], dtype=np.int32),
          mask=np.zeros([caption_length - 1], dtype=np.int32),
          segment_ids=np.zeros([caption_length - 1], dtype=np.int32),
          positions=np.zeros([caption_length - 1], dtype=np.int32)),
      token_outputs=text_utils.TextOutputs(
          token_ids=np.zeros([caption_length - 1], dtype=np.int32),
          mask=np.zeros([caption_length - 1], dtype=np.int32)),
      object_features=image_utils.ObjectDetectionOutput(
          features=np.zeros([num_image_regions, image_feature_size],
                            dtype=np.float32),
          positions=np.zeros([num_image_regions], dtype=np.int32),
          mask=np.zeros([num_image_regions], dtype=np.int32)))
  if params.get("conditional_decoding"):
    features["condition_inputs"] = text_utils.TextInputs(
        token_ids=np.zeros([condition_length], dtype=np.int32),
        mask=np.zeros([condition_length], dtype=np.int32),
        segment_ids=np.zeros([condition_length], dtype=np.int32),
        positions=np.zeros([condition_length], dtype=np.int32))

  return features


def input_fn(params, mode, get_dataset_fns, weights=None, mix_batches=True):
  """Wrapper for getting a dataset and batching it."""
  # Load datasets.
  datasets = [fn(params, mode) for fn in get_dataset_fns]

  # Weird bug where tensor2tensor doens't like non-static shapes on CPU,
  # but is ok when it's on TPU. Luckily we evaluate on TPU in our case.
  if mode != tf_estimator.ModeKeys.PREDICT:
    drop_remainder = True
  elif not params["use_tpu"]:
    tf.logging.warning("Evaluation intended for TPU! Will have to drop "
                       "remainder in batching. Results may be slightly off "
                       "(or use predict_batch_size=1)")
    drop_remainder = True
  else:
    drop_remainder = False

  # Normalize weights.
  if weights is not None:
    weights = np.array(weights).astype(np.float32)
    weights = weights / np.sum(weights)

  # Batch first, mix later.
  if not mix_batches:
    datasets = [
        dset.batch(params["batch_size"], drop_remainder=drop_remainder)
        for dset in datasets
    ]
    dataset = tf.data.experimental.sample_from_datasets(datasets, weights)

  # Mix first, batch later.
  else:
    dataset = tf.data.experimental.sample_from_datasets(datasets, weights)
    dataset = dataset.batch(params["batch_size"], drop_remainder=drop_remainder)

  dataset = dataset.prefetch(params["prefetch_batches"])

  return dataset
