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
"""Utility functions for dealing with images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from language.capwap.utils import tensor_utils
import numpy as np
import tensorflow.compat.v1 as tf

GRID_SIZE = 8

ObjectDetectionOutput = collections.namedtuple(
    "ObjectDetectionOutput", ["features", "positions", "mask"])


class ImageMetadata(object):
  """Wrapper with defaults around image metadata (for both captions and QA)."""

  @staticmethod
  def _get_length(*args):
    """Helper to establish length of arguments."""
    length = 0
    for a in args:
      if a:
        length = len(a)
    return length

  def __init__(
      self,
      image_id=-1,
      question_ids=None,
      questions=None,
      answers=None,
      captions=None,
      objects=None,
  ):
    """Initialization.

    Args:
      image_id: (optional) Identifier for the image.
      question_ids: (optional) List of unique identifiers for the questions.
      questions: (optional) List of question strings.
      answers: (optional) List of answer strings.
      captions: (optional) List of caption strings.
      objects: (optional) Filename of HDF5 file with R-CNN data. Could also be
        an open filehandle.
    """
    self.image_id = image_id
    length = self._get_length(question_ids, questions, answers, captions)
    self.question_ids = question_ids or [-1] * length
    self.questions = questions or ["_"] * length
    self.answers = answers or ["_"] * length
    self.captions = captions or ["_"] * length
    self.objects = objects


def quantize_bbox(bboxes, height, width, grid_size=GRID_SIZE):
  """Transform bounding box coordinates into a position in an N x N grid.

  Args:
    bboxes: <float32> [batch, 4] Object bounding boxes, each with (x_min, y_min,
      x_max, y_max).
    height: Overall image height.
    width: Overall image width.
    grid_size: Partition image into a grid_size x grid_size raster.

  Returns:
    position: <int32> [batch] Grid index of the center of the bounding box.
              Total number of indices is grid_size * grid_size.
  """
  if max(bboxes[:, 2]) > width:
    raise ValueError("Max x should be less than width.")
  if max(bboxes[:, 3]) > height:
    return ValueError("Max y should be less than height.")
  center_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
  center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2
  raster_x = np.floor(center_x / width * grid_size)
  raster_y = np.floor(center_y / height * grid_size)
  position = raster_x * grid_size + raster_y
  return position.astype(np.int64)


def parse_object_features(features, positions, params):
  """Parse ObjectDetectionOutput from TensorProtos."""
  features = tf.io.parse_tensor(features, tf.float32)
  positions = tf.io.parse_tensor(positions, tf.int64)
  positions = tf.cast(positions, tf.int32)
  features = features[:params["num_image_regions"]]
  num_objects = tensor_utils.shape(features, 0)
  padding = tf.maximum(0, params["num_image_regions"] - num_objects)
  features = tf.pad(features, [[0, padding], [0, 0]])
  positions = tf.pad(positions, [[0, padding]])
  features = tf.ensure_shape(
      features, [params["num_image_regions"], params["image_feature_size"]])
  positions = tf.ensure_shape(positions, [params["num_image_regions"]])
  mask = tf.pad(tf.ones(num_objects, dtype=tf.int32), [[0, padding]])
  mask = tf.ensure_shape(mask, [params["num_image_regions"]])
  output = ObjectDetectionOutput(
      features=features, positions=positions, mask=mask)
  return output
