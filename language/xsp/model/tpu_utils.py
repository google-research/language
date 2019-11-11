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
"""Utilities for supporting training on TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import tensorflow as tf

from tensorflow import summary


def _tpu_summary_no_op(*args, **kwargs):
  del args, kwargs  # Unused
  tf.logging.warning("tf.summary.histogram is suppressed on TPU (b/64980426).")


@contextlib.contextmanager
def rewire_summary_calls(use_tpu):
  """Rewires histogram() to nop and scalar() to tpu_summary_scalar.

  Summaries are not yet supported on TPUs (b/64980426). So we provide this
  context manager that can swap out tf.summary.histogram
  to be no-op and tf.summary.scalar to be tpu_summary_scalar. When the context
  exits, the rewiring is reset.
  See tpu_summary_scalar() for option to use.

  Args:
    use_tpu: Whether the model is being executed on TPUs.

  Yields:
    None.
  """
  # Store the original functions so we can put them back when the context ends.
  if use_tpu:
    original_tf_summary_scalar = tf.summary.scalar
    original_tf_summary_histogram = tf.summary.histogram
    original_summary_scalar = summary.scalar
    original_summary_histogram = summary.histogram

    tf.summary.scalar = summary.scalar = _tpu_summary_no_op
    tf.summary.histogram = summary.histogram = _tpu_summary_no_op
    try:
      yield
    finally:
      tf.summary.scalar = original_tf_summary_scalar
      tf.summary.histogram = original_tf_summary_histogram
      summary.scalar = original_summary_scalar
      summary.histogram = original_summary_histogram
  else:
    yield
