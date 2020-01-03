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
"""Exporter to save  the best checkpoint."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import tensorflow.compat.v1 as tf


class BestCheckpointExporter(tf.estimator.Exporter):
  """Exporter that saves the model's best checkpoint.

  We use this over `tf.estimator.BestExporter` since we don't want to
  rely on tensorflow's `SavedModel` exporter method.
  """

  def __init__(self, compare_fn, name='best-checkpoint',
               event_file_pattern='eval/*.tfevents.*'):
    """Construct the exporter.

    Args:
      compare_fn: Function that, given the dictionary of output
                  metrics of the previously best and current checkpoints,
                  returns whether to override the previously best checkpoint
                  with the current one.
      name: Name of the exporter
      event_file_pattern: where to look for events logs
    Raises:
      ValueError: if given incorrect arguments
    """
    self._name = name
    self._compare_fn = compare_fn
    if self._compare_fn is None:
      raise ValueError('`compare_fn` must not be None.')

    self._event_file_pattern = event_file_pattern
    self._model_dir = None
    self._best_eval_result = None

  @property
  def name(self):
    return self._name

  def export(self, estimator, export_path, checkpoint_path,
             eval_result, is_the_final_export):
    del is_the_final_export

    if self._model_dir != estimator.model_dir and self._event_file_pattern:
      tf.logging.info('Loading best metric from event files.')

      self._model_dir = estimator.model_dir
      full_event_file_pattern = os.path.join(self._model_dir,
                                             self._event_file_pattern)
      self._best_eval_result = self._get_best_eval_result(
          full_event_file_pattern)

    if self._best_eval_result is None or self._compare_fn(
        best_eval_result=self._best_eval_result,
        current_eval_result=eval_result):

      tf.logging.info('Performing best checkpoint export.')
      self._best_eval_result = eval_result

      if not tf.gfile.Exists(export_path):
        tf.gfile.MakeDirs(export_path)

      new_files = set()
      for file_path in tf.gfile.Glob(checkpoint_path + '.*'):
        basename = os.path.basename(file_path)
        new_files.add(basename)
        out_file = os.path.join(export_path, basename)
        tf.gfile.Copy(file_path, out_file)

      # Clean out any old files
      for filename in tf.gfile.ListDirectory(export_path):
        if filename not in new_files:
          tf.gfile.Remove(os.path.join(export_path, filename))

  def _get_best_eval_result(self, event_files):
    """Get the best eval result from event files.

    Args:
      event_files: Absolute pattern of event files.
    Returns:
      The best eval result.
    """
    if not event_files:
      return None

    best_eval_result = None
    for event_file in tf.gfile.Glob(os.path.join(event_files)):
      for event in tf.train.summary_iterator(event_file):
        if event.HasField('summary'):
          event_eval_result = {}
          for value in event.summary.value:
            if value.HasField('simple_value'):
              event_eval_result[value.tag] = value.simple_value
          if event_eval_result:
            if best_eval_result is None or self._compare_fn(
                best_eval_result, event_eval_result):
              best_eval_result = event_eval_result
    return best_eval_result
