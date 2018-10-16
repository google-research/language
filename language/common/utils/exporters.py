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
"""Exporters for tf.estimator training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from language.common.utils import file_utils
import tensorflow as tf


class BestSavedModelAndCheckpointExporter(tf.estimator.BestExporter):
  """Exporter that saves the best SavedModel and checkpoint."""

  def __init__(self,
               eval_spec_name,
               serving_input_receiver_fn,
               compare_fn=None,
               metric_name=None,
               higher_is_better=True):
    """Creates an exporter that compares models on the given eval and metric.

    While the SavedModel is useful for inference, the checkpoint is useful for
    warm-starting another stage of training (e.g., fine-tuning).

    Args:
      eval_spec_name: Name of the EvalSpec to use to compare models.
      serving_input_receiver_fn: Callable that returns a ServingInputReceiver.
      compare_fn: Callable that compares eval metrics of two models.  See
        tf.estimator.BestExporter for details on the expected API.  Specify
        either this or `metric_name`.
      metric_name: Name of the eval metric to use to compare models.  Specify
        either this or `compare_fn`.
      higher_is_better: Whether higher or lower eval metric values are better.
        Only used when `metric_name` is specified.
    """
    self._metric_name = metric_name

    def _default_compare_fn(best_eval_result, current_eval_result):
      """Returns True if the current metric is better than the best metric."""
      if higher_is_better:
        return current_eval_result[metric_name] > best_eval_result[metric_name]
      else:
        return current_eval_result[metric_name] < best_eval_result[metric_name]

    super(BestSavedModelAndCheckpointExporter, self).__init__(
        name="best_%s" % eval_spec_name,
        serving_input_receiver_fn=serving_input_receiver_fn,
        event_file_pattern="eval_%s/*.tfevents.*" % eval_spec_name,
        compare_fn=compare_fn or _default_compare_fn)

  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    """Implements Exporter.export()."""
    # Since export() returns None if export was skipped, we can use this to
    # detect when the current model is the new best model.
    exported_dir = super(BestSavedModelAndCheckpointExporter, self).export(
        estimator=estimator,
        export_path=export_path,
        checkpoint_path=checkpoint_path,
        eval_result=eval_result,
        is_the_final_export=is_the_final_export)
    if exported_dir is None:
      return None  # best model unchanged

    checkpoint_dir = os.path.join(export_path, "checkpoint")
    tf.logging.info("Saving new best checkpoint to %s", checkpoint_dir)
    file_utils.make_empty_dir(checkpoint_dir)
    file_utils.copy_files_to_dir(
        source_filepattern=checkpoint_path + ".*", dest_dir=checkpoint_dir)

    # Also save the new best metrics.
    all_metrics = "".join(
        "  %r: %r,\n" % (name, metric)
        for name, metric in sorted(self._best_eval_result.items()))
    file_utils.set_file_contents(
        data="{\n" + all_metrics + "}\n",
        path=os.path.join(export_path, "all_metrics.txt"))
    file_utils.set_file_contents(
        data="%d %r\n" % (self._best_eval_result["global_step"],
                          self._best_eval_result[self._metric_name]),
        path=os.path.join(export_path, "best_metric.txt"))
    file_utils.set_file_contents(
        data=exported_dir + "\n",
        path=os.path.join(export_path, "best_saved_model.txt"))

    return exported_dir
