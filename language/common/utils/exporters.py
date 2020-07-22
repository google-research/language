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

import os
import re

from language.common.utils import file_utils
import six
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub


class BestSavedModelAndCheckpointExporter(tf.estimator.BestExporter):
  """Exporter that saves the best SavedModel and checkpoint."""

  def __init__(self,
               eval_spec_name,
               serving_input_receiver_fn,
               compare_fn=None,
               metric_name=None,
               higher_is_better=True,
               assets_extra=None):
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
      assets_extra: An optional dict specifying how to populate the
        `assets.extra` directory within the exported SavedModel. Each key should
        give the destination path (including the filename) relative to the
        `assets.extra` directory. The corresponding value gives the full path of
        the source file to be copied. For example, the simple case of copying a
        single file without renaming it is specified as
        {'my_asset_file.txt': '/path/to/my_asset_file.txt'}.
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
        compare_fn=compare_fn or _default_compare_fn,
        assets_extra=assets_extra)

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
        data="%s\n" % exported_dir,
        path=os.path.join(export_path, "best_saved_model.txt"))

    return exported_dir


class LatestExporterWithSteps(hub.LatestModuleExporter):
  """Hub exporter that also writes a 'global_step.txt' file."""

  _step_re = re.compile(r"ckpt-(?P<steps>\d+)(?:\.(?:meta|index|data.*))?$")

  def export(self,
             estimator,
             export_path,
             checkpoint_path=None,
             eval_result=None,
             is_the_final_export=None):
    if checkpoint_path is None:
      checkpoint_path = estimator.latest_checkpoint()
    path = super(LatestExporterWithSteps, self).export(
        estimator,
        export_path,
        checkpoint_path=checkpoint_path,
        eval_result=eval_result,
        is_the_final_export=is_the_final_export)
    path = six.ensure_text(path)
    checkpoint_path = six.ensure_text(checkpoint_path)
    tf.logging.info("Path: %s, checkpoint path: %s", path, checkpoint_path)
    if path:
      match = self._step_re.search(checkpoint_path)
      if match:
        with tf.gfile.GFile(os.path.join(path, "global_step.txt"), "w") as f:
          f.write(six.ensure_binary(match.groupdict()["steps"]))
      else:
        tf.logging.warning("Couldn't find step counter in %r", checkpoint_path)


class BestModuleExporter(tf.estimator.Exporter):
  """Export the registered modules with the best metric value."""

  def __init__(self, name, serving_input_fn, compare_fn, exports_to_keep=5):
    """Creates a BestModuleExporter to use with tf.estimator.EvalSpec.

    Args:
      name: unique name of this Exporter, which will be used in the export path.
      serving_input_fn: A function with no arguments that returns a
        ServingInputReceiver. LatestModuleExporter does not care about the
        actual behavior of this function, so any return value that looks like a
        ServingInputReceiver is fine.
      compare_fn: A function that compares two evaluation results. It should
        take two arguments, best_eval_result and current_eval_result, and return
        True if the current result is better; False otherwise. See the
        loss_smaller method for an example.
      exports_to_keep: Number of exports to keep. Older exports will be garbage
        collected. Set to None to disable.
    """
    self._compare_fn = compare_fn
    self._best_eval_result = None
    self._latest_module_exporter = hub.LatestModuleExporter(
        name, serving_input_fn, exports_to_keep=exports_to_keep)

  @property
  def name(self):
    return self._latest_module_exporter.name

  def export(self,
             estimator,
             export_path,
             checkpoint_path,
             eval_result,
             is_the_final_export=None):
    """Actually performs the export of registered Modules."""
    if self._best_eval_result is None or self._compare_fn(
        best_eval_result=self._best_eval_result,
        current_eval_result=eval_result):
      tf.logging.info("Exporting the best modules.")
      self._best_eval_result = eval_result
      return self._latest_module_exporter.export(estimator, export_path)


def metric_compare_fn(key, compare):

  def _compare_fn(best_eval_result, current_eval_result):
    tf.logging.info("Old %s: %s | New %s: %s", key, key, best_eval_result[key],
                    current_eval_result[key])
    return compare(best_eval_result[key], current_eval_result[key])

  return _compare_fn
