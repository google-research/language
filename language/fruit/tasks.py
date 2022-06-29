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
r"""SeqIO Task Definitions."""

import functools
import itertools
import os

from language.fruit import metrics
from language.fruit import postprocessors
from language.fruit import rendering_utils
import seqio
import t5
import tensorflow as tf

Task = rendering_utils.Task
DelimiterType = rendering_utils.DelimiterType
EvidenceMarkerType = rendering_utils.EvidenceMarkerType

# Please point this root to the download folder.
TASK_ROOT = "gs://gresearch/FRUIT/dataset"

FILEPATH_TEMPLATE = os.path.join(
    TASK_ROOT, "{split}",
    "article_pairs.update.{task}.{context}.{delimiter_type}."
    "{evidence_marker_type}.tfrecords")

# Default task parameters
DEFAULT_FEATURE_DESCRIPTION = {
    "inputs": tf.io.FixedLenFeature([], dtype=tf.string),
    "targets": tf.io.FixedLenFeature([], dtype=tf.string),
    "generatable_surfaces": tf.io.FixedLenFeature([], dtype=tf.string),
}
DEFAULT_PREPROCESSORS = [
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    seqio.preprocessors.append_eos_after_trim,
]

# only include edit_rouge now; skip metrics.print_predictions,
# metrics.surface_recall and metrics.exact_match for now.

DEFAULT_METRIC_FNS = [
    metrics.edit_rouge,
]
DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(),
            add_eos=True,
            required=True),
    "targets":
        seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}


def _register_w_defaults(
    name,
    split_to_filepattern,
    task,
    delimiter_type,
):
  """Register a WikiDiff task w/ default params."""

  delimiter_range_pair = rendering_utils.get_default_delimiter_range_pair(
      task,
      delimiter_type,
  )

  metric_fns = [*DEFAULT_METRIC_FNS]

  normalize_fn = functools.partial(
      rendering_utils.normalize,
      delimiter_range_pair=delimiter_range_pair,
      task=task,
  )
  postprocess_fn = functools.partial(
      postprocessors.postprocess_wikidiff,
      vocabulary=t5.data.get_default_vocabulary(),
      normalize_fn=normalize_fn,
  )
  seqio.TaskRegistry.add(
      name,
      source=seqio.TFExampleDataSource(
          split_to_filepattern=split_to_filepattern,
          feature_description=DEFAULT_FEATURE_DESCRIPTION,
      ),
      preprocessors=DEFAULT_PREPROCESSORS,
      postprocess_fn=postprocess_fn,
      metric_fns=metric_fns,
      output_features=DEFAULT_OUTPUT_FEATURES,
  )


def create_task(task, context, delimiter_type,
                evidence_marker_type):
  """Creates a task w/ correct filenames and postprocessing."""

  # Fill in template fields that are common across train/dev/test.
  task_basename = "_".join([
      "wikidiff",
      task.name,
      context,
      delimiter_type.name,
      evidence_marker_type.name,
  ])
  filepath = functools.partial(
      FILEPATH_TEMPLATE.format,
      task=task.name,
      delimiter_type=delimiter_type.name,
      evidence_marker_type=evidence_marker_type.name,
  )
  register = functools.partial(
      _register_w_defaults,
      task=task,
      delimiter_type=delimiter_type,
  )

  # Create train task
  train_prefix = filepath(split="train", context=context)
  train_fnames = [f"{train_prefix}-0000{i}-of-00010" for i in range(9)]
  validation_fname = f"{train_prefix}-00009-of-00010"
  split_to_filepattern = {"train": train_fnames, "validation": validation_fname}
  register(task_basename, split_to_filepattern)

  # Create train (old) task
  train_old_prefix = filepath(split="train_old", context=context)
  train_old_fnames = [f"{train_old_prefix}-0000{i}-of-00010" for i in range(9)]
  validation_old_fname = f"{train_old_prefix}-00009-of-00010"
  split_to_filepattern = {
      "train": train_old_fnames,
      "validation": validation_old_fname
  }
  register(f"{task_basename}_old", split_to_filepattern)

  # Create (all evidence) evaluation tasks.
  dev_prefix = filepath(split="dev", context="all")
  dev_split_to_filepattern = {"test": f"{dev_prefix}-*"}
  register(f"{task_basename}_dev", dev_split_to_filepattern)

  test_prefix = filepath(split="test", context="all")
  test_split_to_filepattern = {"test": f"{test_prefix}-*"}
  register(f"{task_basename}_test", test_split_to_filepattern)

  # Create (oracle evidence) evaluation tasks.
  oracle_dev_prefix = filepath(split="dev", context="gold")
  oracle_dev_split_to_file_pattern = {"test": f"{oracle_dev_prefix}-*"}
  register(f"{task_basename}_oracle_dev", oracle_dev_split_to_file_pattern)

  oracle_test_prefix = filepath(split="test", context="gold")
  oracle_test_split_to_file_pattern = {"test": f"{oracle_test_prefix}-*"}
  register(f"{task_basename}_oracle_test", oracle_test_split_to_file_pattern)

  # Create gold test evaluation tasks.
  gold_test_prefix = filepath(split="gold_test", context="all")
  gold_test_split_to_file_pattern = {"test": gold_test_prefix}  # Only 1 file
  register(f"{task_basename}_gold_test", gold_test_split_to_file_pattern)

  oracle_gold_test_prefix = filepath(split="gold_test", context="gold")
  oracle_gold_test_split_to_file_pattern = {
      "test": oracle_gold_test_prefix
  }  # Only 1 file
  register(f"{task_basename}_oracle_gold_test",
           oracle_gold_test_split_to_file_pattern)


## Where tasks are actually being added ########################################

# Standard task settings
for configuration in itertools.product(
    Task,
    ["all", "gold"],
    DelimiterType,
    EvidenceMarkerType,
):
  create_task(*configuration)

# No evidence baselines
for task_ in Task:
  create_task(task_, "none", DelimiterType.text, EvidenceMarkerType.empty)
