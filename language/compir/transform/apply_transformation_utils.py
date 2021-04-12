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
"""Utils for applyig reversible and lossy transformations."""

import os


from language.compir.utils import dataset_parser_utils
from language.compir.utils import io_utils


def transform(dataset,
              split,
              transformation,
              train_data_path,
              test_data_path,
              output_path,
              prediction_path = None):
  """Applies all possible transformations by calling a parsing function."""
  if transformation not in [
      "none", "rir", "lird", "lird_rir", "lirind", "lirind_rir", "lird2",
      "lird_rir2", "lirind2", "lirind_rir2"
  ]:
    raise RuntimeError(
        "Transformation {} in not supported!".format(transformation))

  train_examples, test_examples, predictions = io_utils.load_data(
      train_data_path, test_data_path, prediction_path)
  parser = dataset_parser_utils.get_parser(dataset)(train_examples,
                                                    test_examples)

  if transformation == "none" or transformation == "lirind":
    train_examps_processed, test_examps_processed = parser.no_transformation()
  elif transformation == "rir" or transformation == "lirind_rir":
    train_examps_processed, test_examps_processed = parser.to_reversible()
  elif transformation == "lird":
    train_examps_processed, test_examps_processed = parser.program_to_lossy()
  elif transformation == "lird_rir":
    train_examps_processed, test_examps_processed = parser.rir_to_lossy()
  elif transformation == "lird2" or transformation == "lirind2":
    is_direct = True if transformation == "lird2" else False
    if not prediction_path:
      raise RuntimeError("No predictions file found!")
    train_examps_processed, test_examps_processed = parser.program_from_lossy(
        predictions, is_direct=is_direct)
  elif transformation == "lird_rir2" or transformation == "lirind_rir2":
    is_direct = True if transformation == "lird_rir2" else False
    if not prediction_path:
      raise RuntimeError("No predictions file found!")
    train_examps_processed, test_examps_processed = parser.rir_from_lossy(
        predictions, is_direct=is_direct)

  output_file = os.path.join(
      output_path, dataset + "_" + split + "_" + transformation + "_{}.tsv")
  io_utils.write_to_tsv(train_examps_processed, output_file.format("train"))
  io_utils.write_to_tsv(test_examps_processed, output_file.format("test"))
