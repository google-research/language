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
"""Utils for loading data from files and writing data to disk."""



from language.compir.dataset_parsers.dataset_parser import Example

from tensorflow.io import gfile


def _load_examples(filename):
  """Loads raw examples from original data file."""
  examples = []
  with gfile.GFile(filename, "r") as input_file:
    for line in input_file:
      splits = line.split("OUT:")
      input_string = splits[0][3:].strip()
      output_string = splits[1].strip()

      examples.append(Example(input_string, output_string))
  return examples


def _load_predictions(filename):
  """Loads predictions from file."""
  predictions = []
  with gfile.GFile(filename, "r") as input_file:
    for line in input_file:
      predictions.append(line.strip())
  return predictions


def load_data(train_data_path, test_data_path,
              prediction_path):
  """Loads train and test data, and possibly loads predictions."""
  train_examples = _load_examples(train_data_path)
  test_examples = _load_examples(test_data_path)
  if prediction_path:
    predictions = _load_predictions(prediction_path)
    num_predictions = len(predictions)
    num_test_examples = len(test_examples)
    if num_predictions != num_test_examples:
      raise RuntimeError(
          "#predictions ({}) must equal #test_examples ({})!".format(
              num_predictions, num_test_examples))
  else:
    predictions = None
  return train_examples, test_examples, predictions


def write_to_tsv(examples, output_filename):
  """Write examples to tsv file."""
  with gfile.GFile(output_filename, "w") as tsv_file:
    for example in examples:
      line = "%s\t%s\n" % (example.utterance, example.program)
      tsv_file.write(line)
  print("Wrote %s examples to %s." % (len(examples), output_filename))
