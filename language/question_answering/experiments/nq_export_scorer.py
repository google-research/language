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
"""Evaluates an exported model on a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gzip import GzipFile
import json

from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("input_data_pattern", None, "Input data pattern. "
                    "Assumed to be recordio.")

flags.DEFINE_string("context_export_dir", None,
                    "SavedModel path for context model.")

flags.DEFINE_string("entity_export_dir", None,
                    "SavedModel path for entity model.")

flags.DEFINE_string("output_path", None,
                    "Json path to export the predicted examples.")


_NULL_INDEX = -1


def _get_example(line):
  """Extract relevant fields from json.

  Args:
    line: String for json line.

  Returns:
    example_id: integer.
    question: string.
    contexts: List of strings.
    context_indices: List of (int32, int32) tuples.
  """
  json_example = json.loads(line)
  example_id = json_example["example_id"]
  long_answer_candidates = json_example["long_answer_candidates"]

  contexts = []
  context_indices = []
  for candidate in long_answer_candidates:
    if candidate["top_level"]:
      # Get candidate start and end indices.
      start_index = candidate["start_token"]
      end_index = candidate["end_token"]
      context_indices.append((start_index, end_index))

      # Get candidate contexts.
      candidate_tokens = json_example["document_tokens"][start_index:end_index]
      candidate_tokens = [t["token"] for t in candidate_tokens]
      candidate_str = u" ".join(candidate_tokens)
      candidate_str = candidate_str.lower()
      contexts.append(candidate_str)

  # Get question.
  question = u" ".join(json_example["question_tokens"])
  question = question.lower()

  return example_id, question, contexts, context_indices


def _annotate_long_answer(predict_fn, question, contexts):
  """Applies the model to the (question, contexts) and returns long answer.

  Args:
    predict_fn: Predictor from tf.contrib.predictor.from_saved_model.
    question: string.
    contexts: List of strings.

  Returns:
    long_answer_idx: integer.
    long_answer_score: float.
  """
  # The inputs are not tokenized here because there are multiple contexts.
  inputs = {"question": question, "context": contexts}

  outputs = predict_fn(inputs)
  long_answer_idx = outputs["idx"]
  long_answer_score = outputs["score"]

  return long_answer_idx, float(long_answer_score)


def _annotate_short_answer(predict_fn, question_tokens, context_tokens):
  """Applies the model to the (question, contexts) and returns long answer.

  Args:
    predict_fn: Predictor from tf.contrib.predictor.from_saved_model.
    question_tokens: List of strings.
    context_tokens: List of strings.

  Returns:
    long_answer_idx: integer.
    long_answer_score: float.
  """
  # The inputs are tokenized unlike in the long answer case, since the goal
  # is to pick out a particular span in a single context.
  inputs = {"question": question_tokens, "context": context_tokens}
  outputs = predict_fn(inputs)
  start_idx = outputs["start_idx"]
  end_idx = outputs["end_idx"]
  short_answer_score = outputs["score"]

  return start_idx, end_idx, float(short_answer_score)


def _annotate_dataset(context_export_dir, entity_export_dir, data_paths):
  """Load examples to annotate.

  Args:
    context_export_dir: String for context model export directory.
    entity_export_dir: String for entity model export directory.
    data_paths: List of data paths.

  Returns:
    List of dictionaries for annotated examples.
  """
  context_predict_fn = tf.contrib.predictor.from_saved_model(context_export_dir)
  entity_predict_fn = tf.contrib.predictor.from_saved_model(entity_export_dir)
  num_processed_examples = 0
  annotated_examples = []
  for data_path in data_paths:
    with GzipFile(fileobj=tf.gfile.GFile(data_path)) as data_file:
      for line in data_file:
        if num_processed_examples % 10 == 0:
          print("Processed %d examples." % num_processed_examples)
        example_id, question, contexts, context_indices = _get_example(line)
        long_answer_idx, long_answer_score = (
            _annotate_long_answer(context_predict_fn, question, contexts))

        if long_answer_idx != _NULL_INDEX:
          long_answer_start_token, long_answer_end_token = (
              context_indices[long_answer_idx])

          # Unlike in long answer, we tokenize before calling the short answer
          # model.
          question_tokens = question.split(" ")
          pred_context_tokens = contexts[long_answer_idx].split(" ")

          (short_answer_start_token,
           short_answer_end_token, short_answer_score) = _annotate_short_answer(
               entity_predict_fn, question_tokens, pred_context_tokens)

          # Offset the start/end indices by the start of the long answer.
          short_answer_start_token += long_answer_start_token
          short_answer_end_token += long_answer_start_token
          short_answer_dict = [{
              "start_token": short_answer_start_token,
              "end_token": short_answer_end_token,
              "start_byte": -1,
              "end_byte": -1
          }]
        else:
          # If long answer is NULL then short answer must also be NULL.
          long_answer_start_token = _NULL_INDEX
          long_answer_end_token = _NULL_INDEX
          short_answer_dict = []
          short_answer_score = -1.0

        annotated_example = {
            "example_id": example_id,
            "long_answer_score": long_answer_score,
            "short_answers_score": short_answer_score,
            "long_answer": {
                "start_token": long_answer_start_token,
                "end_token": long_answer_end_token,
                "start_byte": -1,
                "end_byte": -1
            },
            # While in theory there may be multiple spans indicating the short
            # answer, our model only outputs one span.
            "short_answers": short_answer_dict,
        }
        annotated_examples.append(annotated_example)
        num_processed_examples += 1

  return annotated_examples


def main(_):
  data_paths = tf.gfile.Glob(FLAGS.input_data_pattern)
  annotated_examples = _annotate_dataset(FLAGS.context_export_dir,
                                         FLAGS.entity_export_dir, data_paths)
  print(len(annotated_examples))
  json_dict = {"predictions": annotated_examples}

  print("Done evaluating all examples")
  with tf.gfile.Open(FLAGS.output_path, "w") as out_file:
    out_file.write(json.dumps(json_dict, sort_keys=False))


if __name__ == "__main__":
  tf.app.run()
