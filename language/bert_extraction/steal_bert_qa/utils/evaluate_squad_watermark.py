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
r"""Evaluation script measuring accuracy on Watermark, with victim labels as well as watermarked labels."""

from __future__ import print_function

import json
from bert_extraction.steal_bert_qa.utils import evaluate_squad

import tensorflow.compat.v1 as tf

app = tf.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string('watermark_file', None, 'Path to the watermark questions')
flags.DEFINE_string('watermark_output_file', None,
                    'Path to the predictions on the watermark points')
FLAGS = flags.FLAGS


def evaluate_dataset_preds(dataset, predictions, ans_key):
  """Evaluate word level metrics."""
  f1 = exact_match = total = 0
  for article in dataset:
    for paragraph in article['paragraphs']:
      for qa in paragraph['qas']:
        total += 1
        if qa['id'] not in predictions:
          message = 'Unanswered question ' + qa['id'] + ' will receive score 0.'
          print(message)
          continue
        ground_truths = [x['text'] for x in qa[ans_key]]
        prediction = predictions[qa['id']]
        exact_match += evaluate_squad.metric_max_over_ground_truths(
            evaluate_squad.exact_match_score, prediction, ground_truths)
        f1 += evaluate_squad.metric_max_over_ground_truths(
            evaluate_squad.f1_score, prediction, ground_truths)

  exact_match = 100.0 * exact_match / total
  f1 = 100.0 * f1 / total

  return {'exact_match': exact_match, 'f1': f1, 'instances': total}


def main(_):

  def load_dataset_file(dataset_file):
    with gfile.Open(dataset_file) as df:
      dataset_json = json.load(df)
    data = dataset_json['data']
    return data

  def load_preds_file(prediction_file):
    with gfile.Open(prediction_file) as pf:
      preds = json.load(pf)
    return preds

  dataset = load_dataset_file(FLAGS.watermark_file)
  preds = load_preds_file(FLAGS.watermark_output_file)
  logging.info('Watermark Label Accuracy =')
  logging.info(
      json.dumps(evaluate_dataset_preds(dataset, preds, ans_key='answers')))
  logging.info('Victim Label Accuracy =')
  logging.info(
      json.dumps(
          evaluate_dataset_preds(dataset, preds, ans_key='original_answers')))


if __name__ == '__main__':
  app.run(main)
