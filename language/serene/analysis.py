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
# List as: python3
"""Analyze and interpret model results.

This script does the following:
1. Reads TFDS examples.
2. Reads a model's predictions (formatted like below).
3. Merges these two so that it is easy to find the evidence the model scored
  by claim_id.
4. Compute counts by different factors and output the rank of the first correct
  prediction.
5. Write 3 and 4 to disk.

Model predictions are formatted like:
{'claim_prob': [0.7085524797439575, 0.15355344116687775, 0.13789407908916473],
 'evidence_prob': [0.0008763210498727858],
 'metadata': {'claim_id': 107511,
  'claim_label': 'REFUTES',
  'evidence_label': 'NOT_MATCHING',
  'doc_score': -1,
  'sentence_score': -20,
  'scrape_type': 'ukp_pred',
  'gold': False,
  'retrieved': True,
  'background': False,
  'tfidf_candidate': True,
  'wikipedia_url': 'Deighton',
  'sentence_id': 20}}
"""
import json
import pathlib
import pickle
import sys


from absl import app
from absl import flags
from absl import logging

import dataclasses

from language.serene import config
from language.serene import constants
from language.serene import fever_tfds
from language.serene import types
from language.serene import util

import pandas as pd
import tensorflow.compat.v2 as tf
import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string('model_root', None, 'Directory of model to analyze.')
flags.DEFINE_string('report_dir', None, 'Directory to write report files.')
flags.DEFINE_integer('n_similar_negatives', 0,
                     'TFDS number of similar negatives.')
flags.DEFINE_integer('n_background_negatives', 10,
                     'TFDS number of background negatives.')
flags.DEFINE_string('train_scrape_type', 'lucene',
                    'Scrape used during training.')


@dataclasses.dataclass(frozen=True, eq=True)
class ClaimPrediction:
  refute_probability: Optional[float]
  support_probability: Optional[float]
  not_enough_info_probability: Optional[float]


def get_match_score(model_predictions,
                    example_id):
  if example_id in model_predictions:
    return model_predictions[example_id]['evidence_prob'][0]
  else:
    return None


def get_verify_scores(model_predictions,
                      example_id):
  if example_id in model_predictions:
    scores = model_predictions[example_id]['claim_prob']
    return ClaimPrediction(scores[0], scores[1], scores[2])
  else:
    return ClaimPrediction(None, None, None)


def make_example_id(*, claim_id, wikipedia_url,
                    sentence_id, scrape_type):
  """Create a string example id for claim-evidence pairs.

  Args:
    claim_id: Fever claim id
    wikipedia_url: The wikipedia url of the evidence
    sentence_id: The sentence id of the evidence
    scrape_type: The scrape that this evidence came from

  Returns:
    A string example id
  """
  return f'{claim_id}@{wikipedia_url}@{sentence_id}@{scrape_type}'


def parse_fold(*, fold_name, model_predictions,
               tfds_examples):
  """Parse the examples in the model predictions.

  Args:
    fold_name: Name of fold that examples in rows are from
    model_predictions: Map from example_id to predictions
    tfds_examples: Examples from fever TFDS

  Returns:
    Dataframe merging the examples and predictions
  """
  output_rows = []
  for example in tqdm.tqdm(tfds_examples, mininterval=10):
    meta = json.loads(util.tf_to_str(example['metadata']))
    claim_id = meta['claim_id']
    scrape_type = util.tf_to_str(example['scrape_type'])
    wikipedia_url = util.tf_to_str(example['wikipedia_url'])
    sentence_id = util.tf_to_str(example['sentence_id'])
    ex_id = make_example_id(
        claim_id=claim_id,
        wikipedia_url=wikipedia_url,
        sentence_id=sentence_id,
        scrape_type=scrape_type,
    )
    model_score = get_match_score(model_predictions, ex_id)
    verify_scores = get_verify_scores(model_predictions, ex_id)
    # pyformat: disable
    output_rows.append({
        'evidence_label': constants.EVIDENCE_MATCHING_CLASSES[
            example['evidence_label'].numpy()],
        'claim_label': constants.FEVER_CLASSES[example['claim_label'].numpy()],
        'scrape_type': scrape_type,
        'wikipedia_url': wikipedia_url,
        'sentence_id': sentence_id,
        'retrieved': meta['retrieved'],
        'gold': meta['gold'],
        'background': meta['background'],
        'sentence_score': meta['sentence_score'],
        'doc_score': meta['doc_score'],
        'tfidf_candidate': meta['tfidf_candidate'],
        'claim_text': util.tf_to_str(example['claim_text']),
        'evidence_text': util.tf_to_str(example['evidence_text']),
        'claim_id': meta['claim_id'],
        'model_score': model_score,
        'refute_score': verify_scores.refute_probability,
        'support_score': verify_scores.support_probability,
        'nei_score': verify_scores.not_enough_info_probability,
        'fold': fold_name,
    })
    # pyformat: enable
  df = pd.DataFrame(output_rows)
  return df


def read_model_predictions(
    prediction_path):
  """Read a model's validation predictions and convert to a dictionary.

  Args:
    prediction_path: Path to read predictions from

  Returns:
    A dictionary where values are predictions, and keys are composed of
    the claim_id/wikipedia_url/sentence_id/scrape_type
  """
  model_predictions = util.read_json(prediction_path)
  id_to_predictions = {}
  for pred in model_predictions['predictions']:
    claim_id = pred['metadata']['claim_id']
    scrape_type = pred['metadata']['scrape_type']
    wikipedia_url = pred['metadata']['wikipedia_url']
    sentence_id = pred['metadata']['sentence_id']
    identifier = make_example_id(
        claim_id=claim_id,
        wikipedia_url=wikipedia_url,
        sentence_id=sentence_id,
        scrape_type=scrape_type,
    )
    id_to_predictions[identifier] = pred
  return id_to_predictions


def write_summary(report_dir, df):
  """Write summary statistics about the examples in df.

  For example, this will output how many of each label there is of each type.

  Args:
    report_dir: Directory to write summary to
    df: Dataframe of examples (claim/evidence pairs) to summarize
  """
  pd.set_option('display.max_colwidth', 300)
  with util.safe_open(report_dir / 'summary.txt', 'w') as f:
    f.write('Counts of examples by Evidence label\n')
    f.write(str(df.groupby('evidence_label').count()))
    f.write('\n')
    f.write('Count of examples by fold/scrape/evidence label/claim label\n')
    f.write(
        str(
            df.groupby(['fold', 'scrape_type', 'evidence_label',
                        'claim_label']).count()))
    f.write('\n')
    f.write('Detailed Count of examples\n')
    f.write(
        str(
            df.groupby([
                'fold', 'scrape_type', 'evidence_label', 'claim_label',
                'retrieved', 'gold', 'tfidf_candidate'
            ]).count()))


def write_per_claim_analysis(*, df,
                             claim_lookup,
                             output_path):
  """For each claim, write the examples the model scored and claim summary.

  Args:
    df: Dataframe to read predictions and examples from
    claim_lookup: Lookup from claim_id to fever claim dictionary
    output_path: Path to write analysis to
  """
  claim_predictions = {}
  grouped_df = df.groupby(['scrape_type', 'claim_id'])
  for (scrape_type, claim_id), claim_df in tqdm.tqdm(
      grouped_df, mininterval=10):
    label = claim_lookup[claim_id]['label']
    claim_df = claim_df.sort_values('model_score', ascending=False)
    claim_df = claim_df[[
        'gold', 'tfidf_candidate', 'model_score', 'support_score',
        'refute_score', 'nei_score', 'wikipedia_url', 'sentence_id',
        'evidence_text'
    ]]
    recall_rank = sys.maxsize
    for rank, row in enumerate(claim_df.itertuples(), start=1):
      if row.gold:
        recall_rank = rank
        break

    claim_predictions[(scrape_type, claim_id)] = {
        'df': claim_df,
        'claim_id': claim_id,
        'scrape_type': scrape_type,
        'label': label,
        'rank': recall_rank,
    }

  with util.safe_open(output_path, 'wb') as f:
    pickle.dump(claim_predictions, f)


def main(_):
  tf.enable_v2_behavior()
  flags.mark_flag_as_required('model_root')
  flags.mark_flag_as_required('report_dir')

  root = pathlib.Path(FLAGS.model_root)
  report_dir = pathlib.Path(FLAGS.report_dir)
  logging.info('Reading predictions from model_root: %s', root)
  logging.info('Will write analysis to: %s', report_dir)

  # Config() contains non-model specific configuration, which is why its
  # fine to use this instead of the model's configuration.
  conf = config.Config()
  dev = {c['id']: c for c in util.read_jsonlines(conf.fever_dev)}
  logging.info('Reading fever TFDS examples')
  builder = fever_tfds.FeverEvidence(
      data_dir=util.readahead(conf.fever_evidence_tfds_data),
      n_similar_negatives=FLAGS.n_similar_negatives,
      n_background_negatives=FLAGS.n_background_negatives,
      train_scrape_type=FLAGS.train_scrape_type,
      include_not_enough_info=True,
      title_in_scoring=True)
  val = builder.as_dataset(split='validation')
  val_tfds_examples = [x for x in tqdm.tqdm(val, mininterval=10)]

  logging.info('Reading model predictions')
  model_predictions = read_model_predictions(root / 'val_predictions.json')
  val_df = parse_fold(
      fold_name='val',
      model_predictions=model_predictions,
      tfds_examples=val_tfds_examples)
  df = pd.concat([val_df])

  logging.info('Writing analysis to disk')
  write_summary(report_dir, df)
  write_per_claim_analysis(
      output_path=report_dir / 'claim_evidence_predictions.pickle',
      df=df,
      claim_lookup=dev)


if __name__ == '__main__':
  app.run(main)
