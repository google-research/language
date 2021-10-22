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
# Lint as: python3
"""Tensorflow dataset for fever evidence matching."""

import collections
import json
import os
import random

from absl import logging
import apache_beam as beam
import dataclasses
from language.serene import constants
from language.serene import retrieval_pb2
from language.serene import scrape_db
from language.serene import text_matcher
from language.serene import types
from language.serene import util
from language.serene import wiki_db
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds



@dataclasses.dataclass(frozen=True)
class SentenceLocation:
  """Dataclass to hold the location of an evidence sentence."""
  __slots__ = ['claim_id', 'wikipedia_url', 'sentence_id']
  claim_id: int
  wikipedia_url: Text
  sentence_id: int


def underscore_to_whitespace(text):
  return text.replace('_', ' ')


def merge_validation_gold_and_candidate(examples,
                                        candidates,
                                        n_candidates):
  """Merge gold and candidate examples for the validation fold.

  Note: This mutates the members of examples, specifically adds the
    tfidf_candidate field.

  Args:
    examples: A dictionary of examples to merge with, typically containing only
      gold examples
    candidates: Candidates to possibly include
    n_candidates: The max number of candidates to return

  Returns:
    A list of examples.
  """
  for candidate in candidates:
    key = (candidate['wikipedia_url'], candidate['sentence_id'])
    # Prevent duplicates, just mark them as retrieved and attach doc score
    if key in examples:
      examples[key]['retrieved'] = True
      examples[key]['doc_score'] = candidate['doc_score']
    else:
      examples[key] = candidate

  sorted_candidates = sorted(
      examples.values(), key=lambda x: x['sentence_score'], reverse=True)
  final_candidates = sorted_candidates[:n_candidates]
  final_examples = {}
  for candidate in final_candidates:
    key = (candidate['wikipedia_url'], candidate['sentence_id'])
    candidate['tfidf_candidate'] = True
    final_examples[key] = candidate

  bottom_candidates = sorted_candidates[n_candidates:]
  for candidate in bottom_candidates:
    key = (candidate['wikipedia_url'], candidate['sentence_id'])
    # Force include gold examples, but mark them with tfidf_candidate=False
    # to indicate they should be excluded in validation evaluations.
    if candidate['gold']:
      candidate['tfidf_candidate'] = False
      final_examples[key] = candidate

  return list(final_examples.values())


class ExtractEvidenceForClaim(beam.DoFn):
  """Create evidence examples for each claim."""

  # pyformat: disable
  def __init__(
      self,
      *,
      fever_train_path,
      fever_dev_path,
      fever_test_path,
      train_scrape_type,
      n_similar_negatives,
      n_background_negatives,
      n_inference_candidates,
      n_inference_documents,
      wiki_db_path,
      text_matcher_params_path,
      title_in_scoring,
      drqa_db_path,
      lucene_db_path,
      ukp_docs_train,
      ukp_docs_dev,
      ukp_docs_test,
      max_inference_sentence_id):
    super().__init__()
    self._fever_train_path = fever_train_path
    self._fever_dev_path = fever_dev_path
    self._fever_test_path = fever_test_path
    self._train_scrape_type = train_scrape_type
    self._n_similar_negatives = n_similar_negatives
    self._n_background_negatives = n_background_negatives
    self._n_inference_candidates = n_inference_candidates
    self._n_inference_documents = n_inference_documents
    self._wiki_db_path = wiki_db_path
    self._text_matcher_params_path = text_matcher_params_path
    self._title_in_scoring = title_in_scoring
    self._drqa_db_path = drqa_db_path
    self._lucene_db_path = lucene_db_path
    self._ukp_docs_train_path = ukp_docs_train
    self._ukp_docs_dev_path = ukp_docs_dev
    self._ukp_docs_test_path = ukp_docs_test
    self._max_inference_sentence_id = max_inference_sentence_id
    self._wiki_db: Optional[wiki_db.WikiDatabase] = None
    self._wiki_titles: Optional[Set[Text]] = None
    self._matcher: Optional[text_matcher.TextMatcher] = None
    self._drqa_scrape_table: Optional[collections.ChainMap] = None
    self._lucene_scrape_table: Optional[collections.ChainMap] = None
    self._name_to_scrape: Optional[Text, collections.ChainMap] = None  # pytype: disable=invalid-annotation  # attribute-variable-annotations
    self._ukp_docs: Optional[Dict[int, types.Json]] = None
    self._claim_to_fold: Optional[Dict[int, Text]] = None
    self._n_missing_pages = beam.metrics.Metrics.counter(
        self.__class__, 'n_missing_pages')
  # pyformat: enable

  def setup(self):
    self._claim_to_fold = {}
    train = util.read_jsonlines(self._fever_train_path)
    for claim in train:
      self._claim_to_fold[claim['id']] = 'train'
    dev = util.read_jsonlines(self._fever_dev_path)
    for claim in dev:
      self._claim_to_fold[claim['id']] = 'dev'

    test = util.read_jsonlines(self._fever_test_path)
    for claim in test:
      self._claim_to_fold[claim['id']] = 'test'

    self._wiki_db = wiki_db.WikiDatabase.from_local(self._wiki_db_path)
    self._wiki_titles = set(self._wiki_db.get_wikipedia_urls())
    self._matcher = text_matcher.TextMatcher()
    self._matcher.load(self._text_matcher_params_path)

    drqa_scrape_table = scrape_db.ScrapeDatabase.from_local(self._drqa_db_path)  # pylint: disable=unused-variable
    self._drqa_scrape_table = drqa_scrape_table

    lucene_scrape_table = scrape_db.ScrapeDatabase.from_local(
        self._drqa_db_path)  # pylint: disable=unused-variable
    self._lucene_scrape_table = lucene_scrape_table
    self._name_to_scrape = {
        constants.DRQA:
            self._drqa_scrape_table,
        constants.LUCENE:
            self.
            _lucene_scrape_table  # pytype: disable=annotation-type-mismatch  # attribute-variable-annotations
    }

    ukp_claim_docs = (
        util.read_jsonlines(self._ukp_docs_train_path) +
        util.read_jsonlines(self._ukp_docs_dev_path) +
        util.read_jsonlines(self._ukp_docs_test_path))
    self._ukp_docs = {claim['id']: claim for claim in ukp_claim_docs}

  def _get_retrieved_documents(self, claim_id,
                               scrape_type):
    """Retrieve the appropriate set of documents depending on settings.

    Args:
      claim_id: The claim to get documents for
      scrape_type: The scrape type to use when fetching documents

    Returns:
      A list of documents to generate examples from
    """
    if scrape_type in (constants.DRQA, constants.LUCENE):
      claim_scrape = self._name_to_scrape[scrape_type][str(claim_id)]
      documents = [(doc.doc_id, doc.ir_score) for doc in claim_scrape.documents]
    elif scrape_type in constants.UKP_TYPES:
      if scrape_type == constants.UKP_WIKI:
        claim_scrape = self._ukp_docs[claim_id]['wiki_results']
      elif scrape_type == constants.UKP_PRED:
        claim_scrape = self._ukp_docs[claim_id]['predicted_pages']
      else:
        raise ValueError(f'Invalid scrape type: {scrape_type}')
      documents = [
          # UKP Does not have document scores.
          (doc_id.replace(' ', '_'), -1) for doc_id in claim_scrape
      ]
    else:
      raise ValueError(f'Invalid scrape type: {scrape_type}')
    return documents

  def _get_gold_examples(
      self,
      *,
      claim_json,
      scrape_type,
  ):
    """Create gold examples and seed the example dictionary with them.

    Args:
      claim_json: The json of the claim from fever dataset
      scrape_type: What type to label gold as, technically this isn't that
        scrape type, but it makes grouping by scrape type easier to do later on.

    Returns:
      A dictionary of examples keyed by (wikipedia_url, sentence_id)
    """
    examples = {}
    used_wikipedia_urls = set()
    claim_label = claim_json['label']
    # For NOT ENOUGH INFO, there are no gold examples
    if claim_label == constants.NOT_ENOUGH_INFO:
      return examples, used_wikipedia_urls
    for evidence_set in claim_json['evidence']:
      for evidence in evidence_set:
        wikipedia_url = util.normalize(evidence[2])
        used_wikipedia_urls.add(wikipedia_url)
        sentence_id = evidence[3]
        page = self._wiki_db.get_page(wikipedia_url)
        if page is None:
          raise ValueError(f'Missing page: {wikipedia_url}')
        if sentence_id in page.sentences:
          sentence = page.sentences[sentence_id].text
          key = (wikipedia_url, sentence_id)
          # FEVER sometimes has duplicated evidence if it was picked by
          # multiple raters.
          if key not in examples:
            sentence_with_title = underscore_to_whitespace(
                wikipedia_url) + ' ' + sentence
            examples[key] = {
                'evidence_text': sentence,
                'evidence_text_with_title': sentence_with_title,
                'evidence_label': constants.MATCHING,
                'claim_label': claim_label,
                'gold': True,
                'retrieved': False,
                'background': False,
                'doc_score': -1,
                'wikipedia_url': wikipedia_url,
                'sentence_id': sentence_id,
                'scrape_type': scrape_type,
            }
    return examples, used_wikipedia_urls

  def _get_similar_candidates(
      self,
      *,
      claim_label,
      documents,
      used_wikipedia_urls,
      scrape_type,
  ):
    """Return negative examples that are similar to the claim.

    Args:
      claim_label: The label of the fever claim
      documents: The documents to use
      used_wikipedia_urls: The urls used so far
      scrape_type: The scrape type to use to find candidates

    Returns:
      A list of similar evidence candidates and updated wikipedia url set
    """
    used_wikipedia_urls = set(used_wikipedia_urls)
    candidates: List[types.Json] = []
    for wikipedia_url, ir_score in documents:
      used_wikipedia_urls.add(wikipedia_url)
      parsed_page = self._wiki_db.get_page(wikipedia_url)
      if parsed_page is None:
        if scrape_type in constants.UKP_TYPES:
          self._n_missing_pages.inc()
          continue
        else:
          raise ValueError(f'Missing page: {wikipedia_url}')
      for sentence_id, sentence_struct in parsed_page.sentences.items():
        sentence_with_title = underscore_to_whitespace(
            wikipedia_url) + ' ' + sentence_struct.text
        example = {
            'evidence_text': sentence_struct.text,
            'evidence_text_with_title': sentence_with_title,
            'evidence_label': constants.NOT_MATCHING,
            'claim_label': claim_label,
            'gold': False,
            'retrieved': True,
            'background': False,
            'wikipedia_url': wikipedia_url,
            'sentence_id': sentence_id,
            'doc_score': ir_score,
            'scrape_type': scrape_type,
        }
        candidates.append(example)
    # We want to score and sort the retrieved candidates that are not also gold
    return candidates, used_wikipedia_urls

  def _get_background_candidates(
      self, *, claim_label, used_wikipedia_urls,
      scrape_type):
    """Return background negatives (ie random from wikipedia).

    During inference, we should not get these, hence the shortcut.

    Args:
      claim_label: The label of the fever claim
      used_wikipedia_urls: The wikipedia urls used so far
      scrape_type: What type to label background as, technically this isn't that
        scrape type, but it makes grouping by scrape type easier to do later on.

    Returns:
      A list of background candidates and updated wikipedia urls used.
      Does not mutate the original
    """
    used_wikipedia_urls = set(used_wikipedia_urls)
    background_candidates = []
    while True:
      if len(background_candidates) >= self._n_background_negatives:
        break
      # sample works on sets, choice does not
      wikipedia_url = random.sample(self._wiki_titles, 1)[0]
      if wikipedia_url in used_wikipedia_urls:
        continue
      used_wikipedia_urls.add(wikipedia_url)
      page = self._wiki_db.get_page(wikipedia_url)
      if page is None:
        raise ValueError(f'Missing page: {wikipedia_url}')
      sentence_candidates = list(page.sentences.keys())
      if not sentence_candidates:  # len(sentence_candidates) == 0
        continue
      sentence_id = random.choice(list(page.sentences.keys()))
      sentence = page.sentences[sentence_id].text
      sentence_with_title = underscore_to_whitespace(
          wikipedia_url) + ' ' + sentence
      background_candidates.append({
          'evidence_text': sentence,
          'evidence_text_with_title': sentence_with_title,
          'evidence_label': constants.NOT_MATCHING,
          'claim_label': claim_label,
          'gold': False,
          'retrieved': False,
          'background': True,
          'tfidf_candidate': False,
          'wikipedia_url': wikipedia_url,
          'sentence_id': sentence_id,
          'doc_score': -1,
          'scrape_type': scrape_type,
      })
    return background_candidates, used_wikipedia_urls

  def _create_train_examples(self, claim_json):
    used_wikipedia_urls = set()
    claim_id = claim_json['id']
    claim_text = claim_json['claim']
    claim_label = claim_json['label']

    # Seed examples with gold documents as positives, negs will be added
    examples, gold_used_wikipedia_urls = self._get_gold_examples(
        claim_json=claim_json,
        scrape_type=self._train_scrape_type,
    )
    used_wikipedia_urls = used_wikipedia_urls.union(gold_used_wikipedia_urls)

    # Add retrieved documents as negatives
    documents = self._get_retrieved_documents(
        claim_id,
        scrape_type=self._train_scrape_type,
    )
    candidates, used_wikipedia_urls = self._get_similar_candidates(
        claim_label=claim_label,
        documents=documents,
        used_wikipedia_urls=used_wikipedia_urls,
        scrape_type=self._train_scrape_type,
    )
    for candidate in candidates:
      key = (candidate['wikipedia_url'], candidate['sentence_id'])
      # Prevent duplicates, just mark them as retrieved and attach doc score
      if key in examples:
        examples[key]['retrieved'] = True
        examples[key]['doc_score'] = candidate['doc_score']
      else:
        examples[key] = candidate

    # Score gold and retrieved evidence on the sentence level
    examples_to_scores = list(examples.values())
    # .predict() returns candidates sorted by score
    if self._title_in_scoring:
      text_key = 'evidence_text_with_title'
    else:
      text_key = 'evidence_text'
    scored_examples = self._matcher.predict(
        claim_text, examples_to_scores, text_key=text_key)

    max_candidates = self._n_similar_negatives

    final_candidates = scored_examples[:max_candidates]
    final_examples = {}
    for score, candidate in final_candidates:
      key = (candidate['wikipedia_url'], candidate['sentence_id'])
      candidate['sentence_score'] = score
      candidate['tfidf_candidate'] = True
      final_examples[key] = candidate

    bottom_candidates = scored_examples[max_candidates:]
    for score, candidate in bottom_candidates:
      key = (candidate['wikipedia_url'], candidate['sentence_id'])
      # Force include gold examples, but notate them with false tfidf candidate
      if candidate['gold']:
        candidate['sentence_score'] = score
        candidate['tfidf_candidate'] = False
        final_examples[key] = candidate

    # During inference, we don't want background candidates, its primarily
    # useful for training.
    background_candidates, used_wikipedia_urls = self._get_background_candidates(
        claim_label=claim_label,
        used_wikipedia_urls=used_wikipedia_urls,
        scrape_type=self._train_scrape_type,
    )

    scored_background_candidates = self._matcher.score(
        claim_text, background_candidates, text_key=text_key)
    for score, candidate in zip(scored_background_candidates,
                                background_candidates):
      candidate['sentence_score'] = score
      key = (candidate['wikipedia_url'], candidate['sentence_id'])
      # Since the wikipedia page is never seen, and only one sentence is drawn
      # from each, it is impossible to accidentally duplicate evidence here.
      final_examples[key] = candidate

    return list(final_examples.values())

  def _create_validation_examples(self, *, claim_json,
                                  scrape_type,
                                  n_inference_candidates):
    """Create validation examples for fever task.

    This function follows these steps/guidelines:
    1. Get up to the top n_inference_documents rated documents
    2. Return up to the first thirty sentences in each document
    3. In total, return n_inference_candidates, obtaining the max by iteratively
      getting the 1st sentence of each doc, then second etc.
    4. For debugging, include gold examples not retrieved with these, but mark
      tfidf_candidate False so that they can be filtered out

    Args:
      claim_json: The fever claim to get examples for
      scrape_type: The scrape type to use
      n_inference_candidates: Number of candidates to return

    Returns:
      Examples for validation on fever
    """
    used_wikipedia_urls = set()
    claim_id = claim_json['id']
    claim_label = claim_json['label']

    # Seed examples with gold documents as positives, negs will be added
    examples, gold_used_wikipedia_urls = self._get_gold_examples(
        claim_json=claim_json,
        scrape_type=scrape_type,
    )
    for key in examples:
      examples[key]['sentence_score'] = -examples[key]['sentence_id']
    used_wikipedia_urls = used_wikipedia_urls.union(gold_used_wikipedia_urls)

    # Add retrieved documents as negatives
    # For inference, we generate the input documents for each type of way
    # to get them, the scoring script handles separating this out to create
    # metrics for each method so we can compare
    documents = self._get_retrieved_documents(
        claim_id,
        scrape_type=scrape_type,
    )
    documents = documents[:self._n_inference_documents]
    candidates: List[types.Json] = []
    for wikipedia_url, ir_score in documents:
      parsed_page = self._wiki_db.get_page(wikipedia_url)
      used_wikipedia_urls.add(wikipedia_url)
      if parsed_page is None:
        self._n_missing_pages.inc()
      else:
        for sentence_id in range(self._max_inference_sentence_id):
          if sentence_id in parsed_page.sentences:
            sentence_struct = parsed_page.sentences[sentence_id]
            sentence_with_title = underscore_to_whitespace(
                wikipedia_url) + ' ' + sentence_struct.text
            example = {
                'evidence_text': sentence_struct.text,
                'evidence_text_with_title': sentence_with_title,
                'evidence_label': constants.NOT_MATCHING,
                'claim_label': claim_label,
                'gold': False,
                'retrieved': True,
                'background': False,
                'wikipedia_url': wikipedia_url,
                'sentence_id': sentence_id,
                'doc_score': ir_score,
                'scrape_type': scrape_type,
                # This sorts examples with smallest sentence_id to the top
                'sentence_score': -sentence_id
            }
            candidates.append(example)

    return merge_validation_gold_and_candidate(examples, candidates,
                                               n_inference_candidates)

  def _create_test_examples(self, *, claim_json, scrape_type,
                            n_inference_candidates):
    """Create test examples for fever task.

    This function is similar to create_validation_examples, but handles the
    fact that: (1) there are no gold examples and (2) examples only have fields
    "id" and "claim"

    This function follows these steps/guidelines:
    1. Get up to the top n_inference_documents rated documents
    2. Return up to the first thirty sentences in each document
    3. In total, return n_inference_candidates, obtaining the max by iteratively
      getting the 1st sentence of each doc, then second etc.

    Args:
      claim_json: The fever claim to get examples for
      scrape_type: The scrape type to use
      n_inference_candidates: Number of candidates to return

    Returns:
      Examples for test on fever
    """
    claim_id = claim_json['id']

    # Add retrieved documents as negatives
    # For inference, we generate the input documents for each type of way
    # to get them, the scoring script handles separating this out to create
    # metrics for each method so we can compare
    documents = self._get_retrieved_documents(
        claim_id,
        scrape_type=scrape_type,
    )
    documents = documents[:self._n_inference_documents]
    candidates: List[types.Json] = []
    for wikipedia_url, ir_score in documents:
      parsed_page = self._wiki_db.get_page(wikipedia_url)
      if parsed_page is None:
        self._n_missing_pages.inc()
        continue
      for sentence_id in range(self._max_inference_sentence_id):
        if sentence_id in parsed_page.sentences:
          sentence_struct = parsed_page.sentences[sentence_id]
          sentence_with_title = underscore_to_whitespace(
              wikipedia_url) + ' ' + sentence_struct.text
          example = {
              'evidence_text': sentence_struct.text,
              'evidence_text_with_title': sentence_with_title,
              'evidence_label': constants.NOT_MATCHING,
              # This label does not mean anything since test examples are not
              # labeled, but it must exist and be valid for TFDS to work
              # correctly.
              'claim_label': constants.REFUTES,
              'gold': False,
              'retrieved': True,
              'background': False,
              'tfidf_candidate': True,
              'wikipedia_url': wikipedia_url,
              'sentence_id': sentence_id,
              'doc_score': ir_score,
              'scrape_type': scrape_type,
              # This sorts examples with smallest sentence_id to the top
              'sentence_score': -sentence_id
          }
          candidates.append(example)

    return sorted(
        candidates, reverse=True,
        key=lambda c: c['sentence_score'])[:n_inference_candidates]

  def process(self, claim_json, *args,
              **kwargs):
    """Convert a json claim to a list of claim-evidence example pairs.

    Sketch of this method:
    1. Get the gold examples for the claim
    2. Get the retrieved examples for the claim
    3. Get the background examples for the claim

    Then:
    4. Deduplicate the gold and retrieved examples, maintaining track of where
      they came from
    5. Score the (large) list of gold/retrieved examples with a sentence matcher
    6. Sort by this, and cut to the top evidence
    7. Find any gold evidence not in the top evidence, mark that it was excluded
      And re-add it back if in whole-wiki scenario

    Args:
      claim_json: The claim json from fever
      *args: API Compat
      **kwargs: API Compat

    Returns:
      A list of json formatted examples for the tensorflow dataset
    """
    claim_id = claim_json['id']
    fold = self._claim_to_fold[claim_id]
    if fold == 'train':
      fold_examples = self._create_train_examples(claim_json)
    elif fold == 'test':
      fold_examples = []
      fold_examples.extend(
          self._create_test_examples(
              claim_json=claim_json,
              scrape_type=constants.DRQA,
              n_inference_candidates=self._n_inference_candidates))
      fold_examples.extend(
          self._create_test_examples(
              claim_json=claim_json,
              scrape_type=constants.LUCENE,
              n_inference_candidates=self._n_inference_candidates))
      fold_examples.extend(
          self._create_test_examples(
              claim_json=claim_json,
              scrape_type=constants.UKP_PRED,
              n_inference_candidates=self._n_inference_candidates))
      fold_examples.extend(
          self._create_test_examples(
              claim_json=claim_json,
              scrape_type=constants.UKP_WIKI,
              n_inference_candidates=self._n_inference_candidates))
    elif fold == 'dev':
      fold_examples = []
      fold_examples.extend(
          self._create_validation_examples(
              claim_json=claim_json,
              scrape_type=constants.DRQA,
              n_inference_candidates=self._n_inference_candidates))
      fold_examples.extend(
          self._create_validation_examples(
              claim_json=claim_json,
              scrape_type=constants.LUCENE,
              n_inference_candidates=self._n_inference_candidates))
      fold_examples.extend(
          self._create_validation_examples(
              claim_json=claim_json,
              scrape_type=constants.UKP_PRED,
              n_inference_candidates=self._n_inference_candidates))
      fold_examples.extend(
          self._create_validation_examples(
              claim_json=claim_json,
              scrape_type=constants.UKP_WIKI,
              n_inference_candidates=self._n_inference_candidates))
    else:
      raise ValueError(f'Invalid fold: {fold} for\n{claim_json}')

    serialized_examples = []
    for idx, example in enumerate(fold_examples):
      scrape_type = example['scrape_type']
      metadata = {
          'claim_id': claim_id,
          'claim_label': example['claim_label'],
          'evidence_label': example['evidence_label'],
          'doc_score': example.get('doc_score', -1),
          'sentence_score': example['sentence_score'],
          'scrape_type': scrape_type,
          'gold': example['gold'],
          'retrieved': example['retrieved'],
          'background': example['background'],
          'tfidf_candidate': example['tfidf_candidate'],
          'wikipedia_url': example['wikipedia_url'],
          'sentence_id': example['sentence_id'],
      }
      serialized_examples.append(
          dict(
              example_id=f'{claim_id}-{idx}-{scrape_type}',
              claim_text=claim_json['claim'],
              evidence_text=example['evidence_text'],
              wikipedia_url=example['wikipedia_url'],
              sentence_id=str(example['sentence_id']),
              evidence_label=example['evidence_label'],
              claim_label=example['claim_label'],
              scrape_type=scrape_type,
              metadata=json.dumps(metadata),
          ))
    return serialized_examples


def dataset_path(*, data_dir, scrape_type,
                 include_not_enough_info, title_in_scoring,
                 n_similar_negatives, n_background_negatives):
  """Return the dataset path based on its configuration.

  For example, {data_dir}/type=drqa,n_similar=5,n_background=10,include_nei=true

  Args:
    data_dir: The parent directory to use
    scrape_type: The scrape type (e.g., drqa, lucene)
    include_not_enough_info: Whether to include not enough information claims
    title_in_scoring: Whether to include title in evidence for tfidf scoring
    n_similar_negatives: How many similar negatives tare used
    n_background_negatives: How many background negatives are used

  Returns:
    Path for FeverEvidence TFDS to write to
  """
  parts = [
      f'train_type={scrape_type}', f'n_similar={n_similar_negatives}',
      f'n_background={n_background_negatives}',
      f'include_nei={include_not_enough_info}',
      f'score_title={title_in_scoring}'
  ]
  parts = sorted(parts)
  directory = ','.join(parts)
  return os.path.join(data_dir, directory)


class FeverEvidence(tfds.core.BeamBasedBuilder):
  """TFDS for Fever Evidence Matching."""
  VERSION = tfds.core.Version('0.1.0')

  def __init__(
      self,
      *,
      # Next params optional if loading from data_dir, required for generation.
      title_in_scoring,
      n_similar_negatives,
      n_background_negatives,
      include_not_enough_info,
      train_scrape_type,
      n_inference_documents = None,
      n_inference_candidates = None,
      max_inference_sentence_id = None,
      wiki_db_path = None,
      text_matcher_params_path = None,
      fever_train_path = None,
      fever_dev_path = None,
      fever_test_path = None,
    self._ukp_docs_train_path = ukp_docs_train
    self._ukp_docs_dev_path = ukp_docs_dev
    self._ukp_docs_test_path = ukp_docs_test

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            'example_id':
                tf.string,
            'metadata':
                tf.string,
            'claim_text':
                tfds.features.Text(),
            'evidence_text':
                tfds.features.Text(),
            'wikipedia_url':
                tfds.features.Text(),
            'sentence_id':
                tfds.features.Text(),
            'scrape_type':
                tfds.features.Text(),
            'evidence_label':
                tfds.features.ClassLabel(
                    names=constants.EVIDENCE_MATCHING_CLASSES),
            'claim_label':
                tfds.features.ClassLabel(names=constants.FEVER_CLASSES)
        }),
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                'claim_filepath': self._fever_train,
            },
            num_shards=100,
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                'claim_filepath': self._fever_dev,
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                'claim_filepath': self._fever_test,
            })
    ]

  def _build_pcollection(self, pipeline, claim_filepath):
    """Build a beam pipeline to generate training examples.

    Args:
      pipeline: The pipeline configured with a runner
      claim_filepath: The path to read claims from

    Returns:
      Beam pipeline to compute examples
    """
    claims = util.read_jsonlines(claim_filepath)
    if not self._include_not_enough_info:
      claims = [c for c in claims if c['label'] != constants.NOT_ENOUGH_INFO]
    logging.info('Reading claims from: %s', claim_filepath)
    logging.info('n_similar_negatives=%s', self._n_similar_negatives)
    logging.info('n_background_negatives=%s', self._n_background_negatives)
    return (pipeline
            | 'LoadClaims' >> beam.Create(claims)
            | 'ReshuffleClaims' >> beam.Reshuffle()
            | 'ExtractEvidenceForEachClaim' >> beam.ParDo(
                ExtractEvidenceForClaim(
                    n_similar_negatives=self._n_similar_negatives,
                    n_background_negatives=self._n_background_negatives,
                    n_inference_candidates=self._n_inference_candidates,
                    n_inference_documents=self._n_inference_documents,
                    max_inference_sentence_id=self._max_inference_sentence_id,
                    wiki_db_path=self._wiki_db_path,
                    text_matcher_params_path=self._text_matcher_params_path,
                    title_in_scoring=self._title_in_scoring,
                    train_scrape_type=self._train_scrape_type,
                    ukp_docs_train=self._ukp_docs_train_path,
                    ukp_docs_dev=self._ukp_docs_dev_path,
                    ukp_docs_test=self._ukp_docs_test_path,
                    fever_train_path=self._fever_train,
                    fever_dev_path=self._fever_dev,
                    fever_test_path=self._fever_test,
                    drqa_db_path=self._drqa_db_path,
                    lucene_db_path=self._lucene_db_path,
                ))
            | 'ExampleWithId' >> beam.Map(lambda x: (x['example_id'], x))
            | 'ShuffleExamples' >> beam.Reshuffle())
