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
"""Retrieve textual encyclopedic knowledge.

Methods for retrieving and scoring Wikipedia sentences, and then adding them to
the background segment.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
import json
from absl import flags
from language.tek_representations.utils import util
from nltk import tokenize
FLAGS = flags.FLAGS


def select_entities(begin,
                    end,
                    doc_annotations,
                    confidence_threshold=0.0):
  """Filter out (a) low-confidence, (b) non-Wikipedia (c)out-of-scope annotations."""
  filtered = []
  for annotation in doc_annotations:
    if not annotation['wikidata_id']:
      continue
    if annotation['confidence'] <= confidence_threshold:
      continue
    if not any([
        mention['begin'] >= begin and mention['end'] <= end
        for mention in annotation['mentions']
    ]):
      continue
    annotation['mentions'] = [
        mention for mention in annotation['mentions']
        if mention['begin'] >= begin and mention['end'] <= end
    ]
    filtered.append(annotation)
  filtered = sorted(
      filtered,
      key=lambda x: x['popularity'] if x['popularity'] != -1 else 100000)
  return filtered


def score_sentences(query,
                    doc_json,
                    entity,
                    sentence_scores,
                    max_sentence_len,
                    n=3):
  """Score sentences with respect to the query."""
  sentences = tokenize.sent_tokenize(doc_json['text'])
  query_ngrams = util.get_ngrams(tokenize.word_tokenize(query), n)
  for sentence in sentences:
    sentence_tokens = tokenize.word_tokenize(sentence)
    tokens = tokenize.word_tokenize(
        entity['wikipedia_name']) + [':'] + sentence_tokens[:max_sentence_len]
    sentence_ngrams = util.get_ngrams(tokens, n)
    score = len(set(sentence_ngrams).intersection(query_ngrams)) / max(
        1, len(query_ngrams))
    sentence_scores.append((entity, sentence_tokens), score)


def get_sentence_contexts_by_overlap(question,
                                     begin,
                                     end,
                                     doc_annotations,
                                     tokenizer,
                                     confidence_threshold,
                                     background_quota,
                                     entity_quota,
                                     corpus,
                                     local_cache=None):
  """Get background sentences based on ngram overlap score."""
  entities = select_entities(begin, end, doc_annotations,
                             confidence_threshold)
  local_cache = {} if local_cache is None else local_cache
  sentence_scores = util.TopKMaxList(20)
  for entity in entities:
    try:
      doc = local_cache.get(
          entity['wikidata_id'],
          json.loads(
              corpus[entity['wikidata_id'].encode('utf-8')].decode('utf-8')))
    except KeyError:
      continue
    local_cache[entity['wikidata_id']] = doc
    score_sentences(
        question,
        doc,
        entity,
        sentence_scores,
        entity_quota)
  sentence_scores = sentence_scores.get_top()
  background_sub_tokens = []
  selected_entities, selected_sents = [], [[]]
  previous_entity = None
  for i, ((entity, tokens), _) in enumerate(sentence_scores):
    entity_name = entity['wikipedia_name']
    sub_tokens = []
    if previous_entity != entity_name:
      sub_tokens = [tokenizer.eos] if i > 0 else []
      sub_tokens += tokenizer.tokenize(entity_name)[0] + [
          tokenizer.entity_separator
      ]
    selected_entities.append(entity_name)
    if len(tokens) >= entity_quota:
      tokens = tokens[:entity_quota] + ['.']
    sub_tokens += (tokenizer.tokenize(tokens))[0]
    previous_entity = entity_name
    for sub_token in sub_tokens:
      if len(background_sub_tokens) == background_quota:
        selected_sents += [[]]
        break
      background_sub_tokens += [sub_token]
      selected_sents[-1] += [sub_token]
    if len(background_sub_tokens) == background_quota:
      break
  return background_sub_tokens


def get_following_contexts(all_doc_tokens, tokenizer_fn, index,
                           background_quota):
  background_sub_tokens = []
  for i in range(background_quota):
    split_token_index = index + i
    sub_tokens = tokenizer_fn(all_doc_tokens[split_token_index])
    if len(sub_tokens) + len(background_sub_tokens) > background_quota:
      break
    background_sub_tokens += sub_tokens
  return background_sub_tokens


def byte_to_char_offset(text):
  byte_to_char = {}
  byte_offset = 0
  for i, c in enumerate(text):
    byte_to_char[byte_offset] = i
    byte_offset += len(c.encode('utf-8'))
  return byte_to_char
