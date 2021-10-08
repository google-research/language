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
"""Prepare WebRED dataset for evaluation."""

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import language.bert.tokenization as bert_tokenization
from language.mentionmemory.utils import tokenization_utils
import numpy as np
import spacy
import tensorflow.compat.v2 as tf
import tqdm


FLAGS = flags.FLAGS

flags.DEFINE_string('save_dir', None, 'save directory')
flags.DEFINE_string('data_dir', None, 'Directory where raw data is located.')
flags.DEFINE_string('vocab_path', None, 'Path to tokenizer vocabulary')
flags.DEFINE_integer('max_length', 128, 'max nr of tokens')
flags.DEFINE_integer('max_mentions', 48, 'max nr of mentions')

NO_RELATION = 'no_relation'


def process_sample(
    sample: Dict[str, Any],
    relation_vocab: Dict[str, int],
    spacy_model: Any,
    tokenizer: Any,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
  """Processes WebRED sample and updates relation vocabulary.

  To process a raw WebRED example, we first extract subj and obj and remove the
  annotations from the text. The resulting text is parsed with a spacy model to
  find mention spans, and then tokenized with a BERT tokenizer. If necessary, we
  override some spacy mentions with the subj and obj WebRED mentions.

  Args:
    sample: raw WebRED sample. Needs to contain following fields: token, list of
      token strings. relation, string describing relation between subj and obj.
    relation_vocab: dictionary mapping relation strings to integer labels.
    spacy_model: spacy model used to detect mentions.
    tokenizer: BERT tokenizer.

  Returns:
    Processed WebRED sample and updated relation vocabulary.
  """

  processed_sample = {}

  if sample['num_pos_raters'] < 2:
    relation = NO_RELATION
  else:
    relation = sample['relation']
  if relation not in relation_vocab:
    relation_vocab[relation] = len(relation_vocab)
  label = relation_vocab[relation]
  processed_sample['target'] = [label]

  text = sample['annotated_text']

  # Remove subj and obj annotations from text and store position
  def find_span(input_text: str, pattern: Any,
                prefix_len: int) -> Tuple[int, int]:
    """Find span corresponding to actual subj or obj strings."""
    match = pattern.search(input_text)
    span_start = match.start() + prefix_len + 1
    # We want inclusive spans, hence -2 instead of -1
    span_end = match.end() - 2
    return (span_start, span_end)

  def replace_and_adjust(
      input_text: str, match: Any, prefix_len: int,
      inverted_mapping: np.ndarray) -> Tuple[str, np.ndarray]:
    """Remove subj/obj annotations and adjust token mapping accordingly."""

    original_span_start = match.start() + prefix_len + 1
    original_span_end = match.end() - 1
    actual_string = input_text[original_span_start:original_span_end]
    new_text = input_text[:match.start()] + actual_string + input_text[match
                                                                       .end():]

    # Inverted mapping maps from remaining tokens to positions in original text
    new_inverted_mapping = np.zeros(len(new_text), dtype=np.int32)
    new_inverted_mapping[:match.start()] = inverted_mapping[:match.start()]

    new_span_start = match.start()
    new_span_end = match.start() + len(actual_string)
    new_inverted_mapping[new_span_start:new_span_end] = inverted_mapping[
        original_span_start:original_span_end]
    new_inverted_mapping[new_span_end:] = inverted_mapping[original_span_end +
                                                           1:]

    return new_text, new_inverted_mapping

  inverted_mapping = np.arange(len(text))
  subj_pattern = re.compile('SUBJ{[^}]+}')
  subj_span = find_span(text, subj_pattern, len('SUBJ'))
  obj_pattern = re.compile('OBJ{[^}]+}')
  obj_span = find_span(text, obj_pattern, len('OBJ'))

  # Remove subj/obj annotations from text
  while True:
    subj_match = subj_pattern.search(text)
    if subj_match is None:
      break
    text, inverted_mapping = replace_and_adjust(text, subj_match, len('SUBJ'),
                                                inverted_mapping)

  while True:
    obj_match = obj_pattern.search(text)
    if obj_match is None:
      break
    text, inverted_mapping = replace_and_adjust(text, obj_match, len('OBJ'),
                                                inverted_mapping)

  # Adjust spans for removed tokens
  mapping = np.zeros(len(sample['annotated_text']), dtype=np.int32) - 1
  mapping[inverted_mapping] = np.arange(len(inverted_mapping))
  subj_span = (mapping[subj_span[0]], mapping[subj_span[1]])
  assert subj_span[0] != -1 and subj_span[1] != -1
  obj_span = (mapping[obj_span[0]], mapping[obj_span[1]])
  assert obj_span[0] != -1 and obj_span[1] != -1

  parsed_text = spacy_model(text)

  # We use spacy to parse text, identify noun chunks
  mention_char_spans = []
  mention_char_spans.append(subj_span)
  mention_char_spans.append(obj_span)

  def overlaps(first_span: Tuple[int, int], second_span: Tuple[int,
                                                               int]) -> bool:

    def point_inside_span(point: int, span: Tuple[int, int]) -> bool:
      return span[0] >= point and point <= span[1]

    spans_overlap = (
        point_inside_span(first_span[0], second_span) or
        point_inside_span(first_span[1], second_span) or
        point_inside_span(second_span[0], first_span) or
        point_inside_span(second_span[1], first_span))

    return spans_overlap

  for chunk in parsed_text.noun_chunks:
    span_start_char = parsed_text[chunk.start].idx
    span_last_token = parsed_text[chunk.end - 1]
    span_end_char = span_last_token.idx + len(span_last_token.text) - 1
    char_span = (span_start_char, span_end_char)
    # Append only if does not overlap with subj or obj spans. In case spacy
    # mention annotation disagrees with tacred annotation, we want to favor
    # tacred.

    if not overlaps(char_span, subj_span) and not overlaps(char_span, obj_span):
      mention_char_spans.append(char_span)

  # Sort spans by start char
  start_chars = np.array([span[0] for span in mention_char_spans])
  sorted_indices = np.argsort(start_chars)
  sorted_positions = np.zeros_like(start_chars)
  sorted_positions[sorted_indices] = np.arange(len(sorted_positions))
  sorted_spans = [mention_char_spans[idx] for idx in sorted_indices]

  # Tokenize and get aligned mention positions
  _, text_ids, text_mask, mention_spans, span_indices = tokenization_utils.tokenize_with_mention_spans(
      tokenizer=tokenizer,
      sentence=text,
      spans=sorted_spans,
      max_length=FLAGS.max_length,
      add_bert_tokens=True,
      allow_truncated_spans=True,
  )

  processed_sample['text_ids'] = text_ids
  processed_sample['text_mask'] = text_mask

  # Subj and obj are the first elements of mention spans.
  subj_index = sorted_positions[0]
  obj_index = sorted_positions[1]

  # Some spans may be dropped by the BERT tokenizer. Here we map indices in the
  # original list of spans to the one returned by the tokenizer.
  reverse_span_indices = {
      original_idx: tokenized_idx
      for tokenized_idx, original_idx in enumerate(span_indices)
  }

  # Skip if subj or obj dropped.
  if (subj_index not in reverse_span_indices or
      obj_index not in reverse_span_indices):
    return None, relation_vocab

  subj_index = reverse_span_indices[subj_index]
  obj_index = reverse_span_indices[obj_index]

  # Make sure we don't discard subj or obj
  assert max(subj_index, obj_index) < FLAGS.max_mentions

  processed_sample['subject_mention_indices'] = [subj_index]
  processed_sample['object_mention_indices'] = [obj_index]

  mention_spans = np.array(mention_spans)
  mention_start_positions = mention_spans[:, 0]
  mention_end_positions = mention_spans[:, 1]

  mention_start_positions = mention_start_positions[:FLAGS.max_mentions]
  mention_end_positions = mention_end_positions[:FLAGS.max_mentions]

  mention_pad_shape = (0, FLAGS.max_mentions - len(mention_start_positions))

  mention_mask = np.ones(len(mention_start_positions), dtype=np.int64)
  mention_mask = np.pad(mention_mask, mention_pad_shape, mode='constant')
  mention_start_positions = np.pad(
      mention_start_positions, mention_pad_shape, mode='constant')
  mention_end_positions = np.pad(
      mention_end_positions, mention_pad_shape, mode='constant')

  processed_sample['mention_start_positions'] = mention_start_positions
  processed_sample['mention_end_positions'] = mention_end_positions
  processed_sample['mention_mask'] = mention_mask

  return processed_sample, relation_vocab


def process_data(
    raw_samples: List[Dict[str, Any]],
    relation_vocab: Dict[str, int],
    spacy_model: Any,
    tokenizer: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
  """Process WebRED split and updates relation vocabulary with new relations."""

  skipped = 0
  processed_samples = []
  for sample in tqdm.tqdm(raw_samples):
    processed_sample, relation_vocab = process_sample(
        sample=sample,
        relation_vocab=relation_vocab,
        spacy_model=spacy_model,
        tokenizer=tokenizer,
    )
    if processed_sample is not None:
      processed_samples.append(processed_sample)
    else:
      skipped += 1

  logging.info('Skipped %s samples', skipped)

  return processed_samples, relation_vocab


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tokenizer = bert_tokenization.FullTokenizer(
      FLAGS.vocab_path, do_lower_case=True)

  spacy_model = None
  if spacy_model is None:
    spacy_model = spacy.load('en_core_web_md')

  def get_feature(sample, feature_name, idx=0):
    feature = sample.features.feature[feature_name]
    return getattr(feature, feature.WhichOneof('kind')).value[idx]

  path = os.path.join(FLAGS.data_dir, 'webred_21.tfrecord')
  samples = []
  dataset = tf.data.TFRecordDataset(path)
  for raw_sample in dataset:
    sample = {}
    example = tf.train.Example()
    example.ParseFromString(raw_sample.numpy())
    sample['annotated_text'] = get_feature(example, 'sentence').decode('utf-8')
    sample['relation'] = get_feature(example, 'relation_name').decode('utf-8')
    sample['num_pos_raters'] = get_feature(example, 'num_pos_raters')
    samples.append(sample)

  np.random.seed(0)
  shuffled_indices = np.random.permutation(len(samples))
  shuffled_samples = [samples[idx] for idx in shuffled_indices]
  raw_data = {}
  eval_split_size = int(0.1 * len(samples))
  raw_data['test'] = shuffled_samples[:eval_split_size]
  raw_data['dev'] = shuffled_samples[eval_split_size:2 * eval_split_size]
  raw_data['train'] = shuffled_samples[2 * eval_split_size:]

  processed_data = {}
  relation_vocab = {}

  for split_name, split_data in raw_data.items():
    logging.info('Processing %s split', split_name)
    processed_split_data, relation_vocab = process_data(split_data,
                                                        relation_vocab,
                                                        spacy_model, tokenizer)
    processed_data[split_name] = processed_split_data

  # Create TFRecords
  tf.io.gfile.makedirs(FLAGS.save_dir)
  for split_name, split_data in processed_data.items():
    file_path = os.path.join(FLAGS.save_dir, split_name)
    logging.info('Writing %s split to %s', split_name, file_path)
    writer = tf.io.TFRecordWriter(file_path)
    for sample in split_data:
      features = tf.train.Features(
          feature={
              key: tf.train.Feature(int64_list=tf.train.Int64List(value=value))
              for key, value in sample.items()
          })

      record_bytes = tf.train.Example(features=features).SerializeToString()
      writer.write(record_bytes)

  # save label vocab
  vocab_path = os.path.join(FLAGS.save_dir, 'relation_vocab.json')
  with tf.io.gfile.GFile(vocab_path, 'w+') as vocab_file:
    json.dump(relation_vocab, vocab_file)


if __name__ == '__main__':
  app.run(main)
