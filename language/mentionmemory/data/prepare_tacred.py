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
"""Prepare TACRED dataset for evaluation."""

import json
import os


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
flags.DEFINE_integer('max_mentions', 32, 'max nr of mentions')

flags.DEFINE_list('split_file_names', ['train', 'dev', 'test'],
                  'List of split file names.')


def process_sample(
    sample,
    relation_vocab,
    spacy_model,
    tokenizer,
):
  """Processes Tacred sample and updates relation vocabulary.

  To process a raw Tacred example, we concatenate the token strings with spaces.
  The resulting text is parsed with a spacy model to find mention spans, and
  then tokenized with a BERT tokenizer. If necessary, we override some spacy
  mentions with the subj and obj Tacred mentions.

  Args:
    sample: raw Tacred sample. Needs to contain following fields: token, list of
      token strings. relation, string describing relation between subj and obj.
      subj_start, subj_end, obj_start, obj_end, starting and ending token
      indices for subj and obj (inclusive).
    relation_vocab: dictionary mapping relation strings to integer labels.
    spacy_model: spacy model used to detect mentions.
    tokenizer: BERT tokenizer.

  Returns:
    Processed Tacred sample and updated relation vocabulary.
  """

  processed_sample = {}

  relation = sample['relation']
  if relation not in relation_vocab:
    relation_vocab[relation] = len(relation_vocab)
  label = relation_vocab[relation]
  processed_sample['target'] = [label]

  words = sample['token']
  text = ' '.join(words)
  parsed_text = spacy_model(text)

  # We use spacy to parse text, identify noun chunks
  mention_char_spans = []
  # Compute subj and obj character spans
  word_lens = np.array([len(word) for word in words])
  word_char_offsets = np.cumsum(word_lens) - word_lens
  # Correct for added spaces from join
  word_char_offsets = word_char_offsets + np.arange(len(word_lens))

  def get_char_span(start_word_idx, end_word_idx):
    start_char = word_char_offsets[start_word_idx]
    # Char span is inclusive
    end_char = word_char_offsets[end_word_idx] + word_lens[end_word_idx] - 1
    assert text[start_char:end_char + 1] == ' '.join(
        words[start_word_idx:end_word_idx + 1])
    return (start_char, end_char)

  subj_char_span = get_char_span(sample['subj_start'], sample['subj_end'])
  mention_char_spans.append(subj_char_span)
  obj_char_span = get_char_span(sample['obj_start'], sample['obj_end'])
  mention_char_spans.append(obj_char_span)

  def overlaps(first_span, second_span):

    def point_inside_span(point, span):
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

    if not overlaps(char_span, subj_char_span) and not overlaps(
        char_span, obj_char_span):
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

  final_subj_index = reverse_span_indices[subj_index]
  final_obj_index = reverse_span_indices[obj_index]

  processed_sample['subject_mention_indices'] = final_subj_index
  processed_sample['object_mention_indices'] = final_obj_index

  mention_spans = np.array(mention_spans)
  mention_start_positions = mention_spans[:, 0]
  mention_end_positions = mention_spans[:, 1]

  mention_start_positions = mention_start_positions[:FLAGS.max_mentions]
  mention_end_positions = mention_end_positions[:FLAGS.max_mentions]

  # We should not be truncating subject or object mentions
  assert final_subj_index < FLAGS.max_mentions
  assert final_obj_index < FLAGS.max_mentions

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
    raw_samples,
    relation_vocab,
    spacy_model,
    tokenizer,
):
  """Process Tacred split and updates relation vocabulary with new relations."""

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


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tokenizer = bert_tokenization.FullTokenizer(
      FLAGS.vocab_path, do_lower_case=True)

  spacy_model = None
  if spacy_model is None:
    spacy_model = spacy.load('en_core_web_md')

  raw_data = {}
  for split_file_name in FLAGS.split_file_names:
    path = os.path.join(FLAGS.data_dir, split_file_name + '.json')
    with tf.io.gfile.GFile(path, 'rb') as data_file:
      raw_data[split_file_name] = json.load(data_file)

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
