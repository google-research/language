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
"""Prepare complex web questions dataset."""

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import language.bert.tokenization as bert_tokenization
from language.mentionmemory.utils import default_values
from language.mentionmemory.utils import tokenization_utils
import numpy as np
import spacy
import tensorflow.compat.v2 as tf
import tqdm


FLAGS = flags.FLAGS

flags.DEFINE_string('save_dir', None, 'save directory')
flags.DEFINE_string('data_dir', None, 'Path of raw data directory.')
flags.DEFINE_string('vocab_path', None, 'Path to tokenizer vocabulary')
flags.DEFINE_string('dev_file', 'complexwebq-dev.tfr', 'name of dev file')
flags.DEFINE_string('train_file', 'complexwebq-train.tfr', 'name of train file')
flags.DEFINE_integer('max_length', 128, 'max nr of tokens')
flags.DEFINE_integer('max_mentions', 48, 'max nr of mentions')


def process_sample(
    sample: Dict[str, Any],
    spacy_model: Any,
    tokenizer: Any,
) -> Optional[Dict[str, Any]]:
  """Processes CWQ sample.

  To process a raw CWQ example, we first strip special tokens and convert tokens
  back to text. Then a spacy model extracts noun chunks, and the standard BERT
  tokenizer tokenizes again. Finally we append a mask token to the text, and add
  the answer entity id as a mention at the position of the mask token.

  Args:
    sample: raw CWQ sample. Needs to contain following fields: text_ids,
      target_mention.
    spacy_model: spacy model used to extract mentions.
    tokenizer: BERT tokenizer.

  Returns:
    Processed CWQ sample.
  """
  numpy_sample = {key: value.numpy() for key, value in sample.items()}
  processed_sample = {}

  text_ids = numpy_sample['text_ids']
  target_mention = numpy_sample['target_mention']
  # Skip samples for which the answer is not present in entity vocabulary
  if target_mention == 0:
    return None

  text_ids = text_ids[text_ids.nonzero()]

  # Remove special tokens
  special_tokens_to_remove = [101, 102, 103, 104, 105, 106]
  keep_positions = np.logical_not(np.isin(text_ids, special_tokens_to_remove))
  text_ids = text_ids[keep_positions]

  def detokenize(text_ids, tokenizer):
    text_ids = text_ids[np.nonzero(text_ids)]
    text_tokens = tokenizer.convert_ids_to_tokens(text_ids)
    text_tokens = [
        t[2:] if t.startswith('##') else ' ' + t for t in text_tokens
    ]
    return ''.join(text_tokens)

  # Convert back to text
  text = detokenize(text_ids, tokenizer)

  # Annotate noun chunks
  parsed_text = spacy_model(text)
  mention_char_spans = []
  for chunk in parsed_text.noun_chunks:
    span_start_char = parsed_text[chunk.start].idx
    span_last_token = parsed_text[chunk.end - 1]
    span_end_char = span_last_token.idx + len(span_last_token.text) - 1
    mention_char_spans.append((span_start_char, span_end_char))

  # Tokenize again. Set max length to 1 less than max length as we still have to
  # add the mask token afterwards.
  _, text_ids, text_mask, mention_spans, _ = tokenization_utils.tokenize_with_mention_spans(
      tokenizer=tokenizer,
      sentence=text,
      spans=mention_char_spans,
      max_length=FLAGS.max_length - 1,
      add_bert_tokens=True,
  )
  text_ids = np.array(text_ids)
  text_ids = text_ids[np.nonzero(text_mask)]
  answer_position = len(text_ids)

  # Add mask token
  text_ids = np.append(text_ids, default_values.MASK_TOKEN)

  text_pad_shape = (0, FLAGS.max_length - len(text_ids))
  text_ids = np.pad(text_ids, text_pad_shape)
  text_mask = (text_ids > 0) * 1

  # Truncate mentions, keeping room for final mention.
  mention_spans = mention_spans[:FLAGS.max_mentions - 1]
  mention_spans.append((answer_position, answer_position))
  mention_spans = np.array(mention_spans)

  mention_start_positions = mention_spans[:, 0]
  mention_end_positions = mention_spans[:, 1]

  dense_span_starts = np.zeros_like(text_ids)
  dense_span_starts[mention_start_positions] = 1
  dense_span_ends = np.zeros_like(text_ids)
  dense_span_ends[mention_end_positions] = 1

  dense_mention_ids = np.zeros_like(text_ids)
  dense_mention_ids[answer_position] = target_mention
  dense_mention_mask = (dense_mention_ids > 0) * 1

  processed_sample['text_ids'] = text_ids
  processed_sample['text_mask'] = text_mask
  processed_sample['dense_mention_ids'] = dense_mention_ids
  processed_sample['dense_mention_mask'] = dense_mention_mask
  processed_sample['dense_span_starts'] = dense_span_starts
  processed_sample['dense_span_ends'] = dense_span_ends

  return processed_sample


def process_data(
    raw_samples: List[Dict[str, Any]],
    spacy_model: Any,
    tokenizer: Any,
) -> Tuple[List[Dict[str, Any]], int]:
  """Process CWQ split."""

  processed_samples = []
  skipped = 0
  for sample in tqdm.tqdm(raw_samples):
    processed_sample = process_sample(sample, spacy_model, tokenizer)
    if processed_sample is None:
      skipped += 1
    else:
      processed_samples.append(processed_sample)
  logging.info('Skipped %s samples', skipped)
  return processed_samples, skipped


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  max_length = FLAGS.max_length

  name_to_features = {
      'text_ids': tf.io.FixedLenFeature(max_length, tf.int64),
      'dense_mention_ids': tf.io.FixedLenFeature(max_length, tf.int64),
      'target_mention': tf.io.FixedLenFeature(1, tf.int64),
  }

  split_names = ['train', 'dev']
  split_files = [FLAGS.train_file, FLAGS.dev_file]
  raw_data = {}
  for split_name, split_file in zip(split_names, split_files):
    raw_samples = []
    path = os.path.join(FLAGS.data_dir, split_file)
    logging.info('Loading data from %s', path)
    split_dataset = tf.data.TFRecordDataset(path)
    for sample in split_dataset:
      raw_samples.append(tf.io.parse_single_example(sample, name_to_features))
    raw_data[split_name] = raw_samples

  logging.info('Processing data')
  tokenizer = bert_tokenization.FullTokenizer(
      FLAGS.vocab_path, do_lower_case=True)

  spacy_model = None
  if spacy_model is None:
    spacy_model = spacy.load('en_core_web_md')

  processed_data = {}
  skipped = {}
  for split_name, split_data in raw_data.items():
    processed_split_data, split_skipped = process_data(split_data, spacy_model,
                                                       tokenizer)
    processed_data[split_name] = processed_split_data
    skipped[split_name] = split_skipped

  # Create TFRecords
  tf.io.gfile.makedirs(FLAGS.save_dir)
  for split_name, split_data in processed_data.items():
    file_path = os.path.join(FLAGS.save_dir, split_name)
    logging.info('Writing data to %s', file_path)
    writer = tf.io.TFRecordWriter(file_path)
    for sample in split_data:
      features = tf.train.Features(
          feature={
              key: tf.train.Feature(int64_list=tf.train.Int64List(value=value))
              for key, value in sample.items()
          })

      record_bytes = tf.train.Example(features=features).SerializeToString()
      writer.write(record_bytes)

  # Also save number of skipped samples to help compute true accuracy
  skipped_path = os.path.join(FLAGS.save_dir, 'skipped.json')
  with tf.io.gfile.GFile(skipped_path, 'w+') as skipped_file:
    json.dump(skipped, skipped_file)


if __name__ == '__main__':
  app.run(main)
