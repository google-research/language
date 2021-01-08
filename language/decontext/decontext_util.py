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
"""Utilities for handling decontextualizing examples and predictions."""

import collections
import json

from absl import logging

DecontextLabel = collections.namedtuple('DecontextLabel', [
    'example_id', 'category', 'original_sentence', 'decontextualized_sentence'
])

DecontextExample = collections.namedtuple('DecontextExample', [
    'example_id', 'article_url', 'article_title', 'section_title_list',
    'paragraph_text', 'sent_start_byte_offset', 'sent_end_byte_offset',
    'annotations'
])


def load_decontext_label_from_dict(annot_dict):
  """Reconstruct decontext label from its json representation."""
  label = DecontextLabel(
      example_id=annot_dict['example_id'],
      category=annot_dict['category'],
      original_sentence=annot_dict['original_sentence'],
      decontextualized_sentence=annot_dict['decontextualized_sentence'])
  return label


def load_example_from_jsonl(path_name):
  """Load Decontext input jsonl file into DecontextExample."""
  output_list = []
  with open(path_name) as input_file:
    for line in input_file:
      json_elem = json.loads(line)
      annotations = [
          load_decontext_label_from_dict(annot_dict)
          for annot_dict in json_elem.get('annotations', [])
      ]
      para_text = json_elem['paragraph_text']
      sent_start_offset = json_elem['sentence_start_byte_offset']
      sent_end_offset = json_elem['sentence_end_byte_offset']
      example = DecontextExample(
          example_id=json_elem['example_id'],
          paragraph_text=para_text,
          sent_start_byte_offset=sent_start_offset,
          sent_end_byte_offset=sent_end_offset,
          article_url=json_elem['article_url'],
          article_title=json_elem['page_title'],
          section_title_list=json_elem['section_title'],
          annotations=annotations)
      output_list.append(example)
    return output_list


def load_predictions(fname):
  """Load predictions."""
  example_dict = {}
  with open(fname) as f:
    for line in f:
      json_elem = json.loads(line)
      label = load_decontext_label_from_dict(json_elem)
      if label.example_id in example_dict:
        logging.info('Duplicate predictions for example id %d',
                     label.example_id)
      example_dict[label.example_id] = label
  return example_dict


def load_annotations(fname):
  """Load annotations."""
  annotation_dict = {}
  examples = load_example_from_jsonl(fname)
  for example in examples:
    annotation_dict[example.example_id] = example.annotations
  return annotation_dict
