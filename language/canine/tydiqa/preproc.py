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
"""Performs model-specific preprocessing.

This includes tokenization (i.e. a character splitter for CANINE) and
adding special tokens to the input.

This module does not have any dependencies on TensorFlow and should be re-usable
within your favorite ML/DL framework.
"""

import collections
import functools
import glob
import json
import random

from typing import Any, Dict, List, Mapping, Optional, Sequence, Text, Tuple

from absl import logging
from language.canine.tydiqa import data
from language.canine.tydiqa import tydi_tokenization_interface


def create_entry_from_json(
    json_dict: Mapping[Text, Any],
    tokenizer: tydi_tokenization_interface.TokenizerWithOffsets,
    max_passages: int,
    max_position: int,
    fail_on_invalid: bool,
    ignore_yes_no_answer: bool = True) -> Dict[Text, Any]:
  """Creates an TyDi 'entry' from the raw JSON.

  The 'TyDiEntry' dict is an intermediate format that is later converted into
  the main `TyDiExample` format.

  This function looks up the chunks of text that are candidates for the passage
  answer task, inserts special context tokens such as "[ContextId=0]", and
  creates a byte index to byte index mapping between the document plaintext
  and the concatenation of the passage candidates (these could potentially
  exclude parts of the plaintext document and also include the special tokens).

  In the returned entry, `contexts` includes only the candidate passages and
  has special tokens such as [ContextId=0] added. `span_start` and `span_end`
  are byte-wise indices into `contexts` (not the original corpus plaintext).

  Args:
    json_dict: A single JSONL line, deserialized into a dict.
    tokenizer: Used to create special marker symbols to insert into the text.
    max_passages: see FLAGS.max_passages.
    max_position: see FLAGS.max_position.
    fail_on_invalid: Immediately stop if an error is found?
    ignore_yes_no_answer: Ignore yes_no_answer type in answer's input_text.

  Returns:
    If a failure was encountered and `fail_on_invalid=False`, then returns
    an empty `dict`. Otherwise returns:
    'TyDiEntry' type: a dict-based format consumed by downstream functions:
    entry = {
        "name": str,
        "id": str,
        "language": str,
        "question": {"input_text": str},
        "answer": {
          "candidate_id": annotated_idx,
          "span_text": "",
          "span_start": -1,
          "span_end": -1,
          "input_text": "passage",
        }
        "has_correct_context": bool,
        # Includes special tokens appended.
        "contexts": str,
        # Context index to byte offset in `contexts`.
        "context_to_plaintext_offset": Dict[int, int],
        "plaintext" = json_dict["document_plaintext"]
    }
  """
  for passage_answer in json_dict["passage_answer_candidates"]:
    if (passage_answer["plaintext_start_byte"] == -1 or
        passage_answer["plaintext_end_byte"] == -1):
      # Invalid byte offset found.
      return {}

  add_candidate_types_and_positions(json_dict, max_position)

  # annotated_idx: index of the first annotated context, -1 if null.
  # annotated_min_ans: minimal answer start and end char offsets,
  #                    (-1, -1) if null.
  annotation, annotated_idx, annotated_min_ans = data.get_first_annotation(
      json_dict, max_passages)
  question = {"input_text": json_dict["question_text"]}
  answer = {
      "candidate_id": annotated_idx,
      "span_text": "",
      "span_start": -1,
      "span_end": -1,
      "input_text": "passage",
  }

  # Yes/no answers are added in the input text.
  if annotation is not None:
    assert annotation["yes_no_answer"] in ("YES", "NO", "NONE")
    if annotation["yes_no_answer"] in ("YES", "NO"):
      answer["input_text"] = annotation["yes_no_answer"].lower()

  # Add a minimal answer if one was found.
  if annotated_min_ans != (-1, -1):
    answer["input_text"] = "minimal"
    span_text = data.get_candidate_text(json_dict, annotated_idx).text

    try:
      answer["span_text"] = data.byte_slice(span_text, annotated_min_ans[0],
                                            annotated_min_ans[1])
    except UnicodeDecodeError:
      logging.error("UnicodeDecodeError for example: %s",
                    json_dict["example_id"])
      if fail_on_invalid:
        raise
      return {}
    # local (passage) byte offset
    answer["span_start"], answer["span_end"] = annotated_min_ans
    try:
      expected_answer_text = data.get_text_span(
          json_dict, {
              "plaintext_start_byte":
                  annotation["minimal_answer"]["plaintext_start_byte"],
              "plaintext_end_byte":
                  annotation["minimal_answer"]["plaintext_end_byte"],
          }).text
    except UnicodeDecodeError:
      logging.error("UnicodeDecodeError for example: %s",
                    json_dict["example_id"])
      if fail_on_invalid:
        raise
      return {}
    if expected_answer_text != answer["span_text"]:
      error_message = ("Extracted answer did not match expected answer:"
                       "'{}' vs '{}'".format(expected_answer_text,
                                             answer["span_text"]))
      if fail_on_invalid:
        raise ValueError(error_message)
      else:
        logging.warn(error_message)
        return {}

  # Add a passage answer if one was found
  elif annotation and annotation["passage_answer"]["candidate_index"] >= 0:
    if ignore_yes_no_answer:
      answer["input_text"] = "passage"
    elif annotation["yes_no_answer"] not in ("YES", "NO"):
      # If yes/no answer is added in the input text, keep it. Otherwise,
      # set the answer input text to passage.
      answer["input_text"] = "passage"
    answer["span_text"] = data.get_candidate_text(json_dict, annotated_idx).text
    answer["span_start"] = 0
    answer["span_end"] = data.byte_len(answer["span_text"])

  context_idxs = []
  context_list = []
  for idx, _ in data.candidates_iter(json_dict):
    context = {
        "id": idx,
        "type": get_candidate_type_and_position(json_dict, idx)
    }
    # Get list of all byte positions of the candidate and its plaintext.
    # Unpack `TextSpan` tuple.
    context["text_map"], context["text"] = data.get_candidate_text(
        json_dict, idx)
    if not context["text"]:
      logging.error("ERROR: Found example with empty context %d.", idx)
      if fail_on_invalid:
        raise ValueError(
            "ERROR: Found example with empty context {}.".format(idx))
      return {}
    context_idxs.append(idx)
    context_list.append(context)
    if len(context_list) >= max_passages:
      break

  # Assemble the entry to be returned.
  entry = {
      "name": json_dict["document_title"],
      "id": str(json_dict["example_id"]),
      "language": json_dict["language"],
      "question": question,
      "answer": answer,
      "has_correct_context": annotated_idx in context_idxs
  }
  all_contexts_with_tokens = []
  # `offset` is a byte offset relative to `contexts` (concatenated candidate
  # passages with special tokens added).
  offset = 0
  context_to_plaintext_offset = []
  for idx, context in zip(context_idxs, context_list):
    special_token = tokenizer.get_passage_marker(context["id"])
    all_contexts_with_tokens.append(special_token)
    context_to_plaintext_offset.append([-1] * data.byte_len(special_token))
    # Account for the special token and its trailing space (due to the join
    # operation below)
    offset += data.byte_len(special_token) + 1

    if context["id"] == annotated_idx:
      answer["span_start"] += offset
      answer["span_end"] += offset
    if context["text"]:
      all_contexts_with_tokens.append(context["text"])
      # Account for the text and its trailing space (due to the join
      # operation below)
      offset += data.byte_len(context["text"]) + 1
      context_to_plaintext_offset.append(context["text_map"])
    else:
      if fail_on_invalid:
        raise ValueError("Found example with empty context.")

  # When we join the contexts together with spaces below, we'll add an extra
  # byte to each one, so we have to account for these by adding a -1 (no
  # assigned wordpiece) index at each *boundary*. It's easier to do this here
  # than above since we don't want to accidentally add extra indices after the
  # last context.
  context_to_plaintext_offset = functools.reduce(lambda a, b: a + [-1] + b,
                                                 context_to_plaintext_offset)

  entry["contexts"] = " ".join(all_contexts_with_tokens)
  entry["context_to_plaintext_offset"] = context_to_plaintext_offset
  entry["plaintext"] = json_dict["document_plaintext"]

  if annotated_idx in context_idxs:
    try:
      expected = data.byte_slice(entry["contexts"], answer["span_start"],
                                 answer["span_end"])
    except UnicodeDecodeError:
      logging.error("UnicodeDecodeError for example: %s",
                    json_dict["example_id"])
      if fail_on_invalid:
        raise
      return {}
    # This is a sanity check to ensure that the calculated start and end
    # indices match the reported span text. If this assert fails, it is likely
    # a bug in the data preparation code above. (expected, answer["span_text"])
    if expected != answer["span_text"]:
      logging.warn("*** pruned example id: %d ***", json_dict["example_id"])
      logging.warn("*** %s, %s ***", expected, answer["span_text"])
      return {}
  return entry


def add_candidate_types_and_positions(json_dict, max_position):
  """Adds type and position info to each candidate in the document."""
  count = 0
  for cand in json_dict["passage_answer_candidates"]:
    if count < max_position:
      count += 1
    cand["type_and_position"] = "[Paragraph=%d]" % count


def get_candidate_type_and_position(json_dict, idx):
  """Returns type and position info for the candidate at the given index."""
  if idx == -1:
    return "[NoLongAnswer]"
  else:
    # Gets the 'type_and_position' for this candidate as added by
    # `add_candidate_types_and_positions`. Note that this key is not present
    # in the original TyDi QA corpus.
    return json_dict["passage_answer_candidates"][idx]["type_and_position"]


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id: int,
               example_index: int,
               language_id: data.Language,
               doc_span_index: int,
               wp_start_offset: Sequence[int],
               wp_end_offset: Sequence[int],
               input_ids: Sequence[int],
               input_mask: Sequence[int],
               segment_ids: Sequence[int],
               start_position: Optional[int] = None,
               end_position: Optional[int] = None,
               answer_text: Text = "",
               answer_type: data.AnswerType = data.AnswerType.MINIMAL):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.language_id = language_id
    self.wp_start_offset = wp_start_offset
    self.wp_end_offset = wp_end_offset
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.start_position = start_position  # Index of wordpiece span start.
    self.end_position = end_position  # Index of wordpiece span end (inclusive).
    self.answer_text = answer_text
    self.answer_type = answer_type


def convert_examples_to_features(
    tydi_examples, is_training,
    tokenizer: tydi_tokenization_interface.TokenizerWithOffsets,
    max_question_length, max_seq_length, doc_stride, include_unknowns,
    output_fn):
  """Converts `TyDiExample`s into `InputFeatures` and sends them to `output_fn`.

  Each entry is split into multiple `InputFeatures`, which contains windows

  spans of the article text that contain no more than N wordpieces so as to
  fit within BERT's context window.

  This function assigns `unique_ids` to features, which allow us to identify
  which example a shorter window of text corresponds to.

  Args:
    tydi_examples: generator of `TyDiExample`s generated by
      `read_tydi_examples`.
    is_training: boolean flag
    tokenizer: converts strings into input ids.
    max_question_length: see FLAGS.max_question_length.
    max_seq_length: see FLAGS.max_seq_length.
    doc_stride: see FLAGS.doc_stride.
    include_unknowns: see FLAGS.include_unknowns.
    output_fn: output function to be applied to the features generated from
      examples.

  Returns:
    num_spans_to_id: a dictionary containing a mapping from number of features
      to a list of example ids that has that number of features.
    num_examples: Number of examples from the `tydi_examples` generator.
  """
  num_spans_to_ids = collections.defaultdict(list)
  num_examples = 0
  for tydi_example in tydi_examples:
    example_index = tydi_example.example_id
    # Each TyDi entry is split into multiple features, each of
    # FLAGS.max_seq_length word pieces.
    errors = []
    features = convert_single_example(
        tydi_example,
        tokenizer=tokenizer,
        is_training=is_training,
        max_question_length=max_question_length,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        include_unknowns=include_unknowns,
        errors=errors)
    num_examples += 1

    num_spans_to_ids[len(features)].append(tydi_example.example_id)

    for feature in features:
      feature.example_index = example_index
      # This integer `unique_id` is used for compute_predictions
      # to merge features with example. Both `example_index` and
      # `doc_span_index` are integers, so this works primarily by virtue of
      # the `example_index`s being uniformly distributed with many unoccupied
      # indices between them so as to make collissions unlikely.
      feature.unique_id = feature.example_index + feature.doc_span_index
      output_fn(feature)

  return num_spans_to_ids, num_examples


def find_nearest_wordpiece_index(offset_index: int,
                                 offset_to_wp: Mapping[int, int],
                                 scan_right: bool = True) -> int:
  """According to offset_to_wp dictionary, find the wordpiece index for offset.

  Some offsets do not have mapping to word piece index if they are delimited.
  If scan_right is True, we return the word piece index of nearest right byte,
  nearest left byte otherwise.

  Args:
    offset_index: the target byte offset.
    offset_to_wp: a dictionary mapping from byte offset to wordpiece index.
    scan_right: When there is no valid wordpiece for the offset_index, will
      consider offset_index+i if this is set to True, offset_index-i otherwise.

  Returns:
    The index of the nearest word piece of `offset_index`
    or -1 if no match is possible.
  """
  for i in range(0, len(offset_to_wp.items())):
    next_ind = offset_index + i if scan_right else offset_index - i
    if next_ind >= 0 and next_ind in offset_to_wp:
      return_ind = offset_to_wp[next_ind]
      # offset has a match.
      if return_ind > -1:
        return return_ind
  return -1


def convert_single_example(
    tydi_example: data.TyDiExample,
    tokenizer: tydi_tokenization_interface.TokenizerWithOffsets,
    is_training,
    max_question_length,
    max_seq_length,
    doc_stride,
    include_unknowns,
    errors,
    debug_info=None) -> List[InputFeatures]:
  """Converts a single `TyDiExample` into a list of InputFeatures.

  Args:
    tydi_example: `TyDiExample` from a single JSON line in the corpus.
    tokenizer: Tokenizer object that supports `tokenize` and
      `tokenize_with_offsets`.
    is_training: Are we generating these examples for training? (as opposed to
      inference).
    max_question_length: see FLAGS.max_question_length.
    max_seq_length: see FLAGS.max_seq_length.
    doc_stride: see FLAGS.doc_stride.
    include_unknowns: see FLAGS.include_unknowns.
    errors: List to be populated with error strings.
    debug_info: Dict to be populated with debugging information (e.g. how the
      strings were tokenized, etc.)

  Returns:
    List of `InputFeature`s.
  """
  features = []

  question_wordpieces = tokenizer.tokenize(tydi_example.question)
  # `tydi_example.contexts` includes the entire document (article) worth of
  # candidate passages concatenated with special tokens such as '[ContextId=0]'.
  all_doc_wp, contexts_start_offsets, contexts_end_offsets, offset_to_wp = (
      tokenizer.tokenize_with_offsets(tydi_example.contexts))

  # Check invariants.
  for i in contexts_start_offsets:
    if i > 0:
      assert i < len(tydi_example.context_to_plaintext_offset), (
          "Expected {} to be in `context_to_plaintext_offset` "
          "byte_len(contexts)={}".format(i,
                                         data.byte_len(tydi_example.contexts)))
  for i in contexts_end_offsets:
    if i > 0:
      assert i < len(tydi_example.context_to_plaintext_offset), (
          "Expected {} to be in `context_to_plaintext_offset` "
          "byte_len(contexts)={}".format(i,
                                         data.byte_len(tydi_example.contexts)))

  # The offsets `contexts_start_offsets` and `contexts_end_offsets` are
  # initially in terms of `tydi_example.contexts`, but we need them with regard
  # to the original plaintext from the input corpus.
  # `wp_start_offsets` and `wp_end_offsets` are byte-wise offsets with regard
  # to the original corpus plaintext.
  wp_start_offsets, wp_end_offsets = create_mapping(
      contexts_start_offsets, contexts_end_offsets,
      tydi_example.context_to_plaintext_offset)

  if len(question_wordpieces) > max_question_length:
    # Keeps only the last `max_question_length` wordpieces of the question.
    question_wordpieces = question_wordpieces[-max_question_length:]
  # Inserts the special question marker in front of the question.
  question_wordpieces.insert(0, tokenizer.get_vocab_id("[Q]"))
  if debug_info is not None:
    debug_info["query_wp_ids"] = question_wordpieces

  # DOCUMENT PROCESSING
  # The -3 accounts for
  # 1. [CLS] -- Special BERT class token, which is always first.
  # 2. [SEP] -- Special separator token, placed after question.
  # 3. [SEP] -- Special separator token, placed after article content.
  max_wordpieces_for_doc = max_seq_length - len(question_wordpieces) - 3
  assert max_wordpieces_for_doc >= 0
  if debug_info is not None:
    debug_info["all_doc_wp_ids"] = all_doc_wp

  # We can have documents that are longer than the maximum sequence length.
  # To deal with this we do a sliding window approach, where we take chunks
  # of up to our max length with a stride of `doc_stride`.
  doc_span = collections.namedtuple("DocSpan", ["start", "length"])
  doc_spans = []
  doc_span_start_wp_offset = 0

  while doc_span_start_wp_offset < len(all_doc_wp):
    length = len(all_doc_wp) - doc_span_start_wp_offset
    length = min(length, max_wordpieces_for_doc)
    doc_spans.append(doc_span(start=doc_span_start_wp_offset, length=length))
    if doc_span_start_wp_offset + length == len(all_doc_wp):
      break
    doc_span_start_wp_offset += min(length, doc_stride)

  # ANSWER PROCESSING
  if is_training:
    # Make sure these `TyDiExample`s were created with `is_training=True`.
    assert tydi_example.start_byte_offset is not None
    assert tydi_example.end_byte_offset is not None

    # First, check if we have any hope of finding a start/end wordpiece.
    # (If `offset_to_wp` is empty for this byte range, then it's pointless to
    # try to find a wordpiece range).
    # The `*_byte_offsets` in `tydi_example`.
    has_wordpiece = False
    for i in range(tydi_example.start_byte_offset,
                   tydi_example.end_byte_offset):
      if offset_to_wp.get(i, -1) >= 0:
        has_wordpiece = True
        break
    if not has_wordpiece:
      if debug_info is not None:
        searched_offset_to_wp = []
        for i in range(tydi_example.start_byte_offset,
                       tydi_example.end_byte_offset):
          searched_offset_to_wp.append(i)
        debug_info["offset_to_wp"] = offset_to_wp
        debug_info["searched_offset_to_wp"] = searched_offset_to_wp
      # It looks like the most likely cause of these issues is not having
      # whitespace between Latin/non-Latin scripts, which causes the tokenizer
      # to produce large chunks of non-wordpieced output. Unsurprisingly, the
      # vast majority of such problems arise in Thai and Japanese, which
      # typically do not use space between words.
      errors.append(
          "All byte indices between start/end offset have no assigned "
          "wordpiece.")
      return []

    # Find the indices of the first and last (inclusive) pieces of the answer
    # span. The end_byte_offset is exclusive, so we start our search for the
    # last piece from `end_byte_offset - 1` to ensure we do not select the piece
    # that follows the span.
    assert tydi_example.start_byte_offset <= tydi_example.end_byte_offset
    wp_start_position = find_nearest_wordpiece_index(
        tydi_example.start_byte_offset, offset_to_wp, scan_right=True)
    wp_end_position = find_nearest_wordpiece_index(
        tydi_example.end_byte_offset - 1, offset_to_wp, scan_right=False)

    # Sometimes there's no wordpiece at all for all the offsets in answer,
    # in such case, treat it as a null example.
    if wp_start_position == -1:
      errors.append("No starting wordpiece found.")
      return []
    if wp_end_position == -1:
      errors.append("No ending wordpiece found.")
      return []

  for doc_span_index, doc_span in enumerate(doc_spans):
    wps = []
    wps.append(tokenizer.get_vocab_id("[CLS]"))
    segment_ids = []
    segment_ids.append(0)
    wps.extend(question_wordpieces)
    segment_ids.extend([0] * len(question_wordpieces))
    wps.append(tokenizer.get_vocab_id("[SEP]"))
    segment_ids.append(0)

    wp_start_offset = [-1] * len(wps)
    wp_end_offset = [-1] * len(wps)

    for i in range(doc_span.length):
      split_token_index = doc_span.start + i
      wp_start_offset.append(wp_start_offsets[split_token_index])
      wp_end_offset.append(wp_end_offsets[split_token_index])
      wps.append(all_doc_wp[split_token_index])
      segment_ids.append(1)
    wps.append(tokenizer.get_vocab_id("[SEP]"))
    wp_start_offset.append(-1)
    wp_end_offset.append(-1)
    segment_ids.append(1)
    assert len(wps) == len(segment_ids)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(wps)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(wps)
    padding = [0] * padding_length
    padding_offset = [-1] * padding_length
    wps.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)
    wp_start_offset.extend(padding_offset)
    wp_end_offset.extend(padding_offset)

    assert len(wps) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(wp_start_offset) == max_seq_length
    assert len(wp_end_offset) == max_seq_length

    start_position = None
    end_position = None
    answer_type = None
    answer_text = ""
    wp_error = False
    if is_training:
      doc_start = doc_span.start
      doc_end = doc_span.start + doc_span.length - 1
      # For training, if our document chunk does not contain an annotation
      # we throw it out, since there is nothing to predict.
      contains_an_annotation = (
          wp_start_position >= doc_start and wp_end_position <= doc_end)
      if ((not contains_an_annotation) or
          tydi_example.answer.type == data.AnswerType.UNKNOWN):
        # If an example has unknown answer type or does not contain the answer
        # span, then we only include it with probability --include_unknowns.
        # When we include an example with unknown answer type, we set the first
        # token of the passage to be the annotated short span.
        if (include_unknowns < 0 or random.random() > include_unknowns):
          continue
        start_position = 0
        end_position = 0
        answer_type = data.AnswerType.UNKNOWN
      else:
        doc_offset = len(question_wordpieces) + 2  #  one for CLS, one for SEP.

        start_position = wp_start_position - doc_start + doc_offset
        end_position = wp_end_position - doc_start + doc_offset
        answer_type = tydi_example.answer.type
        answer_start_byte_offset = wp_start_offset[start_position]
        answer_end_byte_offset = wp_end_offset[end_position]
        answer_text = tydi_example.contexts[
            answer_start_byte_offset:answer_end_byte_offset]

        try:
          assert answer_start_byte_offset > -1
          assert answer_end_byte_offset > -1
          assert end_position >= start_position
        except AssertionError:
          errors.append("wp_error")
          wp_error = True
          logging.info(wp_start_position, wp_end_position)
          logging.info(tydi_example.start_byte_offset,
                       tydi_example.end_byte_offset)
          logging.info(start_position, end_position)
          logging.info(doc_offset, doc_start)
          logging.info(tydi_example.example_id)
          logging.info("Error: end position smaller than start: %d",
                       tydi_example.example_id)

    feature = InputFeatures(
        unique_id=-1,  #  this gets assigned afterwards.
        example_index=tydi_example.example_id,
        language_id=tydi_example.language_id,
        doc_span_index=doc_span_index,
        wp_start_offset=wp_start_offset,
        wp_end_offset=wp_end_offset,
        input_ids=wps,
        input_mask=input_mask,
        segment_ids=segment_ids,
        start_position=start_position,
        end_position=end_position,
        answer_text=answer_text,
        answer_type=answer_type)
    if not wp_error:
      features.append(feature)

  return features


def create_mapping(
    start_offsets: Sequence[int],
    end_offsets: Sequence[int],
    context_to_plaintext_offset: Sequence[int],
) -> Tuple[List[int], List[int]]:
  """Creates a mapping from context offsets to plaintext offsets.

  Args:
    start_offsets: List of offsets relative to a TyDi entry's `contexts`.
    end_offsets: List of offsets relative to a TyDi entry's `contexts`.
    context_to_plaintext_offset: Mapping `contexts` offsets to plaintext
      offsets.

  Returns:
    List of offsets relative to the original corpus plaintext.
  """

  plaintext_start_offsets = [
      context_to_plaintext_offset[i] if i >= 0 else -1 for i in start_offsets
  ]
  plaintext_end_offsets = [
      context_to_plaintext_offset[i] if i >= 0 else -1 for i in end_offsets
  ]
  return plaintext_start_offsets, plaintext_end_offsets


def read_tydi_examples(
    input_file, tokenizer: tydi_tokenization_interface.TokenizerWithOffsets,
    is_training, max_passages, max_position, fail_on_invalid, open_fn):
  """Read a TyDi json file into a list of `TyDiExample`.

  Delegates to `preproc.create_entry_from_json` to add special tokens to
  input and handle character offset tracking.

  Args:
    input_file: Path or glob to input JSONL files to be read (possibly gzipped).
    tokenizer: Used to create special marker symbols to insert into the text.
    is_training: Should we create training samples? (as opposed to eval
      samples).
    max_passages: See FLAGS.max_passages.
    max_position: See FLAGS.max_position.
    fail_on_invalid: Should we immediately stop processing if an error is
      encountered?
    open_fn: A function that returns a file object given a path. Usually
      `tf_io.gopen`; could be standard Python `open` if using this module
      outside Tensorflow.

  Yields:
    `TyDiExample`s
  """
  input_paths = glob.glob(input_file)
  if not input_paths:
    raise ValueError("No paths matching glob '{}'".format(input_file))

  non_valid_count = 0
  n = 0
  for path in input_paths:
    logging.info("Reading: %s", path)
    with open_fn(path) as input_file:
      logging.info(path)
      for line in input_file:
        json_dict = json.loads(line, object_pairs_hook=collections.OrderedDict)
        entry = create_entry_from_json(
            json_dict,
            tokenizer,
            max_passages=max_passages,
            max_position=max_position,
            fail_on_invalid=fail_on_invalid)
        if entry:
          tydi_example = data.to_tydi_example(entry, is_training)
          n += 1
          yield tydi_example
        else:
          if fail_on_invalid:
            raise ValueError("Found invalid example.")
          non_valid_count += 1

  if n == 0:
    raise ValueError(
        "No surviving examples from input_file '{}'".format(input_file))

  logging.info("*** # surviving examples %d ***", n)
  logging.info("*** # pruned examples %d ***", non_valid_count)
