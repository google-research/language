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
"""Utilities for rendering sentences and evidence to and from text."""

import collections
import dataclasses
import difflib
import enum
import itertools
import re


from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf
from typing_extensions import Protocol

# Default extra_id ranges for sentence and evidence delimiters.
SENTENCE_START = 0
SENTENCE_END = EVIDENCE_START = 32
EVIDENCE_END = 100

CONTEXT_MARKER = "[CONTEXT]"
EDIT_MARKER = "[EDIT]"
ADDITION_MARKER = "[ADDITION]"
DELETION_MARKER = "[DELETION]"

IOU_THRESHOLD = 0.1


class Task(enum.Enum):
  diff = "diff"
  fullgen = "fullgen"
  controllable = "controllable"


class DelimiterType(enum.Enum):
  text = "text"
  extra_id = "extra_id"


class EvidenceMarkerType(enum.Enum):
  empty = "empty"
  reference = "reference"


class Renderable(Protocol):
  """Abstract base class for objects that can be rendered to text."""

  def __str__(self):
    Ellipsis


def render_all(
    renderables,
    sep = " ",
):
  """Render an iterable of renderable objects.

  Arguments:
    renderables: An iterable of renderable objects.
    sep: The seperator.

  Returns:
    The rendered text.
  """
  return sep.join(map(lambda x: str(x).strip(), renderables)).strip()


# Define a null regex that is incapable of matching any string as the unique
# inverse of an empty fstring.
_NULL_REGEX = re.compile(r"a^")


@dataclasses.dataclass
class InvertibleDelimiterConstructor:
  """Creates / detects indexed renderables that delimit fields within text.

  Attributes:
    fstring: A format string with a `uid` field used to create the delimiter.
    regex: A regular expression with a `uid` group used to detect instances of
      the delimiters within a piece of text.
  """
  fstring: str
  regex: re.Pattern

  def create_delimiter(self, uid):
    return self.fstring.format(uid=uid)

  def __post_init__(self):
    # Edge case: Null delimiters.
    if not self.fstring and self.regex == _NULL_REGEX:
      return
    # Check that regex is able to recover the uid from the format string.
    delimiter = self.create_delimiter(uid=0)
    match = self.regex.search(delimiter)
    if match is not None:
      uid = int(match.group("uid"))
    else:
      uid = None
    if uid != 0:
      raise ValueError(
          f"regex '{self.regex}' does not invert fstring '{self.fstring}'")


null_delimiter_constructor = InvertibleDelimiterConstructor(
    fstring="",
    regex=_NULL_REGEX,
)

square_bracket_delimiter_constructor = InvertibleDelimiterConstructor(
    fstring="[{uid}]", regex=re.compile(r"\[(?P<uid>\d+)\]"))

parens_delimiter_constructor = InvertibleDelimiterConstructor(
    fstring="({uid})", regex=re.compile(r"\((?P<uid>\d+)\)"))

extra_id_delimiter_constructor = InvertibleDelimiterConstructor(
    fstring="<extra_id_{uid}>", regex=re.compile(r"<extra_id_(?P<uid>\d+)>"))


class InvertibleDelimiterRange:
  """Helper class for generating / detecting delimiters within a fixed range.

  Attributes:
    constructor: An invertible delimiter constructor.
    start: (optional) First uid to use when iterating. Default: 0.
    end: (optional) Last uid to use when iterating. If None then uids are
      unbounded. Default: None.
  """

  def __init__(
      self,
      constructor,
      start = 0,
      end = None,
  ):
    self.constructor = constructor
    self.start = start
    self.end = end

  def __iter__(self):

    def delimiter_generator():
      if self.end is not None:
        iterator = range(self.start, self.end)
      else:
        iterator = itertools.count(self.start)
      for i in iterator:
        yield self.constructor.create_delimiter(uid=i)

    return delimiter_generator()

  def _in_range(self, uid):
    """Returns whether candidate UID is within the configured range."""
    start = self.start
    end = float("inf") if self.end is None else self.end
    return start <= uid < end

  def finditer(self, x):
    """Like re.finditer but skips matches outside of configured range."""
    for match in self.constructor.regex.finditer(x):
      uid = int(match.group("uid"))
      if self._in_range(uid):
        yield match

  def finduids(self, x):
    """Like re.findall but returns ints instead of strings."""
    matches = (int(m.group("uid")) for m in self.constructor.regex.finditer(x))
    filtered_matches = filter(self._in_range, matches)
    return list(filtered_matches)

  def split(self, x, keep_delims = False):
    """Like re.split but skips matches outside of configured range."""
    prev_end = 0
    out = []
    for match in self.finditer(x):
      out.append(x[prev_end:match.start()].strip())
      prev_end = match.start() if keep_delims else match.end()
    out.append(x[prev_end:].strip())
    return out

  def remove_delims(self, x):
    """Removes delimiters from a piece of text."""
    return " ".join(self.split(x)).strip()

  def copy_replace(self, x, src):
    """Replaces delimiters in a piece of text with corresponding strings."""
    prev_end = 0
    tmp = []
    for match in self.finditer(x):
      # Copy text between matches
      between = x[prev_end:match.start()].strip()
      if between:
        tmp.append(between)
      # Replace delims with source material
      uid = int(match.group("uid"))
      try:
        tmp.append(src[uid])
      except IndexError:
        logging.error("Bad uid %i for input '%s' with src '%s'", uid, x, src)
      prev_end = match.end()
    # Copy any remaining text
    between = x[prev_end:].strip()
    if between:
      tmp.append(between)
    return " ".join(tmp).strip()


class Evidence:
  """Evidence supporting an edit appearing in the `targets` field."""

  def __init__(
      self,
      delimiter,
      title,
      section,
      text,
      entity_ids = None,
  ):
    self.delimiter = delimiter
    self.title = title
    self.section = section
    self.text = text
    self.entity_ids = entity_ids if entity_ids is not None else set()

  def __str__(self):
    return render_all((self.delimiter, self.title, self.section, self.text))


class Sentence:
  """A delimited sentence w/ entity id & evidence metadata."""

  def __init__(
      self,
      delimiter,
      text,
      evidence_marker_type,
      entity_ids = None,
      evidence = None,
  ):
    self.delimiter = delimiter
    self.text = text
    self.evidence_marker_type = evidence_marker_type
    self.entity_ids = entity_ids if entity_ids is not None else set()
    self.evidence = evidence if evidence is not None else list()

  def __str__(self):
    fields = [self.delimiter]
    if self.evidence_marker_type == EvidenceMarkerType.reference:
      fields.extend(e.delimiter for e in self.evidence)
    fields.append(self.text)
    return render_all(fields)


@dataclasses.dataclass
class DelimiterRangePair:
  sentence_delimiter_range: InvertibleDelimiterRange
  evidence_delimiter_range: InvertibleDelimiterRange


def get_default_delimiter_range_pair(
    task,
    delimiter_type,
):
  """Default delimiter range pair for a task/delimiter type combo."""
  if delimiter_type == DelimiterType.text:
    if task == Task.fullgen:
      sentence_delimiter_range = InvertibleDelimiterRange(
          null_delimiter_constructor,)
    else:
      sentence_delimiter_range = InvertibleDelimiterRange(
          square_bracket_delimiter_constructor,)
    evidence_delimiter_range = InvertibleDelimiterRange(
        parens_delimiter_constructor,)
  # TODO(rloganiv): Is there a reason we don't use null delimiters for fullgen
  # here?
  elif delimiter_type == DelimiterType.extra_id:
    sentence_delimiter_range = InvertibleDelimiterRange(
        extra_id_delimiter_constructor,
        start=SENTENCE_START,
        end=SENTENCE_END,
    )
    evidence_delimiter_range = InvertibleDelimiterRange(
        extra_id_delimiter_constructor,
        start=EVIDENCE_START,
        end=EVIDENCE_END,
    )
  return DelimiterRangePair(
      sentence_delimiter_range=sentence_delimiter_range,
      evidence_delimiter_range=evidence_delimiter_range,
  )


class Tokenizer(Protocol):
  """Abstract class for tokenizers."""

  def tokenize(self, text, *args, **kwargs):
    Ellipsis


class SpanTokenizer(Protocol):
  """Abstract class for span tokenizers."""

  def span_tokenize(
      self,
      text,
      *args,
      **kwargs,
  ):
    Ellipsis


def iou_score(sent_a, sent_b):
  """Token intersection over union score between two sentences."""
  sent_a_tokens = set(sent_a.split(" "))
  sent_b_tokens = set(sent_b.split(" "))
  intersection = sent_a_tokens & sent_b_tokens
  union = sent_a_tokens | sent_b_tokens
  return len(intersection) / len(union)


def score_matrix(sents_a, sents_b):
  """Creates a matrix of match scores."""
  scores = np.zeros((len(sents_a), len(sents_b)))
  for i, sent_a in enumerate(sents_a):
    for j, sent_b in enumerate(sents_b):
      scores[i, j] = iou_score(sent_a, sent_b)
  return scores


def _recursion(scores, i):
  """Recursive approach for finding best matches."""
  if scores.shape[0] == 0 or scores.shape[1] == 0:
    return [], 0.0
  index = np.argmax(scores[0, :])
  score = scores[0, index]
  if score > IOU_THRESHOLD:
    seq_, score_ = _recursion(scores[1:, 1:], i + 1)
    return [index + i, *seq_], score_ + score
  else:
    seq_, score_ = _recursion(scores[1:, :], i)
    return [None, *seq_], score_


def match_sents(sents_a, sents_b):
  """Finds best matching between two lists of sentences."""
  scores = score_matrix(sents_a, sents_b)
  best_score = -float("inf")
  best_seq = None
  for i in range(len(sents_a)):
    seq, score = _recursion(scores[i:,], 0)
    if score > best_score:
      best_seq = [None] * i + seq
      best_score = score
  return best_seq, best_score


class Seq2SeqInputBuilder:
  """Returns a callable object that creates seq2seq inputs."""

  def __init__(
      self,
      task,
      delimiter_range_pair,
      tokenizer,
      sentence_tokenizer,
      include_source = False,
      include_evidence = True,
      include_distractors = True,
      evidence_marker_type = EvidenceMarkerType.empty,
      max_input_length = 1024,
      filter_no_diff = True,
  ):
    self.task = task
    self.delimiter_range_pair = delimiter_range_pair
    self.tokenizer = tokenizer
    self.sentence_tokenizer = sentence_tokenizer
    self.include_source = include_source
    self.include_evidence = include_evidence
    self.include_distractors = include_distractors
    self.evidence_marker_type = evidence_marker_type
    self.max_input_length = max_input_length
    self.filter_no_diff = filter_no_diff

    self.sentence_delimiter_range = (
        delimiter_range_pair.sentence_delimiter_range)
    self.evidence_delimiter_range = (
        delimiter_range_pair.evidence_delimiter_range)

  def _convert_to_evidence(
      self,
      annotated_mentions,
  ):
    """Converts annotated mentions to evidence renderables."""
    evidence = []
    for delimiter, annotated_mention in zip(self.evidence_delimiter_range,
                                            annotated_mentions):
      mention = annotated_mention["mention"]
      title = mention["title"]
      section = mention["section"]
      text = mention["text"]
      entity_ids = {x["id"] for x in mention["entities"]} | {mention["title"]}
      e = Evidence(
          delimiter=delimiter,
          title=title,
          section=section,
          text=text,
          entity_ids=entity_ids)
      evidence.append(e)
    return evidence

  def _convert_source_sentences(
      self,
      source,
  ):
    """Converts source text to sentence renderables."""
    sentences = []
    for delimiter, (start, end) in zip(
        self.sentence_delimiter_range,
        self.sentence_tokenizer.span_tokenize(source["text"])):
      text = source["text"][start:end]
      # Source sentences do not contain evidence markers unless we are in the
      # controllable setting.
      if self.task == Task.controllable:
        evidence_marker_type = self.evidence_marker_type
      else:
        evidence_marker_type = EvidenceMarkerType.empty
      sentence = Sentence(
          delimiter, text, evidence_marker_type=evidence_marker_type)
      sentences.append(sentence)
    return sentences

  def _convert_target_sentences(
      self,
      target,
  ):
    """Converts target text to sentence renderables."""
    sentences = []
    for start, end in self.sentence_tokenizer.span_tokenize(target["text"]):
      text = target["text"][start:end]
      entities = target["added_entities"]
      entity_ids = {
          e["id"] for e in entities if e["start"] >= start and e["end"] <= end
      }
      # Target sentences do not contain evidence markers if we are in the
      # controllable setting.
      if self.task == Task.controllable:
        evidence_marker_type = EvidenceMarkerType.empty
      else:
        evidence_marker_type = self.evidence_marker_type
      sentence = Sentence(
          "",  # Targets sentences are never delimited.
          text,
          evidence_marker_type=evidence_marker_type,
          entity_ids=entity_ids,
      )
      sentences.append(sentence)
    return sentences

  def _get_generatable_entity_ids(
      self,
      target,
      evidence,
  ):
    """Gets set of generatable entity ids."""
    generatable_entity_ids = set()
    for entity in target["added_entities"]:
      if any(entity["id"] in e.entity_ids for e in evidence):
        generatable_entity_ids.add(entity["id"])
    return generatable_entity_ids

  def _get_generatable_surfaces(
      self,
      target,
      annotated_mentions,
      generatable_entity_ids,
  ):
    """Retrieves surfaces for generatable entities."""
    generatable_surfaces = collections.defaultdict(set)
    for entity in target["added_entities"]:
      if entity["id"] in generatable_entity_ids:
        surface = target["text"][entity["start"]:entity["end"]]
        generatable_surfaces[entity["id"]].add(surface)
    for annotated_mention in annotated_mentions:
      mention = annotated_mention["mention"]
      for entity in mention["entities"]:
        if entity["id"] in generatable_entity_ids:
          surface = mention["text"][entity["start"]:entity["end"]]
          generatable_surfaces[entity["id"]].add(surface)
    generatable_surfaces = {k: list(v) for k, v in generatable_surfaces.items()}
    return generatable_surfaces

  def _get_length(self, x):
    """Gets the number of tokens in a string."""
    tokens = self.tokenizer.tokenize(x)
    length = tokens.shape[0]
    return length

  def _get_controllable_inputs_targets(
      self,
      source_sentences,
      target_sentences,
      evidence,
  ):
    """Constructs the inputs and targets for the controllable setting."""
    # The difference in the controllable setting is that, when we determine what
    # evidence justifies an update, we need to transfer the marker to the
    # associated source sentence.
    inputs_renderables = []
    targets_renderables = []

    source_texts = [str(s.text).strip() for s in source_sentences]
    target_texts = [str(t.text).strip() for t in target_sentences]
    cruncher = difflib.SequenceMatcher(a=source_texts, b=target_texts)
    update_pairs = []
    for tag, alo, ahi, blo, bhi in cruncher.get_opcodes():
      if tag == "replace":
        # Compute the best matching
        source_sentences_subset = source_sentences[alo:ahi]
        target_sentences_subset = target_sentences[blo:bhi]
        source_texts_subset = source_texts[alo:ahi]
        target_texts_subset = target_texts[blo:bhi]
        best_match, _ = match_sents(source_texts_subset, target_texts_subset)

        # Depending on whether or not there is a match add an addition, edit,
        # or deletion.
        last_match_idx = 0
        for source_sentence, match_idx in zip(source_sentences_subset,
                                              best_match):
          # If source sent didn't get matched it needs to be deleted and we
          # proceed.
          if match_idx is None:
            inputs_renderables.extend([DELETION_MARKER, source_sentence])
            continue
          # Otherwise, we first check if there is a gap between the last and
          # current match index. If there is then target sentences need to be
          # added.
          if match_idx != last_match_idx:
            for target_sentence in target_sentences_subset[
                last_match_idx:match_idx]:
              placeholder_input = Sentence(ADDITION_MARKER, "",
                                           self.evidence_marker_type)
              inputs_renderables.append(placeholder_input)
              targets_renderables.append(target_sentence)
              update_pairs.append((placeholder_input, target_sentence))
          # We then add the sentence pair and update the last match index.
          inputs_renderables.extend([EDIT_MARKER, source_sentence])
          targets_renderables.append(target_sentences_subset[match_idx])
          update_pairs.append(
              (source_sentence, target_sentences_subset[match_idx]))
          last_match_idx = match_idx + 1
        # Lastly we add any remaining target sentences.
        if last_match_idx != len(target_sentences_subset):
          for target_sentence in target_sentences_subset[last_match_idx:]:
            placeholder_input = Sentence(ADDITION_MARKER, "",
                                         self.evidence_marker_type)
            inputs_renderables.append(placeholder_input)
            targets_renderables.append(target_sentence)
            update_pairs.append((placeholder_input, target_sentence))
      elif tag == "insert":
        for i in range(blo, bhi):
          # We create a placeholder sentence to associate evidence to later
          placeholder_input = Sentence(ADDITION_MARKER, "",
                                       self.evidence_marker_type)
          inputs_renderables.append(placeholder_input)
          targets_renderables.append(target_sentences[i])
          update_pairs.append((placeholder_input, target_sentences[i]))
      elif tag == "equal":
        inputs_renderables.extend(source_sentences[alo:ahi])
        targets_renderables.extend(
            s.delimiter for s in source_sentences[alo:ahi])
      elif tag == "delete":
        for s in source_sentences[alo:ahi]:
          inputs_renderables.extend([DELETION_MARKER, s])

    # Add context
    inputs_renderables.append(CONTEXT_MARKER)
    inputs_temp = render_all(inputs_renderables)
    input_length = self._get_length(inputs_temp)
    for e in evidence:
      evidence_length = self._get_length(str(e))
      if input_length + evidence_length > self.max_input_length:
        continue
      # Check if evidence supports an update
      is_support = False
      for src, tgt in update_pairs:
        # THIS IS WHERE THE MAGIC HAPPENS
        if tgt.entity_ids & e.entity_ids:
          src.evidence.append(e)
          is_support = True
      if (is_support and self.include_evidence) or self.include_distractors:
        inputs_renderables.append(e)
      input_length += evidence_length

    inputs = render_all(inputs_renderables)
    targets = render_all(targets_renderables)

    yield {"inputs": inputs, "targets": targets}

  def _get_inputs_targets(
      self,
      source_sentences,
      target_sentences,
      evidence,
  ):
    """Produces inputs / targets from sentences and evidence."""
    # Inputs and targets constructed by pasting together renderables.
    inputs_renderables = source_sentences.copy()
    targets_renderables = []

    # Regardless of task we need to know diff to determine updates. Also, we're
    # using only the text and stripping to avoid issues with delimiters and
    # whitespace.
    source_texts = [str(s.text).strip() for s in source_sentences]
    target_texts = [str(t.text).strip() for t in target_sentences]
    cruncher = difflib.SequenceMatcher(a=source_texts, b=target_texts)
    updated = []
    for tag, alo, ahi, blo, bhi in cruncher.get_opcodes():
      if tag in ("replace", "insert"):
        targets_renderables.extend(target_sentences[blo:bhi])
        updated.extend(target_sentences[blo:bhi])
      elif tag == "equal":
        if self.task == Task.fullgen:
          targets_renderables.extend(target_sentences[blo:bhi])
        else:
          targets_renderables.extend(
              s.delimiter for s in source_sentences[alo:ahi])
      elif tag == "delete":
        continue
    if self.filter_no_diff and not updated:
      return

    # Add context
    inputs_renderables.append(CONTEXT_MARKER)
    inputs_temp = render_all(inputs_renderables)
    input_length = self._get_length(inputs_temp)
    for e in evidence:
      evidence_length = self._get_length(str(e))
      if input_length + evidence_length > self.max_input_length:
        continue
      # Check if evidence supports an update
      is_support = False
      for u in updated:
        if u.entity_ids & e.entity_ids:
          u.evidence.append(e)
          is_support = True
      if (is_support and self.include_evidence) or self.include_distractors:
        inputs_renderables.append(e)
      input_length += evidence_length

    inputs = render_all(inputs_renderables)
    targets = render_all(targets_renderables)

    yield {"inputs": inputs, "targets": targets}

  def __call__(
      self,
      article_pair,
  ):
    source = article_pair.get("source_article")
    target = article_pair.get("target_article")
    annotated_mentions = article_pair.get("annotated_mentions")
    if not source or not target:
      return

    evidence = self._convert_to_evidence(annotated_mentions)
    # TODO(rloganiv): Can avoid these routines now as everything that is
    # generatable is determined during input / target construction.
    generatable_entity_ids = self._get_generatable_entity_ids(target, evidence)
    generatable_surfaces = self._get_generatable_surfaces(
        target, annotated_mentions, generatable_entity_ids)
    source_sentences = self._convert_source_sentences(source)
    target_sentences = self._convert_target_sentences(target)

    if self.task == Task.controllable:
      inputs_targets_fn = self._get_controllable_inputs_targets
    else:
      inputs_targets_fn = self._get_inputs_targets

    for instance in inputs_targets_fn(source_sentences, target_sentences,
                                      evidence):
      yield {
          **instance,
          "id": target["id"],
          "generatable_surfaces": generatable_surfaces,
      }


def normalize(
    source,
    edit,
    delimiter_range_pair,
    task = None,
):
  """Applies sentence level edits to a source document."""

  evidence_delimiter_range = delimiter_range_pair.evidence_delimiter_range
  sentence_delimiter_range = delimiter_range_pair.sentence_delimiter_range

  # Remove context from clean source.
  source, _ = source.split(CONTEXT_MARKER)
  source = source.strip()

  # In controllable setting we need to remove all of the additional markers from
  # the input in order to make it similar to the diff format.
  if task == Task.controllable:
    source = source.replace(ADDITION_MARKER, "")
    source = source.replace(DELETION_MARKER, "")
    source = source.replace(EDIT_MARKER, "")
    source = evidence_delimiter_range.remove_delims(source)

  # Split source into sentences using sentence delimiters, and recover clean
  # source by joining delimiter-less sentences.
  clean_source = sentence_delimiter_range.remove_delims(source)

  # Remove evidence delimiters from edit if present.
  clean_edit = evidence_delimiter_range.remove_delims(edit)

  # Iterate over copy delimiters in edit to identify which source sentences
  # to keep vs edit and join them together.
  # NOTE: Source UIDs are zero-delimited, however there will be at least one
  # (possibly empty) string before the first delimiter, hence we always reject
  # the first "sentence" in the split.
  source_sentences = sentence_delimiter_range.split(source)[1:]
  clean_edited = sentence_delimiter_range.copy_replace(
      clean_edit, src=source_sentences)

  #  # Lastly remove any double whitespace.
  clean_source = re.sub(r"\s{2,}", " ", clean_source)
  clean_edited = re.sub(r"\s{2,}", " ", clean_edited)

  return clean_source, clean_edited


def extract_additions(source, target):
  """Simple heuristic for extracting added text from normalized inputs/outputs."""
  normalized_additions = []
  for match in re.finditer(r"[^.]+\.?", target):
    if match.group(0) not in source:
      normalized_additions.append(match.group(0).strip())
  return normalized_additions
