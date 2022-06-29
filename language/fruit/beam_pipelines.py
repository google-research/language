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
r"""Pipelines for processing data using Apache Beam."""

import difflib
import json
import os

from absl import logging
import apache_beam as beam
from language.fruit import rendering_utils
from language.fruit import tf_utils
from language.fruit import wiki_utils
import matplotlib.pyplot as plt
import nltk
import numpy as np
import tensorflow as tf
import tensorflow_text as tft

MAX_MENTIONS = 1024
SENTENCE_TOKENIZER_PATH = "tokenizers/punkt/english.pickle"

Task = rendering_utils.Task
DelimiterType = rendering_utils.DelimiterType
EvidenceMarkerType = rendering_utils.EvidenceMarkerType

## General #####################################################################


def clean_titles(element):
  """Cleans article titles."""
  element["title"] = wiki_utils.clean_wikilink(element["title"])
  return element


def filter_articles(element):
  """Filters articles."""
  if "redirect" not in element:
    beam.metrics.Metrics.counter(__name__, "articles").inc()
    yield element


def filter_redirects(element):
  """Filters redirects."""
  if "redirect" in element:
    beam.metrics.Metrics.counter(__name__, "redirects").inc()
    yield element


## Redirect Table ##############################################################


def redirect_to_id(element, lookup):
  """Creates mapping of redirect titles to article ids."""
  redirect_title = element["title"]
  article_title = wiki_utils.clean_wikilink(element["redirect"])
  if article_title in lookup:
    yield redirect_title, lookup[article_title]
  else:
    # Even if we haven't seen the article we still use the provided redirect.
    logging.debug("Unresolved redirect: %s", redirect_title)
    beam.metrics.Metrics.counter(__name__, "unresolved redirects").inc()
    yield redirect_title, article_title


def redirect_table_pipeline(input_jsonl, output_tsv):
  """Beam pipeline for producing a redirect table as a TSV."""

  def pipeline(root):
    all_entries = (
        root
        | "ReadInput" >> beam.io.ReadFromText(input_jsonl)
        | "ParseJSONL" >> beam.Map(json.loads)
        | "CleanTitles" >> beam.Map(clean_titles))
    articles = (all_entries | "FilterArticles" >> beam.FlatMap(filter_articles))
    redirects = (
        all_entries
        | "FilterRedirects" >> beam.FlatMap(filter_redirects))
    article_id_lookup = (
        articles
        | "ArticleTitlesToIDs" >> beam.Map(lambda x: (x["title"], x["title"])))
    article_id_lookup_dict = beam.pvalue.AsDict(article_id_lookup)
    redirect_id_lookup = (
        redirects
        | "RedirectTitlesToIDs" >> beam.FlatMap(redirect_to_id,
                                                article_id_lookup_dict))
    _ = ((article_id_lookup, redirect_id_lookup)
         | "CombineTitlesToIDs" >> beam.Flatten()
         | "AsTSV" >> beam.Map("\t".join)
         | "Serialize" >> beam.io.WriteToText(output_tsv))

  return pipeline


## Process Snapshot ############################################################


def parse_wikitext(article, keep_tables, truncate):
  """Parses the wikitext markup in an article to text."""
  article["text"] = wiki_utils.process_wikitext(
      article["text"],
      keep_tables=keep_tables,
      truncate=truncate,
  )
  if not article["text"]:
    beam.metrics.Metrics.counter(__name__, "textless_articles").inc()
    logging.debug("Textless article: %s (%i)", article["title"], article["id"])
    return
  yield article["title"], article


def split_sections(element):
  """Splits an article into sections, treating tables separately."""
  key, article = element
  total = 0
  for subkey, section in wiki_utils.split_article(article["text"]):
    if not section:
      continue
    beam.metrics.Metrics.counter(__name__, "sections").inc()
    new_key = (key, subkey)
    article["text"] = section
    yield new_key, article
    total += 1
  beam.metrics.Metrics.distribution(__name__, "sections_dist").update(total)


class ProcessArticles(beam.PTransform):
  """Read and process a Wikipedia JSONL dump."""

  def __init__(self, keep_tables, truncate):
    super().__init__()
    self.keep_tables = keep_tables
    self.truncate = truncate

  def expand(self, input_or_inputs):
    # Get all articles and redirects
    all_entries = (
        input_or_inputs
        | "Parse JSONL" >> beam.Map(json.loads)
        | "Clean Titles" >> beam.Map(clean_titles))
    articles = (
        all_entries
        | "Filter Articles" >> beam.FlatMap(filter_articles))
    clean_articles = (
        articles
        | "Parse Wikitext" >> beam.FlatMap(
            parse_wikitext,
            keep_tables=self.keep_tables,
            truncate=self.truncate)
        | "Split Sections" >> beam.FlatMap(split_sections))
    return clean_articles


def clean_merge(element):
  """Cleans up merged articles."""
  key, value = element
  # If there is not corresponding target article then mention was either
  # removed or there was some kind of structural change to the article that we
  # can't handle so we'll skip this pair.
  if not value["target_article"]:
    logging.debug("No target article: %s", key)
    beam.metrics.Metrics.counter(__name__, "source_sections_w_o_target").inc()
    return
  # If we've somehow managed to pair more than one article/section together,
  # then skip.
  if len(value["source_article"]) > 1 or len(value["target_article"]) > 1:
    logging.debug("Not one-to-one: %s", key)
    beam.metrics.Metrics.counter(__name__, "non_1_to_1_sections").inc()
    return
  # Clean up the joined table by removing singletons from lists, and replacing
  # empty lists with None.
  value = {k: v[0] if v else None for k, v in value.items()}
  # Lastly, ignore cases where the ID has changed
  if value["source_article"]:
    if value["source_article"]["id"] != value["target_article"]["id"]:
      logging.debug("ID mismatch: %s", key)
      beam.metrics.Metrics.counter(__name__, "id_mismatches").inc()
      return
    if (value["source_article"]["text"] == "DISAMBIGUATION" or
        value["target_article"]["text"] == "DISAMBIGUATION"):
      beam.metrics.Metrics.counter(__name__, "disambiguation_pages").inc()
      return
  yield key, value


def get_text(value):
  """Retrieves text from article pairs w/ logic for new articles."""
  if value["source_article"]:
    source_text = value["source_article"]["text"]
  else:
    source_text = ""
  target_text = value["target_article"]["text"]
  return source_text, target_text


def disambiguate(entities, lookup):
  """Disambiguates redirects."""
  out = []
  for entity in entities:
    try:
      disambiguated = lookup[entity["id"]]
    except KeyError:
      beam.metrics.Metrics.counter(__name__, "unknown_ids").inc()
      disambiguated = entity["id"]
    else:
      beam.metrics.Metrics.counter(__name__, "known_ids").inc()
    out.append({
        "id": disambiguated,
        "start": entity["start"],
        "end": entity["end"]
    })
  return out


class DiffArticles(beam.DoFn):
  """Computes diffs between articles."""

  # pylint: disable=abstract-method

  def __init__(self, use_source_mentions):
    super().__init__()
    self.use_source_mentions = use_source_mentions

  def table_diff(self, key, value, source_id_lookup, target_id_lookup):
    """Computes diffs for tables."""
    title, section = key
    source_text, target_text = get_text(value)

    clean_source_text, source_entities = wiki_utils.process_wikilinks(
        source_text)
    source_entities = disambiguate(source_entities, source_id_lookup)
    source_entity_ids = set(
        x["id"] for x in source_entities if x["id"] != title)

    ## Source mentions
    if self.use_source_mentions and source_entity_ids:
      beam.metrics.Metrics.counter(__name__, "source_tables").inc()
      mention = {
          "title": title,
          "section": section,
          "text": clean_source_text,
          "is_update": False,
          "entities": source_entities,
          "added_entities": [],
      }
      new_key = hash(json.dumps(mention))
      mention["key"] = new_key
      yield mention

    ## Updated mentions
    clean_target_text, target_entities = wiki_utils.process_wikilinks(
        target_text)
    target_entities = disambiguate(target_entities, target_id_lookup)
    target_entity_ids = set(
        x["id"] for x in target_entities if x["id"] != title)

    # Added entities must satisfy two criteria:
    # 1. The link must not appear in the source article
    # 2. The surface must not appear in the source article
    added_entity_ids = target_entity_ids - source_entity_ids
    added_entities = []
    for entity in target_entities:
      if entity["id"] not in added_entity_ids:
        continue
      if clean_target_text[entity["start"]:entity["end"]] in clean_source_text:
        continue
      else:
        added_entities.append(entity)
    if added_entity_ids and clean_source_text != clean_target_text:
      beam.metrics.Metrics.counter(__name__, "updated_tables").inc()
      mention = {
          "title": title,
          "section": section,
          "text": clean_target_text,
          "is_update": True,
          "entities": target_entities,
          "added_entities": added_entities,
      }
      new_key = hash(json.dumps(mention))
      mention["key"] = new_key
      yield mention

  def text_diff(self, key, value, source_id_lookup, target_id_lookup):
    """Computes diffs for tables."""
    title, section = key
    source_text, target_text = get_text(value)

    ## Source mentions
    source_sentences = nltk.sent_tokenize(source_text)
    clean_source_sentences = []
    source_entity_ids = set()
    for sentence in source_sentences:
      clean_sentence, entities = wiki_utils.process_wikilinks(sentence)
      entities = disambiguate(entities, source_id_lookup)
      entity_ids = set(x["id"] for x in entities if x["id"] != title)

      if self.use_source_mentions and entity_ids:
        beam.metrics.Metrics.counter(__name__, "source_mentions").inc()
        mention = {
            "title": title,
            "section": section,
            "text": clean_sentence,
            "is_update": False,
            "entities": entities,
            "added_entities": [],
        }
        new_key = hash(json.dumps(mention))
        mention["key"] = new_key
        yield mention

      clean_source_sentences.append(clean_sentence)
      source_entity_ids.update(entity_ids)

    target_sentences = nltk.sent_tokenize(target_text)
    clean_target_sentences = []
    target_entities = []
    for sentence in target_sentences:
      clean_sentence, entities = wiki_utils.process_wikilinks(sentence)
      entities = disambiguate(entities, target_id_lookup)
      clean_target_sentences.append(clean_sentence)
      target_entities.append(entities)

    # Use difflib to ignore sentences whose surface text has not been updated.
    cruncher = difflib.SequenceMatcher(
        a=clean_source_sentences, b=clean_target_sentences)
    opcodes = filter(lambda x: x[0] in ["insert", "replace"],
                     cruncher.get_opcodes())
    for *_, lo, hi in opcodes:
      for sentence, entities in zip(clean_target_sentences[lo:hi],
                                    target_entities[lo:hi]):
        entity_ids = set(x["id"] for x in entities if x["id"] != title)
        added_entity_ids = entity_ids - source_entity_ids
        added_entities = []
        for entity in entities:
          if entity["id"] not in added_entity_ids:
            continue
          if sentence[entity["start"]:entity["end"]] in clean_source_sentences:
            continue
          else:
            added_entities.append(entity)
        if added_entity_ids:
          beam.metrics.Metrics.counter(__name__, "updated_mentions").inc()
          mention = {
              "title": title,
              "section": section,
              "text": sentence,
              "is_update": True,
              "entities": entities,
              "added_entities": added_entities,
          }
          new_key = hash(json.dumps(mention))
          mention["key"] = new_key
          yield mention

  def process(self, element, source_id_lookup, target_id_lookup):
    """Processes element."""
    # pylint: disable=arguments-differ
    key, value = element
    _, section = key
    if "Table-" in section:
      yield from self.table_diff(key, value, source_id_lookup, target_id_lookup)
    else:
      yield from self.text_diff(key, value, source_id_lookup, target_id_lookup)


def rekey_mentions(element):
  """Re-keys mentions in preparation for article merge."""
  if element["is_update"]:
    for entity in element["added_entities"]:
      yield entity["id"], element
  else:
    for entity in element["entities"]:
      yield entity["id"], element


def filter_rekey(element, source_id_lookup, target_id_lookup):
  """Filter merged articles and rekey by entity id."""
  key, value = element
  title, section = key

  # Only keep introductory paragraphs.
  if "INTRODUCTION" not in section:
    return

  source_text, target_text = get_text(value)
  clean_source_text, source_entities = wiki_utils.process_wikilinks(source_text)
  clean_target_text, target_entities = wiki_utils.process_wikilinks(target_text)
  updated = True
  if clean_source_text == clean_target_text:
    updated = False
  source_entities = disambiguate(source_entities, source_id_lookup)
  source_entity_ids = set(x["id"] for x in source_entities if x != title)
  target_entities = disambiguate(target_entities, target_id_lookup)
  target_entity_ids = set(x["id"] for x in target_entities if x != title)
  added_entity_ids = target_entity_ids - source_entity_ids
  added_entities = []
  for entity in target_entities:
    if entity["id"] not in added_entity_ids:
      continue
    if clean_target_text[entity["start"]:entity["end"]] in clean_source_text:
      continue
    else:
      added_entities.append(entity)
  if not added_entities:
    updated = False
  if value["source_article"]:
    value["source_article"]["text"] = clean_source_text
    value["source_article"]["entities"] = source_entities
    value["source_article"]["added_entities"] = []
  else:
    value["source_article"] = {
        "id": None,
        "ns": None,
        "text": "",
        "entities": [],
        "added_entities": []
    }
  value["target_article"]["text"] = clean_target_text
  value["target_article"]["entities"] = target_entities
  value["target_article"]["added_entities"] = added_entities
  value["updated"] = updated
  #  value["added_entity_ids"] = list(added_entity_ids)
  if updated:
    beam.metrics.Metrics.counter(__name__, "updated_articles").inc()
  else:
    beam.metrics.Metrics.counter(__name__, "unchanged_articles").inc()
  yield title, value


def annotate_mentions(element, third_party):
  """Annotate reciprocal mentions."""
  key, value = element
  if not value["merged_articles"]:
    logging.debug("Article-less mention: %s", key)
    beam.metrics.Metrics.counter(__name__, "article-less_mentions").inc()
    return
  if len(value["merged_articles"]) > 1:
    logging.debug("Many-to-many article mention: %s", key)
    beam.metrics.Metrics.counter(__name__,
                                 "many-to-many_article_mentions").inc()
    return
  if not value["mentions"]:
    beam.metrics.Metrics.counter(__name__, "mention-less_articles").inc()
  else:
    beam.metrics.Metrics.counter(__name__, "articles_w_mentions").inc()

  merged_articles = value["merged_articles"].pop()
  out = {
      "source_article": merged_articles["source_article"],
      "target_article": merged_articles["target_article"],
      "updated": merged_articles["updated"],
  }
  # Reciprocal mentions are those whose subject (optionally + added entities)
  # intersect the article's added entities.
  entity_ids = set(
      x["id"] for x in merged_articles["target_article"]["added_entities"])
  num_positives = 0
  annotated_mentions = []
  for mention in value["mentions"]:
    label = 0
    if mention["title"] in entity_ids:
      label = 1
      num_positives += 1
    elif third_party:
      if set(x["id"] for x in mention["entities"]) & entity_ids:
        label = 1
        num_positives += 1
    annotated_mentions.append({
        "mention": mention,
        "label": label,
    })
  annotated_mentions.sort(
      key=lambda x: (x["mention"]["is_update"], x["label"]), reverse=True)
  annotated_mentions = annotated_mentions[:MAX_MENTIONS]
  out["annotated_mentions"] = annotated_mentions

  yield out


def process_snapshot_pipeline(
    source_jsonl,
    target_jsonl,
    output_dir,
    source_redirects,
    target_redirects,
    keep_tables=True,
    third_party=True,
    truncate=False,
    use_source_mentions=True,
):
  """Pipeline for processing Wikipedia snapshots."""

  def pipeline(root):
    source_articles = (
        root
        | "ReadSource" >> beam.io.ReadFromText(source_jsonl)
        | "ProcessSourceArticles" >> ProcessArticles(
            keep_tables=keep_tables, truncate=truncate))
    target_articles = (
        root
        | "ReadTarget" >> beam.io.ReadFromText(target_jsonl)
        | "ProcessTargetArticles" >> ProcessArticles(
            keep_tables=keep_tables, truncate=truncate))
    merged_articles = ({
        "source_article": source_articles,
        "target_article": target_articles,
    }
                       | "MergeArticles" >> beam.CoGroupByKey()
                       | "ReshardBeforeClean" >> beam.Reshuffle()
                       | "CleanMergedArticles" >> beam.FlatMap(clean_merge)
                       | "ReshardAfterClean" >> beam.Reshuffle())
    source_id_lookup = beam.pvalue.AsDict(
        root
        | "ReadSourceRedirects" >> beam.io.ReadFromText(source_redirects)
        | "ParseSourceRedirects" >> beam.Map(lambda x: x.split("\t")))
    target_id_lookup = beam.pvalue.AsDict(
        root
        | "ReadTargetRedirects" >> beam.io.ReadFromText(target_redirects)
        | "ParseTargetRedirects" >> beam.Map(lambda x: x.split("\t")))
    mentions = (
        merged_articles
        | "ExtractMentions" >> beam.ParDo(
            DiffArticles(use_source_mentions),
            source_id_lookup=source_id_lookup,
            target_id_lookup=target_id_lookup)
        | "ReshardMentions" >> beam.Reshuffle())
    _ = (
        mentions
        | "MentionToJSONL" >> beam.Map(json.dumps)
        | "SerializeMentions" >> beam.io.WriteToText(
            os.path.join(output_dir, "mentions.jsonl")))
    mentions = (
        mentions
        | "RekeyMentions" >> beam.FlatMap(rekey_mentions)
        | "ReshardMentions(II)" >> beam.Reshuffle())
    merged_articles = (
        merged_articles
        | "FilterAndRe-Key" >> beam.FlatMap(
            filter_rekey,
            source_id_lookup=source_id_lookup,
            target_id_lookup=target_id_lookup)
        | "ReshardArticles" >> beam.Reshuffle())
    _ = ({
        "merged_articles": merged_articles,
        "mentions": mentions,
    }
         | "MergeArticlesAndMentions" >> beam.CoGroupByKey()
         | "ReshardMerged" >> beam.Reshuffle()
         | "FilterAndAnnotateMentions" >> beam.FlatMap(
             annotate_mentions, third_party=third_party)
         | "ArticlePairToJSONL" >> beam.Map(json.dumps)
         | "SerializeArticlePairs" >> beam.io.WriteToText(
             os.path.join(output_dir, "article_pairs.jsonl")))

  return pipeline


## Filter for Generation #######################################################


class FilterForGeneration(beam.DoFn):
  """Filters ArticlePair protos to make a lightweight generation dataset."""

  def __init__(
      self,
      vocab_model_file,
      max_article_length=512,
      excessive_length_strategy="truncate",
      max_mention_length=256,
      max_mentions=256,
      excessive_mention_strategy="truncate",
      use_source_mentions=True,
      include_new_articles=False,
  ):
    self.vocab_model_file = vocab_model_file
    self.max_article_length = max_article_length
    self.excessive_length_strategy = excessive_length_strategy
    self.max_mention_length = max_mention_length
    self.max_mentions = max_mentions
    self.excessive_mention_strategy = excessive_mention_strategy
    self.use_source_mentions = use_source_mentions
    self.include_new_articles = include_new_articles

  def setup(self):
    self.tokenizer = tft.SentencepieceTokenizer(
        model=tf.io.gfile.GFile(self.vocab_model_file, "rb").read())

  def enforce_article_length(self, article):
    """Subroutine for enforcing article length."""
    tokens = self.tokenizer.tokenize(article["text"])
    if tokens.shape[0] > self.max_article_length:
      if self.excessive_length_strategy == "discard":
        beam.metrics.Metrics.counter(__name__, "article_discard").inc()
        return
      elif self.excessive_length_strategy == "truncate":
        beam.metrics.Metrics.counter(__name__, "article_truncate").inc()
        article["text"] = self.truncate_text(article["text"])
        article["entities"] = [
            x for x in article["entities"] if x["end"] < len(article["text"])
        ]
        article["added_entities"] = [
            x for x in article["added_entities"]
            if x["end"] < len(article["text"])
        ]
    return article

  def truncate_text(self, text):
    """Truncates text to last complete sentence that fits in window."""
    # Break into sentences
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [self.tokenizer.tokenize(x) for x in sentences]
    lengths = [x.shape[0] for x in tokenized_sentences]
    cutoff_index = 0
    cumulative_length = 0
    for length in lengths:
      cumulative_length += length
      if cumulative_length > self.max_article_length:
        break
      cutoff_index += 1
    return " ".join(sentences[:cutoff_index])

  def enforce_mention_constraints(self, annotated_mentions):

    def _filter_fn(x):
      if not (self.use_source_mentions or x["mention"]["is_update"]):
        return False
      tokens = self.tokenizer.tokenize(x["mention"]["text"])
      if tokens.shape[0] > self.max_mention_length:
        return False
      return True

    new_annotated_mentions = list(filter(_filter_fn, annotated_mentions))

    if len(new_annotated_mentions) > self.max_mentions:
      if self.excessive_mention_strategy == "discard":
        return
      elif self.excessive_mention_strategy == "truncate":
        beam.metrics.Metrics.counter(__name__, "truncated_mentions").inc()
        # Prioritize retaining positive evidence and updated mentions
        new_annotated_mentions.sort(
            key=lambda x: (x["label"], x["mention"]["is_update"]), reverse=True)
        new_annotated_mentions = new_annotated_mentions[:self.max_mentions]

    return new_annotated_mentions

  def process(self, element):
    # Skip over unwanted data.
    if not element["updated"]:
      beam.metrics.Metrics.counter(__name__, "not_updated").inc()
      return

    # Potentially skip over new articles.
    if not element["source_article"]["text"] and not self.include_new_articles:
      beam.metrics.Metrics.counter(__name__, "filtered_new_articles").inc()
      return

    # Enforce length constraints on source and target article.
    element["source_article"] = self.enforce_article_length(
        element["source_article"])
    element["target_article"] = self.enforce_article_length(
        element["target_article"])
    if not element["source_article"] or not element["target_article"]:
      beam.metrics.Metrics.counter(__name__, "discarded_due_to_length").inc()
      return
    if element["source_article"]["text"] == element["target_article"]["text"]:
      beam.metrics.Metrics.counter(__name__, "same_after_truncation").inc()
      return

    # Filter annotated mentions.
    element["annotated_mentions"] = self.enforce_mention_constraints(
        element["annotated_mentions"])
    if not element["annotated_mentions"]:
      beam.metrics.Metrics.counter(__name__, "discarded_due_to_mentions").inc()
      return
    if not any(x["label"] for x in element["annotated_mentions"]):
      beam.metrics.Metrics.counter(__name__,
                                   "discarded_due_to_no_evidence").inc()
      return

    beam.metrics.Metrics.counter(__name__, "retained").inc()

    yield element


def filter_for_generation_pipeline(input_pattern,
                                   output_pattern,
                                   vocab_model_file,
                                   max_article_length=512,
                                   excessive_length_strategy="truncate",
                                   max_mention_length=256,
                                   max_mentions=256,
                                   excessive_mention_strategy="truncate",
                                   use_source_mentions=True,
                                   include_new_articles=False,
                                   dry_run=False):
  """Returns filter for generation beam pipeline."""

  def pipeline(root):
    filtered = (
        root
        | "Read" >> beam.io.ReadFromText(input_pattern)
        | "ParseJSONL" >> beam.Map(json.loads)
        | "FilterForGeneration" >> beam.ParDo(
            FilterForGeneration(vocab_model_file, max_article_length,
                                excessive_length_strategy, max_mention_length,
                                max_mentions, excessive_mention_strategy,
                                use_source_mentions, include_new_articles)))
    if not dry_run:
      _ = (
          filtered
          | "ToJSONL" >> beam.Map(json.dumps)
          | "Write" >> beam.io.WriteToText(output_pattern))

  return pipeline


## To TFExamples ###############################################################


class ToSeq2SeqInput(beam.DoFn):
  """Converts an article pair to an input for a Seq2Seq model."""

  def __init__(
      self,
      vocab_model_file,
      task,
      delimiter_type,
      include_source=False,
      include_evidence=True,
      include_distractors=True,
      evidence_marker_type=EvidenceMarkerType.empty,
      max_input_length=1024,
      filter_no_diff=True,
  ):
    self.vocab_model_file = vocab_model_file
    self.task = task
    self.delimiter_type = delimiter_type
    self.include_source = include_source
    self.include_evidence = include_evidence
    self.include_distractors = include_distractors
    self.evidence_marker_type = evidence_marker_type
    self.max_input_length = max_input_length
    self.filter_no_diff = filter_no_diff

  def setup(self):
    self.tokenizer = tft.SentencepieceTokenizer(
        model=tf.io.gfile.GFile(self.vocab_model_file, "rb").read())
    self.sentence_tokenizer = nltk.load(SENTENCE_TOKENIZER_PATH)
    self.delimiter_range_pair = rendering_utils.get_default_delimiter_range_pair(
        task=self.task,
        delimiter_type=self.delimiter_type,
    )

  def process(self, element):
    builder = rendering_utils.Seq2SeqInputBuilder(
        task=self.task,
        delimiter_range_pair=self.delimiter_range_pair,
        tokenizer=self.tokenizer,
        sentence_tokenizer=self.sentence_tokenizer,
        include_source=self.include_source,
        include_evidence=self.include_evidence,
        include_distractors=self.include_distractors,
        evidence_marker_type=self.evidence_marker_type,
        max_input_length=self.max_input_length,
        filter_no_diff=self.filter_no_diff,
    )
    yield from builder(element)


class GetLengths(beam.DoFn):
  """Gets the lengths of text inputs."""

  def __init__(self, vocab_model_file):
    super().__init__()
    self.vocab_model_file = vocab_model_file

  def setup(self):
    self.tokenizer = tft.SentencepieceTokenizer(
        model=tf.io.gfile.GFile(self.vocab_model_file, "rb").read())

  def process(self, element):

    def length(x):
      tokens = self.tokenizer.tokenize(x)
      length = tokens.shape[0]
      return length

    yield {k: [length(v)] for k, v in element.items() if isinstance(v, str)}


def combine_lengths(elements):
  """Aggregates lengths from multiple PCollections."""
  out = {"inputs": [], "targets": []}
  for element in elements:
    out["inputs"].extend(element["inputs"])
    out["targets"].extend(element["targets"])
  return out


def plot_histogram(element, path):
  """Plots a length histogram."""
  _, ax = plt.subplots(nrows=2, ncols=1)
  bins = np.r_[np.linspace(0, 8192, num=512), np.inf]
  ax[0].hist(element["inputs"], bins=bins, density=True, cumulative=True)
  ax[0].set_xlim([0, 8192])
  ax[0].set_title("Inputs")
  ax[1].hist(element["targets"], bins=bins, density=True, cumulative=True)
  ax[1].set_xlim([0, 8192])
  ax[1].set_title("Targets")

  plt.tight_layout()

  with tf.io.gfile.GFile(path, "wb") as f:
    plt.savefig(f, format="png", dpi=300)


def to_tfrecords_pipeline(
    input_pattern,
    output_pattern,
    vocab_model_file,
    task,
    delimiter_type,
    include_source = False,
    include_evidence = True,
    include_distractors = True,
    evidence_marker_type = EvidenceMarkerType.empty,
    max_input_length = 1024,
    filter_no_diff = True,
    plot_lengths = False,
):
  """Beam pipeline for converting article pairs to tfrecords."""

  def pipeline(root):
    seq2seq_inputs = (
        root
        | "Read" >> beam.io.ReadFromText(input_pattern)
        | "ParseJSONL" >> beam.Map(json.loads)
        | "ToSeq2SeqInput" >> beam.ParDo(
            ToSeq2SeqInput(
                vocab_model_file=vocab_model_file,
                task=task,
                delimiter_type=delimiter_type,
                include_source=include_source,
                include_evidence=include_evidence,
                include_distractors=include_distractors,
                evidence_marker_type=evidence_marker_type,
                max_input_length=max_input_length,
                filter_no_diff=filter_no_diff)))
    _ = (
        seq2seq_inputs
        | "ReShuffle" >> beam.Reshuffle()
        | "ToExample" >> beam.Map(tf_utils.to_example)
        | "Write" >> beam.io.tfrecordio.WriteToTFRecord(
            output_pattern,
            coder=beam.coders.ProtoCoder(tf.train.Example),
            num_shards=10))
    if plot_lengths:
      _ = (
          seq2seq_inputs
          | "GetLengths" >> beam.ParDo(GetLengths(vocab_model_file))
          | "Combine" >> beam.CombineGlobally(combine_lengths)
          | "PlotHistogram" >> beam.Map(
              plot_histogram, path=output_pattern + ".hist.png"))

  return pipeline
