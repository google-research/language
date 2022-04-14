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
"""Script to preprocess wiki.txt.

Extracts documents (sentences) with their titles, tokenizes, and links entities
from the KB.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import json
import re

from absl import app
from absl import flags

import nltk
from tqdm import tqdm

MIN_ENTITY_LEN = 3
ARTICLES = frozenset([u"the", u"a", u"an"])
REMOVE_ARTICLES = False

document_id = 0

FLAGS = flags.FLAGS

flags.DEFINE_string("wiki_file", None, "Wikimovie corpus.")

flags.DEFINE_string("entity_file", None, "List of entities from the KB.")

flags.DEFINE_string("output_file", None, "Processed wiki corpus.")


def _link_entities(line, overall_regex, pattern_to_entities):
  """Return list of matching entities in text."""
  matches = []

  def _add_entities(match, text, entities):
    for entity in entities:
      m_st = match.start()
      m_en = match.start() + len(match.group(1))
      matches.append({
          "name": entity[0],
          "kb_id": entity[1],
          "text": text[m_st:m_en],
          "start": m_st,
      })

  for match in re.finditer(overall_regex, line):
    ent = match.group(1).lower()
    if ent not in pattern_to_entities:
      print("Did not find %s" % ent)
      continue
    _add_entities(match, line, pattern_to_entities[ent])
  return matches


def _preprocess_line(line, overall_regex, pattern_to_entities):
  """Tokenize, link and lower-case."""
  text = u" ".join(nltk.word_tokenize(line))
  entities = _link_entities(text, overall_regex, pattern_to_entities)
  return {"context": text.lower(), "mentions": entities}


def process_document(text, title, overall_regex, pattern_to_entities):
  """Process a single line from wiki.txt."""
  global document_id
  doc_obj = _preprocess_line(text, overall_regex, pattern_to_entities)
  title_obj = _preprocess_line(title, overall_regex, pattern_to_entities)
  doc_obj["id"] = document_id
  doc_obj["title"] = title_obj
  # Prepend title to the context.
  doc_obj["context"] = doc_obj["title"]["context"] + " . " + doc_obj["context"]
  extra = len(doc_obj["title"]["context"]) + 3
  for mention in doc_obj["mentions"]:
    mention["start"] += extra
  doc_obj["mentions"] = doc_obj["title"]["mentions"] + doc_obj["mentions"]
  document_id += 1
  return doc_obj


def main(_):
  # read entity vocab
  print("Reading entities and building regexes...")
  entity_vocab = {
      ee: ei
      for ei, ee in enumerate(io.open(FLAGS.entity_file).read().splitlines())
  }
  pattern_to_entities = {}
  all_patterns = []
  for entity, eid in tqdm(entity_vocab.items()):
    if len(entity) < MIN_ENTITY_LEN:
      continue
    if re.match(r"\w", entity[-1]) is None:
      entity_clean = entity[:-1]
    else:
      entity_clean = entity
    entity_toks = nltk.word_tokenize(entity_clean)
    if (REMOVE_ARTICLES and len(entity_toks) > 1 and
        entity_toks[0].lower() in ARTICLES):
      entity_clean = u" ".join(tok.lower() for tok in entity_toks[1:])
    else:
      entity_clean = u" ".join(tok.lower() for tok in entity_toks)
    pattern = r"(?:(?<=\s)|(?<=^)){}(?:(?=\s)|(?=$))".format(
        re.escape(entity_clean))
    all_patterns.append(pattern)
    if entity_clean not in pattern_to_entities:
      pattern_to_entities[entity_clean] = []
    pattern_to_entities[entity_clean].append((entity, eid))
  all_patterns = sorted(all_patterns, key=lambda x: -len(x))
  overall_pattern = u"(?=({}))".format(u"|".join(all_patterns))
  overall_regex = re.compile(overall_pattern, flags=re.UNICODE | re.IGNORECASE)

  # process wiki docs
  print("Processing wiki sentences...")
  with io.open(FLAGS.wiki_file) as f, open(FLAGS.output_file, "w") as fo:
    context = []
    title = None
    for _, line in tqdm(enumerate(f)):
      if title is None:
        title = line.strip().split(None, 1)[1]
      elif line == u"\n":
        fo.write(
            json.dumps(
                process_document(" ".join(context), title, overall_regex,
                                 pattern_to_entities)) + u"\n")
        title = None
        context = []
      else:
        context.append(line.strip().split(None, 1)[1])


if __name__ == "__main__":
  app.run(main)
