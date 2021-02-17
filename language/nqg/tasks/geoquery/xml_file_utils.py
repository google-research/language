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
"""Utilities for reading XML datafile for GeoQuery."""

from xml.etree import ElementTree

from tensorflow.io import gfile


def process_utterance(utterance):
  """Lowercase and remove punctuation."""
  return utterance.lower().rstrip("?").rstrip(".").rstrip().replace(" '", "")


def process_funql(funql):
  """Remove quotes and unnecessary spaces."""
  funql = funql.replace("'", "")
  funql = funql.replace(",  ", ",")
  funql = funql.replace(", ", ",")
  funql = funql.replace(" ,", ",")
  return funql


def load_xml_tree(corpus):
  with gfile.GFile(corpus, "r") as xml_file:
    return ElementTree.fromstring(xml_file.read())


def get_utterance(example_root):
  for utterance in example_root.findall("nl"):
    if utterance.attrib["lang"] == "en":
      return process_utterance(utterance.text.strip())
  raise ValueError("Could not find utterance.")


def get_funql(example_root):
  for mrl in example_root.findall("mrl"):
    if mrl.attrib["lang"] == "geo-funql":
      return process_funql(mrl.text.strip())
  raise ValueError("Could not find funql.")


def read_examples(corpus):
  examples = []
  root = load_xml_tree(corpus)
  for example_root in root:
    examples.append((get_utterance(example_root), get_funql(example_root)))
  return examples
