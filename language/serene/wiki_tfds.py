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
# Lint as: python3
"""Create a tensorflow dataset from every sentence in Wikipedia dump."""


import apache_beam as beam
from language.serene import fever_pb2
from language.serene import wiki_db
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class ExtractSentences(beam.DoFn):
  """Beam operation to extract sentences from wiki page."""

  def __init__(self, *, max_sentence_id, wiki_db_path):
    super().__init__()
    self._max_sentence_id = max_sentence_id
    self._wiki_db_path = wiki_db_path
    self._db: Optional[wiki_db.WikiDatabase] = None

  def setup(self):
    self._db = wiki_db.WikiDatabase.from_local(self._wiki_db_path)

  def process(self, wikipedia_url, *args, **kwargs):
    """Given a Wikipedia page, return an example for each sentence.

    Args:
      wikipedia_url: A Wikipedia url (title)
      *args: Unused, kept for API compat
      **kwargs: Unused, kept for API compat

    Yields:
      An example for Tensorflow Datasets containing page, sentence, and text
    """
    page: fever_pb2.WikipediaDump = self._db.get_page(wikipedia_url)  # pytype: disable=annotation-type-mismatch  # attribute-variable-annotations
    for sentence_id, sentence in page.sentences.items():
      if sentence_id <= self._max_sentence_id:
        key = f'{wikipedia_url}@@@{sentence_id}'
        yield (key, {
            'wikipedia_url': wikipedia_url,
            'sentence_id': sentence_id,
            'text': sentence.text,
            'text_wikipedia_url': wikipedia_url,
            'text_sentence_id': str(sentence_id),
        })


class WikipediaText(tfds.core.BeamBasedBuilder):
  """Tensorflow dataset for Wikipedia sentences."""

  VERSION = tfds.core.Version('0.1.0')

  def __init__(self,
               max_sentence_id = None,
               wiki_db_path = None,
               data_dir = None,
               config=None,
               version=None):
    super().__init__(data_dir=data_dir, config=config, version=version)
    self._max_sentence_id = max_sentence_id
    self._wiki_db_path = wiki_db_path

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'wikipedia_url': tf.string,
            'sentence_id': tf.int64,
            'text': tfds.features.Text(),
            'text_wikipedia_url': tfds.features.Text(),
            'text_sentence_id': tfds.features.Text(),
        }),
        supervised_keys=None)

  def _split_generators(self, dl_manager):
    return [tfds.core.SplitGenerator(name=tfds.Split.VALIDATION,)]

  def _build_pcollection(self, pipeline, **kwargs):
    db = wiki_db.WikiDatabase.from_local(self._wiki_db_path)
    wikipedia_urls = db.get_wikipedia_urls()
    return (pipeline
            | 'LoadPages' >> beam.Create(wikipedia_urls)
            | 'Repartition' >> beam.Reshuffle()
            | 'ExtractSentences' >> beam.ParDo(
                ExtractSentences(
                    max_sentence_id=self._max_sentence_id,
                    wiki_db_path=self._wiki_db_path)))
