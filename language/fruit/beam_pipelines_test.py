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
"""Tests for beam_pipelines.

Currently just running smoke tests on well-formatted data.
"""
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from language.fruit import beam_pipelines
from language.fruit import rendering_utils

TESTDATA_DIR = "language/fruit/testdata/"
VOCAB_FILE = ("gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model")


class RedirectTablePipelineTest(absltest.TestCase):

  def test_io(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      pipeline = beam_pipelines.redirect_table_pipeline(
          input_jsonl=os.path.join(TESTDATA_DIR, "test_source_articles.jsonl"),
          output_tsv=os.path.join(tmpdir, "redirects.tsv"))
      with beam.Pipeline() as p:
        pipeline(p)


class ProcessSnapshotPipelineTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          "keep_tables": True,
          "third_party": True,
          "truncate": True,
          "use_source_mentions": True
      },
      {
          "keep_tables": False,
          "third_party": False,
          "truncate": False,
          "use_source_mentions": False
      },
  )
  def test_io(
      self,
      keep_tables,
      third_party,
      truncate,
      use_source_mentions,
  ):
    with tempfile.TemporaryDirectory() as tmpdir:
      pipeline = beam_pipelines.process_snapshot_pipeline(
          source_jsonl=os.path.join(TESTDATA_DIR, "test_source_articles.jsonl"),
          target_jsonl=os.path.join(TESTDATA_DIR, "test_target_articles.jsonl"),
          output_dir=tmpdir,
          source_redirects=os.path.join(TESTDATA_DIR, "test_redirects.tsv"),
          target_redirects=os.path.join(TESTDATA_DIR, "test_redirects.tsv"),
          keep_tables=keep_tables,
          third_party=third_party,
          truncate=truncate,
          use_source_mentions=use_source_mentions,
      )
      with beam.Pipeline() as p:
        pipeline(p)


class FilterForGenerationTest(absltest.TestCase):

  def test_io(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      pipeline = beam_pipelines.filter_for_generation_pipeline(
          input_pattern=os.path.join(TESTDATA_DIR, "test_article_pairs.jsonl"),
          output_pattern=os.path.join(tmpdir, "output.jsonl"),
          vocab_model_file=VOCAB_FILE,
      )
      with beam.Pipeline() as p:
        pipeline(p)


class ToTFRecordsTest(absltest.TestCase):

  def test_io(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      pipeline = beam_pipelines.to_tfrecords_pipeline(
          input_pattern=os.path.join(TESTDATA_DIR, "test_article_pairs.jsonl"),
          output_pattern=os.path.join(tmpdir, "output.tfrecord"),
          vocab_model_file=VOCAB_FILE,
          task=rendering_utils.Task.diff,
          delimiter_type=rendering_utils.DelimiterType.text,
          plot_lengths=True,
      )
      with beam.Pipeline() as p:
        pipeline(p)


if __name__ == "__main__":
  absltest.main()
