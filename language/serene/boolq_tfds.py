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
"""TF Dataset for BoolQ in same format as Fever TFDS."""
import json

from language.serene import constants
from language.serene import util
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds


class BoolQClaims(tfds.core.GeneratorBasedBuilder):
  """TFDS for treating boolq as fact verification."""

  VERSION = tfds.core.Version('0.1.0')

  def __init__(self,
               *,
               boolq_train_path,
               boolq_dev_path,
               data_dir=None,
               config=None,
               version=None):
    super().__init__(data_dir=data_dir, config=config, version=version)
    self._boolq_train_path = boolq_train_path
    self._boolq_dev_path = boolq_dev_path

  def _generate_examples(self, boolq_filepath, fold):
    boolq_claims = util.read_jsonlines(boolq_filepath)
    for idx, claim in enumerate(boolq_claims):
      example_id = f'{fold}-{idx}'
      example = {
          'example_id':
              example_id,
          'claim_text':
              claim['question'],
          'evidence_text':
              claim['passage'],
          'wikipedia_url':
              claim['title'],
          'sentence_id':
              '0',
          # This is effectively gold evidence
          'evidence_label':
              constants.MATCHING,
          'claim_label':
              constants.SUPPORTS if claim['answer'] else constants.REFUTES,
          'metadata':
              json.dumps({})
      }
      yield example_id, example

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            'example_id':
                tf.string,
            'metadata':
                tf.string,
            'claim_text':
                tfds.features.Text(),
            'evidence_text':
                tfds.features.Text(),
            'wikipedia_url':
                tfds.features.Text(),
            'sentence_id':
                tfds.features.Text(),
            'evidence_label':
                tfds.features.ClassLabel(
                    names=constants.EVIDENCE_MATCHING_CLASSES),
            'claim_label':
                tfds.features.ClassLabel(names=constants.FEVER_CLASSES)
        }),
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                'boolq_filepath': self._boolq_train_path,
                'fold': 'train',
            },
            num_shards=100,
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                'boolq_filepath': self._boolq_dev_path,
                'fold': 'dev',
            },
        )
    ]
