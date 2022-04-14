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
"""TFDS for only claims."""
import json


from language.serene import constants
from language.serene import util
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds


class ClaimDataset(tfds.core.GeneratorBasedBuilder):
  """Claim only datasets for fever, useful for embedding only claims."""

  VERSION = tfds.core.Version("0.1.0")

  def __init__(
      self, *,
      fever_train_path = None,
      fever_dev_path = None,
      data_dir = None,
      config=None):
    super().__init__(data_dir=data_dir, config=config)
    self._fever_train_path = fever_train_path
    self._fever_dev_path = fever_dev_path

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            "example_id":
                tf.string,
            "metadata":
                tf.string,
            "claim_text":
                tfds.features.Text(),
            "evidence_text":
                tfds.features.Text(),
            "wikipedia_url":
                tfds.features.Text(),
            "sentence_id":
                tfds.features.Text(),
            "scrape_type":
                tfds.features.Text(),
            "evidence_label":
                tfds.features.ClassLabel(
                    names=constants.EVIDENCE_MATCHING_CLASSES),
            "claim_label":
                tfds.features.ClassLabel(names=constants.FEVER_CLASSES)
        }))

  def _split_generators(self, dl_manager):
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={"filepath": self._fever_train_path}
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"filepath": self._fever_dev_path}
        )
    ]

  def _generate_examples(self, filepath, **kwargs):
    fever_claims = util.read_jsonlines(filepath)
    for claim in fever_claims:
      claim_id = claim["id"]
      claim_text = claim["claim"]
      claim_label = claim["label"]
      example_id = f"{claim_id}"
      yield claim_id, {
          "example_id": example_id,
          "claim_text": claim_text,
          "evidence_text": "",
          "wikipedia_url": "",
          # Ordinarily, this would (possibly) be concatenated to the evidence
          # but since this is claim only, I'm using a null integer value
          "sentence_id": "-1",
          # This label doesn't matter here since its claim only
          "evidence_label": constants.NOT_MATCHING,
          "claim_label": claim_label,
          "scrape_type": "",
          "metadata": json.dumps({
              "claim_id": claim_id,
          })
      }
