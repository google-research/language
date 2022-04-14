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
"""Config for fever data."""

import dataclasses


@dataclasses.dataclass
class Config:
  """Global configuration object for fever code, create an instance of this.

  This is used across multiple CLI scripts so that scripts may reference their
  values through a single consistent (typed) class, instead of untyped/unchecked
  FLAGS.
  """

  bert_base_uncased_model: Text = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1'
  bert_large_uncased_model: Text = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1'


  train_claim_ids_npy: Text = 'train_claim_ids.npy'
  train_embeddings_npy: Text = 'train_embeddings.npy'
  val_claim_ids_npy: Text = 'val_claim_ids.npy'
  val_embeddings_npy: Text = 'val_embeddings.npy'
  max_claim_tokens: int = 20
  max_evidence_tokens: int = 60
  max_evidence: int = 5
  n_similar_negatives: int = 5
  n_background_negatives: int = 5
  n_inference_candidates: int = 200
  n_inference_documents: int = 10
  include_not_enough_info: bool = True
  max_inference_sentence_id: int = 30
  claim_loss_weight: float = 0.5
  title_in_scoring: bool = True
