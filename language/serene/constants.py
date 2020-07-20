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
"""Constants for fever data."""


VERIFIABLE = 'VERIFIABLE'
NOT_VERIFIABLE = 'NOT VERIFIABLE'

# Classes used for claim classification and labeling which evidence
# support/refute the claim
NOT_ENOUGH_INFO = 'NOT ENOUGH INFO'
REFUTES = 'REFUTES'
SUPPORTS = 'SUPPORTS'
FEVER_CLASSES = [REFUTES, SUPPORTS, NOT_ENOUGH_INFO]

# Classes used for scoring candidate evidence relevance
MATCHING = 'MATCHING'
NOT_MATCHING = 'NOT_MATCHING'
EVIDENCE_MATCHING_CLASSES = [NOT_MATCHING, MATCHING]

UKP_WIKI = 'ukp_wiki'
UKP_PRED = 'ukp_pred'
UKP_TYPES = [UKP_PRED, UKP_WIKI]
DRQA = 'drqa'
LUCENE = 'lucene'
DOC_TYPES = [UKP_WIKI, UKP_PRED, DRQA, LUCENE]
