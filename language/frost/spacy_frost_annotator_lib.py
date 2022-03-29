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
"""Spacy Frost Processor."""

import os
import spacy


# FROST Constants
ENTITYCHAIN_START_TOKEN = "[CONTENT]"
SUMMARY_START_TOKEN = "[SUMMARY]"
ENTITY_SEPARATOR = " | "
ENTITY_SENTENCE_SEPARATOR = " ||| "

# Prepare Spacy processor
SPACY_MODEL_OR_PATH = "en_core_web_sm"
SPACY_PROCESSOR = spacy.load(SPACY_MODEL_OR_PATH)


def get_spacy_frost_processor_fn():
  """Gets Spacy Frost processor."""

  def _annotate_text_with_entityplans(text):
    entity_plans = []
    for text_sent in SPACY_PROCESSOR(text.replace("\n", " ")).sents:
      entity_plans.append(
          ENTITY_SEPARATOR.join(
              [entity.text for entity in SPACY_PROCESSOR(text_sent.text).ents]))
    text_with_entityplans = (
        ENTITYCHAIN_START_TOKEN + " " +
        ENTITY_SENTENCE_SEPARATOR.join(entity_plans) + " " +
        SUMMARY_START_TOKEN + " " + text)
    return text_with_entityplans

  return _annotate_text_with_entityplans
