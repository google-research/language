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
"""Contains constants required for the model."""

# TODO(alanesuhr): These are used in convert_to_tf_examples.py.
#  Use these constants instead of strings there.

# These constants define the keys used for the TFRecords.
COPIABLE_INPUT_KEY = 'copiable_input'
ALIGNED_KEY = 'utterance_schema_alignment'
SEGMENT_ID_KEY = 'segment_ids'
FOREIGN_KEY_KEY = 'indicates_foreign_key'
SOURCE_WORDPIECES_KEY = 'source_wordpieces'
SOURCE_LEN_KEY = 'source_len'
LANGUAGE_KEY = 'language'
REGION_KEY = 'region'
TAG_KEY = 'tag'

OUTPUT_TYPE_KEY = 'type'
WEIGHT_KEY = 'weight'
TARGET_ACTION_TYPES_KEY = 'target_action_types'
TARGET_ACTION_IDS_KEY = 'target_action_ids'
TARGET_LEN_KEY = 'target_len'

SCORES_KEY = 'scores'

# Symbol IDs.
TARGET_START_SYMBOL_ID = 2
TARGET_END_SYMBOL_ID = 1
PAD_SYMBOL_ID = 0

GENERATE_ACTION = 1
COPY_ACTION = 2

NUM_RESERVED_OUTPUT_SYMBOLS = 3

PREDICTED_ACTION_TYPES = 'predicted_action_types'
PREDICTED_ACTION_IDS = 'predicted_action_ids'
