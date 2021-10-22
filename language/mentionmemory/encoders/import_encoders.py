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
"""Import encoders so that decorated encoders are added to registry."""

# pylint: disable=unused-import
from language.mentionmemory.encoders import bert_encoder
from language.mentionmemory.encoders import eae_encoder
from language.mentionmemory.encoders import mauto_encoder
from language.mentionmemory.encoders import mention_memory_encoder
from language.mentionmemory.encoders import readtwice_encoder

# pylint: enable=unused-import
