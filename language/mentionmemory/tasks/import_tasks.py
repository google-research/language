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
"""Import tasks so that decorated tasks are added to registry."""

# pylint: disable=unused-import
# Block of imports needed to allow different tasks to get registered
# with task registry.
from language.mentionmemory.tasks import eae_task
from language.mentionmemory.tasks import embedding_based_entity_qa_task
from language.mentionmemory.tasks import example_task
from language.mentionmemory.tasks import mauto_task
from language.mentionmemory.tasks import mention_based_entity_qa_task
from language.mentionmemory.tasks import mention_memory_task
from language.mentionmemory.tasks import readtwice_task
from language.mentionmemory.tasks import relation_classifier_task
from language.mentionmemory.tasks import text_classifier
from language.mentionmemory.tasks import ultra_fine_entity_typing_task
