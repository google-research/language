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
"""Gather default values in central location."""

import flax.linen as nn

from language.mentionmemory.utils import initializers

kernel_init = initializers.truncated_normal(stddev=0.02)
bias_init = nn.initializers.zeros
layer_norm_epsilon = 1e-12

ENTITY_START_TOKEN = 1
ENTITY_END_TOKEN = 2

# Value typically used to prevent division by zero.
SMALL_NUMBER = 1e-8
