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
"""Common utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Attention types.
ATT_LUONG = "luong"
ATT_LUONG_SCALED = "luong_scaled"
ATT_BAHDANAU = "bahdanau"
ATT_BAHDANAU_NORM = "bahdanau_norm"
ATT_TYPES = (ATT_LUONG, ATT_LUONG_SCALED, ATT_BAHDANAU, ATT_BAHDANAU_NORM)

# Encoder types.
ENC_UNI = "uni"
ENC_BI = "bi"
ENC_GNMT = "gnmt"
ENC_TYPES = (ENC_UNI, ENC_BI, ENC_GNMT)

# Decoder types.
DEC_BASIC = "basic"
DEC_ATTENTIVE = "attentive"
DEC_TYPES = (DEC_BASIC, DEC_ATTENTIVE)


# Language model types.
LM_L2R = "left2right"
LM_TYPES = (LM_L2R,)
