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
"""ORQA ops."""
import os
import tensorflow.compat.v1 as tf

try:
  orqa_ops
except NameError:
  orqa_ops = tf.load_op_library(
      os.path.join(os.path.dirname(os.path.abspath(__file__)), "orqa_ops.so"))

has_answer = orqa_ops.has_answer
reader_inputs = orqa_ops.reader_inputs
