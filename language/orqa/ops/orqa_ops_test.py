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
"""Tests for ORQA ops."""
from language.orqa import ops as orqa_ops
import tensorflow.compat.v1 as tf


class OrqaOpsTest(tf.test.TestCase):

  def test_reader_inputs(self):
    concat_inputs = orqa_ops.reader_inputs(
        question_token_ids=[0, 1],
        block_token_ids=[[2, 3, 4], [5, 6, 0]],
        block_lengths=[3, 2],
        block_token_map=[[1, 2, 5], [1, 3, 4]],
        answer_token_ids=[[3, 4], [7, 0]],
        answer_lengths=[2, 1],
        cls_token_id=10,
        sep_token_id=11,
        max_sequence_len=10)

    self.assertAllEqual(
        concat_inputs.token_ids.numpy(),
        [[10, 0, 1, 11, 2, 3, 4, 11, 0, 0], [10, 0, 1, 11, 5, 6, 11, 0, 0, 0]])
    self.assertAllEqual(
        concat_inputs.mask.numpy(),
        [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
    self.assertAllEqual(
        concat_inputs.segment_ids.numpy(),
        [[0, 0, 0, 0, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]])
    self.assertAllEqual(
        concat_inputs.block_mask.numpy(),
        [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]])
    self.assertAllEqual(concat_inputs.token_map.numpy(),
                        [[-1, -1, -1, -1, 1, 2, 5, -1, -1, -1],
                         [-1, -1, -1, -1, 1, 3, -1, -1, -1, -1]])
    self.assertAllEqual(concat_inputs.gold_starts.numpy(), [[5], [-1]])
    self.assertAllEqual(concat_inputs.gold_ends.numpy(), [[6], [-1]])

  def test_has_answer(self):
    result = orqa_ops.has_answer(blocks=["abcdefg", "hijklmn"], answers=["hij"])
    self.assertAllEqual(result.numpy(), [False, True])

if __name__ == "__main__":
  tf.test.main()
