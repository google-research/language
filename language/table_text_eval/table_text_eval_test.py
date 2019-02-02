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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from language.table_text_eval import table_text_eval
import tensorflow as tf

TEST_PREDS = [
    ["michael", "dahlquist", "(", "december", "22", ",", "1965", "--", "july",
     "14", ",", "2005", ")", "was", "a", "drummer", "in", "the", "california",
     "band", "grateful", "dead", "."],
    ["michael", "dahlquist", "(", "december", "22", ",", "1965", "--",
     "july", "14", ",", "2005", ")", "was", "a", "drummer", "."],
    ["michael", "dahlquist", "(", "december", "22", ",", "1965", "--", "july",
     "14", ",", "2005", ")", "was", "a", "drummer", "from", "seattle",
     ",", "washington", "."],
]

TEST_REF = [
    "michael", "dahlquist", "(", "december", "22", ",", "1965", "--", "july",
    "14", ",", "2005", ")", "was", "a", "drummer", "in", "the", "seattle",
    "band", "silkworm", "."
]

TEST_TABLE = [
    [["birth", "name"], ["michael", "dahlquist"]],
    [["born"], ["december", "22", ",", "1965"]],
    [["birth", "place"], ["seattle", ",", "washington"]],
    [["died"], ["july", "14", ",", "2005"]],
    [["death", "place"], ["skokie", ",", "illinois"]],
    [["genres"], ["male"]],
    [["occupation(s)"], ["drummer"]],
    [["instruments"], ["drums"]],
]

TEST_SCORES = [0.688, 0.746, 0.786]


class ParentV1Test(tf.test.TestCase):

  def test_parent(self):
    for pred, score in zip(TEST_PREDS, TEST_SCORES):
      print("prediction = %s" % " ".join(pred))
      _, _, parent_score, _ = table_text_eval.parent([pred], [[TEST_REF]],
                                                     [TEST_TABLE],
                                                     lambda_weight=0.9)
      print(parent_score)
      self.assertAlmostEqual(score, parent_score, places=3)


if __name__ == "__main__":
  tf.test.main()
