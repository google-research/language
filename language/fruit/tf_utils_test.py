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
from absl.testing import absltest
from language.fruit import tf_utils


class TfUtilsTest(absltest.TestCase):

  def test_maybe_decode(self):
    x = b"A test string"
    self.assertIsInstance(tf_utils.maybe_decode(x), str)

  def test_maybe_encode(self):
    x = "A test string"
    self.assertIsInstance(tf_utils.maybe_encode(x), bytes)

  def test_to_example(self):
    dictionary = {
        "inputs": "foo",
        "targets": "bar",
        "generatable_surfaces": {
            "baz": ["buzz"]
        },
        "id": 42
    }
    example = tf_utils.to_example(dictionary)
    self.assertEqual(
        example.features.feature["inputs"].bytes_list.value,
        [b"foo"],
    )
    self.assertEqual(
        example.features.feature["targets"].bytes_list.value,
        [b"bar"],
    )
    self.assertEqual(
        example.features.feature["generatable_surfaces"].bytes_list.value,
        [b'{"baz": ["buzz"]}'],
    )
    self.assertEqual(
        example.features.feature["id"].int64_list.value,
        [42],
    )

  def test_postprocess_example(self):
    example = {
        "inputs": b"foo",
        "targets": b"bar",
        "generatable_surfaces": b'{"baz": ["buzz"]}',
        "id": 42
    }
    expected = {
        "inputs": "foo",
        "targets": "bar",
        "generatable_surfaces": {
            "baz": ["buzz"]
        },
        "id": 42
    }
    observed = tf_utils.postprocess_example(example)
    self.assertDictEqual(observed, expected)


if __name__ == "__main__":
  absltest.main()
