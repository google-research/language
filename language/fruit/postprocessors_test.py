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
from language.fruit import postprocessors
from seqio import test_utils


class PostProcessWikiDiffTest(absltest.TestCase):

  def test_postprocess_wikidiff(self):
    vocabulary = test_utils.bertwordpiece_vocab()
    output = b"output"
    example = {
        "inputs_pretokenized": "this is a test blah",
        "inputs": [106, 105, 104, 107],
        "generatable_surfaces": '{"e0": ["e0"]}',
    }

    # NOTE: Normalize fn's in `rendering_utils` are already tested, so we don't
    # need to check they work again here.
    def _mock_normalize_fn(x, y):
      return x, y

    expected = {
        "inputs": "this is a test",
        "targets": "output",
        "normalized_inputs": "this is a test",
        "normalized_targets": "output",
        "generatable_surfaces": {
            "e0": ["e0"]
        },
    }
    observed = postprocessors.postprocess_wikidiff(
        output,
        vocabulary,
        example=example,
        normalize_fn=_mock_normalize_fn,
    )
    self.assertDictEqual(expected, observed)


if __name__ == "__main__":
  absltest.main()
