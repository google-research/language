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
"""Tests for top_utils."""

from absl.testing import absltest
from language.casper.utils import top_utils


class TopUtilsTest(absltest.TestCase):

  def test_round_trip(self):
    serialized = ("[IN:GET_CALL_TIME [SL:CONTACT "
                  "[IN:GET_CONTACT [SL:TYPE_RELATION Mum ] ] ] "
                  "[SL:DATE_TIME yesterday evening ] ]")
    lf = top_utils.deserialize_top(serialized)
    roundtrip_serialized = lf.serialize()
    self.assertEqual(serialized, roundtrip_serialized)

  def test_invalid(self):
    # Extra close-bracket
    serialized = "[IN:GET_CALL_TIME [SL:DATE_TIME now ] ] ]"
    lf = top_utils.deserialize_top(serialized)
    self.assertIsNone(lf)
    # Missing a close-bracket
    serialized = "[IN:GET_CALL_TIME [SL:DATE_TIME now ]"
    lf = top_utils.deserialize_top(serialized)
    self.assertIsNone(lf)

  def test_get_frame(self):
    serialized = ("[IN:GET_CALL_TIME [SL:CONTACT "
                  "[IN:GET_CONTACT [SL:TYPE_RELATION Mum ] ] ] "
                  "[SL:DATE_TIME yesterday evening ] ]")
    expected_frame = ("IN:GET_CALL_TIME-"
                      "SL:CONTACT.IN:GET_CONTACT.SL:TYPE_RELATION-"
                      "SL:DATE_TIME")
    actual_frame = top_utils.get_frame_top(serialized)
    self.assertEqual(expected_frame, actual_frame)

  def test_format(self):
    serialized = ("[IN:GET_CALL_TIME [SL:CONTACT "
                  "[IN:GET_CONTACT [SL:TYPE_RELATION Mum ] ] ] "
                  "[SL:DATE_TIME yesterday evening ] ]")
    expected_formatted = ("[IN get call time = [SL contact = "
                          "[IN get contact = [SL type relation = Mum]]] "
                          "[SL date time = yesterday evening]]")
    formatted = top_utils.format_serialized(serialized)
    self.assertEqual(expected_formatted, formatted)
    deformatted = top_utils.deformat_serialized(formatted)
    self.assertEqual(serialized, deformatted)
    # Try calling deserialize_top on the formatted string
    lf = top_utils.deserialize_top(formatted)
    roundtrip_serialized = lf.serialize()
    self.assertEqual(serialized, roundtrip_serialized)


if __name__ == "__main__":
  absltest.main()
