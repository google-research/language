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
"""Tests for cogs_converter."""
from language.compgen.csl.tasks.cogs.tools import cogs_converter
import tensorflow as tf


class CogsConverterTest(tf.test.TestCase):

  def test_single_entity(self):
    # James investigated .
    lf = "investigate . agent ( x _ 1 , James )"
    expected = "investigate ( agent = James )"
    self.assertEqual(cogs_converter.cogs_lf_to_funcall(lf), expected)

    # A cat floated .
    lf = "cat ( x _ 1 ) AND float . theme ( x _ 2 , x _ 1 )"
    expected = "float ( theme = cat )"
    self.assertEqual(cogs_converter.cogs_lf_to_funcall(lf), expected)

    # The captain ate .
    lf = "* captain ( x _ 1 ) ; eat . agent ( x _ 2 , x _ 1 )"
    expected = "eat ( agent = * captain )"
    self.assertEqual(cogs_converter.cogs_lf_to_funcall(lf), expected)

  def test_flat_lfs(self):
    # The sailor dusted a boy .
    lf = ("* sailor ( x _ 1 ) ; dust . agent ( x _ 2 , x _ 1 ) "
          "AND dust . theme ( x _ 2 , x _ 4 ) AND boy ( x _ 4 )")
    expected = "dust ( agent = * sailor , theme = boy )"
    self.assertEqual(cogs_converter.cogs_lf_to_funcall(lf), expected)

    # Eleanor sold Evelyn the cake .
    lf = ("* cake ( x _ 4 ) ; sell . agent ( x _ 1 , Eleanor ) "
          "AND sell . recipient ( x _ 1 , Evelyn ) "
          "AND sell . theme ( x _ 1 , x _ 4 )")
    expected = "sell ( agent = Eleanor , recipient = Evelyn , theme = * cake )"
    self.assertEqual(cogs_converter.cogs_lf_to_funcall(lf), expected)

  def test_nested_lfs(self):
    # The girl needed to cook .
    lf = ("* girl ( x _ 1 ) ; need . agent ( x _ 2 , x _ 1 ) "
          "AND need . xcomp ( x _ 2 , x _ 4 ) "
          "AND cook . agent ( x _ 4 , x _ 1 )")
    expected = "need ( agent = * girl , xcomp = cook ( agent = * girl ) )"
    self.assertEqual(cogs_converter.cogs_lf_to_funcall(lf), expected)

    # The penguin dreamed that Emma wanted to paint .
    lf = ("* penguin ( x _ 1 ) ; dream . agent ( x _ 2 , x _ 1 ) "
          "AND dream . ccomp ( x _ 2 , x _ 5 ) "
          "AND want . agent ( x _ 5 , Emma ) "
          "AND want . xcomp ( x _ 5 , x _ 7 ) "
          "AND paint . agent ( x _ 7 , Emma )")
    expected = ("dream ( agent = * penguin , ccomp = want ( "
                "agent = Emma , xcomp = paint ( agent = Emma ) ) )")
    self.assertEqual(cogs_converter.cogs_lf_to_funcall(lf), expected)

    # The dog in a bakery in the bag sneezed .
    lf = ("* dog ( x _ 1 ) ; * bag ( x _ 7 ) ; "
          "dog . nmod . in ( x _ 1 , x _ 4 ) "
          "AND bakery ( x _ 4 ) AND bakery . nmod . in ( x _ 4 , x _ 7 ) "
          "AND sneeze . agent ( x _ 8 , x _ 1 )")
    expected = ("sneeze ( agent = * dog ( nmod . in = bakery ( "
                "nmod . in = * bag ) ) )")
    self.assertEqual(cogs_converter.cogs_lf_to_funcall(lf), expected)


if __name__ == "__main__":
  tf.test.main()
