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
"""Tests for utils."""

import os

from absl.testing import absltest
from language.wino_dict import utils
import spacy

_EXAMPLE = utils.WinoDictExample(
    source='winograd',
    sentence=('The city councilmen refused the demonstrators a permit because '
              '{{option}} {word} violence.'),
    pronoun='they',
    answer1=utils.WinoDictAnswer(
        idx=0,
        word='feared',
        option='The city councilmen',
        lemma='fear',
        pos='VERB',
        tag='VBD',
        morph='Tense=Past|VerbForm=Fin',
        definition='be scared of, or want to avoid an object',
        examples=[],
        fake_word='plested',
        fake_lemma='plest',
        fake_root='plest',
    ),
    answer2=utils.WinoDictAnswer(
        idx=1,
        word='advocated',
        option='The demonstrators',
        lemma='advocate',
        pos='VERB',
        tag='VBD',
        morph='Tense=Past|VerbForm=Fin',
        definition='publicly recommend or support',
        examples=[],
        fake_word='plested',
        fake_lemma='plest',
        fake_root='plest',
    ),
)

_EXAMPLE2 = utils.WinoDictExample(
    source='winograd',
    sentence=('The trophy doesn\'t fit into the brown suitcase because '
              '{{option}} is too {word}.'),
    pronoun='it',
    answer1=utils.WinoDictAnswer(
        idx=2,
        word='large',
        option='the trophy',
        lemma='large',
        pos='ADJ',
        tag='JJ',
        morph='Degree=Pos',
        definition='above average in size or number or quantity or magnitude or extent',
        examples=[
            'a large city', 'set out for the big city', 'a large sum',
            'a big (or large) barn', 'a large family', 'big businesses',
            'a big expenditure', 'a large number of newspapers',
            'a big group of scientists', 'large areas of the world'
        ],
        fake_word='huge',
        fake_lemma='huge',
        fake_root='huge',
    ),
    answer2=utils.WinoDictAnswer(
        idx=3,
        word='small',
        option='the suitcase',
        lemma='small',
        pos='ADJ',
        tag='JJ',
        morph='Degree=Pos',
        definition='limited or below average in number or quantity or magnitude or extent',
        examples=[
            'a little dining room', 'a little house', 'a small car',
            'a little (or small) group'
        ],
        fake_word='tiny',
        fake_lemma='tiny',
        fake_root='tiny',
    ),
)

_WINOGRADS = [
    utils.WinogradExample(
        source='winograd',
        sentence=('The trophy doesn\'t fit into the brown suitcase because '
                  '{{option}} is too large.'),
        option1='the trophy',
        option2='the suitcase',
        label=0,
        idx=2,
        group_key='',
        pronoun='it',
    ),
    utils.WinogradExample(
        source='winograd',
        sentence=('The trophy doesn\'t fit into the brown suitcase because '
                  '{{option}} is too small.'),
        option1='the trophy',
        option2='the suitcase',
        label=1,
        idx=3,
        group_key='',
        pronoun='it',
    ),
]


class UtilsTest(absltest.TestCase):

  def test_get_diff(self):
    self.assertEqual(
        utils._get_diff('The new sentence.', 'The old sentence.', 'X'),
        ('The X sentence.', ('new', 'old')))

  def test_definition_first_strategy(self):
    expected_result = (
        ('The verb to plest means to be scared of, or want to avoid an ' +
         'object. The city councilmen refused the demonstrators a permit ' +
         'because The city councilmen', 'plested violence.'),
        ('The verb to plest means to be scared of, or want to avoid an ' +
         'object. The city councilmen refused the demonstrators a permit ' +
         'because The demonstrators', 'plested violence.'),
        ('The verb to plest means to publicly recommend or support. The city ' +
         'councilmen refused the demonstrators a permit because The ' +
         'demonstrators', 'plested violence.'),
        ('The verb to plest means to publicly recommend or support. The city ' +
         'councilmen refused the demonstrators a permit because The city ' +
         'councilmen', 'plested violence.'))
    self.assertEqual(utils.definition_first_strategy(_EXAMPLE), expected_result)

  def test_synonym_first_strategy(self):
    expected_result = (
        ('The meaning of plest is similar to fear. The city councilmen ' +
         'refused the demonstrators a permit ' + 'because The city councilmen',
         'plested violence.'),
        ('The meaning of plest is similar to fear. The city councilmen ' +
         'refused the demonstrators a permit ' + 'because The demonstrators',
         'plested violence.'),
        ('The meaning of plest is similar to advocate. The city ' +
         'councilmen refused the demonstrators a permit because The ' +
         'demonstrators', 'plested violence.'),
        ('The meaning of plest is similar to advocate. The city ' +
         'councilmen refused the demonstrators a permit because The city ' +
         'councilmen', 'plested violence.'))
    self.assertEqual(utils.synonym_first_strategy(_EXAMPLE), expected_result)

  def test_build_few_shot_prompts(self):
    expected_result = (
        ('The trophy doesn\'t fit into the brown suitcase because the trophy ' +
         'is too huge.\n\nThe city councilmen refused the demonstrators a ' +
         'permit because The city councilmen', 'plested violence.'),
        ('The trophy doesn\'t fit into the brown suitcase because the trophy ' +
         'is too huge.\n\nThe city councilmen refused the demonstrators a ' +
         'permit because The demonstrators', 'plested violence.'),
        ('The trophy doesn\'t fit into the brown suitcase because the ' +
         'suitcase is too tiny.\n\nThe city councilmen refused the ' +
         'demonstrators a permit because The demonstrators',
         'plested violence.'),
        ('The trophy doesn\'t fit into the brown suitcase because the ' +
         'suitcase is too tiny.\n\nThe city councilmen refused the ' +
         'demonstrators a permit because The city councilmen',
         'plested violence.'))
    self.assertEqual(
        utils._build_few_shot_prompts(_EXAMPLE, [_EXAMPLE2],
                                      utils.no_definition_strategy),
        expected_result)

  def test_analyze(self):
    nlp = spacy.load(spacy_model)

    def create_words(seed, word, lemma, tag):
      del seed, word, lemma, tag
      return ('word_squall', 'lemma_squall', 'root_squall')

    expected_result = {
        'definition': '(used of color) having a relatively small amount of '
                      'coloring agent',
        'examples': [
            'light blue', 'light colors such as pastels',
            'a light-colored powder'
        ],
        'fake_lemma': 'lemma_squall',
        'fake_word': 'word_squall',
        'fake_root': 'root_squall',
        'lemma': 'light',
        'morph': 'Degree=Cmp',
        'pos': 'ADJ',
        'tag': 'JJR',
    }
    self.assertEqual(
        utils._analyze(
            nlp,
            ('The stain was {word} on Jason\'s shirt than Donald\'s because ' +
             '{{option}} spilled red wine and not white wine.'), 'lighter',
            'Donald', 'darker', create_words), expected_result)

  def test_make_comparative(self):
    self.assertEqual(utils._make_comparative('hot'), 'hotter')
    self.assertEqual(utils._make_comparative('iot'), 'ioter')
    self.assertEqual(utils._make_comparative('thin'), 'thinner')
    self.assertEqual(utils._make_comparative('dole'), 'doler')
    self.assertEqual(utils._make_comparative('doly'), 'dolier')

  def test_get_winodict(self):

    def create_words(seed, word, lemma, tag):
      del seed, lemma, tag
      return ('tiny', 'tiny', 'tiny') if word == 'small' else ('huge', 'huge',
                                                               'huge')

    self.assertEqual(
        list(
            utils.get_winodict(_WINOGRADS, create_words,
                               'en_core_web_md-3.0.0a1')), [_EXAMPLE2])

  def test_create_word(self):
    for tag in ('JJR', 'RBR'):
      self.assertEqual(
          utils.create_word('0', '_', '_', tag, [dict(JJ='swip', root='swip')]),
          ('swipper', 'swip', 'swip'))
      candidates = [dict(JJ='swiplong', root='swiplong')]
      for seed in range(10):
        self.assertEqual(
            utils.create_word(str(seed), '_', '_', tag, candidates),
            ('more swiplong', 'swiplong', 'swiplong'))


if __name__ == '__main__':
  absltest.main()
