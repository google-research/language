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
"""Helper classes for the WinoDict dataset."""

import collections
import csv
import dataclasses
import json
import os
import random
import re
import shutil
from typing import Callable, Mapping, Sequence, Tuple, Union, Any, List, Iterator
import zipfile

from absl import logging
from nltk.corpus import wordnet
import spacy
import tensorflow as tf
import tensorflow_datasets as tfds

ROOT = 'root'
_PROPN = 'PROPN'
_FORMAT_OPTION = '{option}'
_OPTION = '{' + _FORMAT_OPTION + '}'
_WORD = '{word}'
_POS_TAGS = dict(
    VERB=wordnet.VERB,
    NOUN=wordnet.NOUN,
    ADJ=wordnet.ADJ,
    ADV=wordnet.ADV,
)
_SYNONYM = 'The meaning of {lemma} is similar to {synonym}.'
_DEFINITIONS = dict(
    VERB='The verb to {lemma} means to {definition}.',
    NOUN='The word {lemma} refers to {definition}.',
    ADJ='The meaning of {lemma} is {definition}.',
    ADV='The word {lemma} means {definition}.',
)
_SYNSETS = dict(
    ab='abdominal.n.01',
    accept='accept.v.02',
    appear='appear.v.05',
    available='available.s.02',
    bitter='bitter.s.06',
    clear='clear.a.04',
    concentration='concentration.n.05',
    cooperative='cooperative.a.02',
    crash='rapid.s.01',  # For crash diet.
    crush='crush.v.05',
    dark='dark.a.02',  # For colors.
    deer='stag.n.02',
    deliver='deliver.v.02',
    distribution='distribution.n.04',
    embarrassed='embarrassed.s.02',
    employ='hire.v.01',
    fail='fail.v.07',
    firm='firm.s.02',
    flexible='flexible.a.02',
    fluid='fluid.s.02',
    fresh='clean.s.04',
    furious='angered.s.01',
    gear='gear.n.04',
    give='give.v.03',
    great='bang-up.s.01',
    grip='intrigue.v.01',
    hurt='anguished.s.01',
    inflexible='inflexible.a.03',
    itchy='itchy.s.02',
    judge='judge.v.05',
    leave='leave.v.07',
    level='level.s.03',
    long='long.a.02',  # Spatial sense.
    loose='loose.a.03',
    make='make.v.15',
    minority='minority.n.02',
    nasty='filthy.s.01',
    need='need.v.03',
    offer='offer.v.02',
    pat='stroke.v.01',
    persistent='dogged.s.01',
    plain='plain.s.06',
    remember='remember.v.02',
    right='correct.a.01',
    run='operate.v.01',
    savory='piquant.s.01',
    send='send.v.06',
    shame='embarrassment.n.01',
    share='share.v.02',
    sharp='acute.s.03',
    silence='silence.n.02',
    skeptical='doubting.s.01',
    sloppy='haphazard.s.02',
    stay='stay.v.02',
    stop='discontinue.v.01',
    straight='straight.a.03',
    stretch='stretch.v.02',
    strip='airstrip.n.01',
    sun='sunlight.n.01',
    sympathetic='sympathetic.a.02',
    take='take.v.08',
    tough='rugged.s.04',
    treat='treat.v.05',
    well='good.a.01',
    withering='austere.s.02',
)
_ADV_SYNSETS = dict(
    long='long.r.01',
    here='here.r.01',
    inside='inside.r.01',
    outside='outside.r.01',
    away='off.r.02',
)
_SURVEY_TEMPLATE = ('{num}. {definition}\n<br><br>\nIn the sentence '
                    '<strong>\'{sentence}\'</strong> the term '
                    '<strong>{pronoun}</strong> is more likely to refer to:\n\n'
                    '{option1}\n{option2}\n\n')

AnswerID = Union[str, int]
# inputs-targets pairs to be scores.
Pair = Tuple[str, str]
# For each WinoDict exampel we derived four pairs to score.
Prompts = Tuple[Pair, Pair, Pair, Pair]
# Function that takes seed, word, lemma and tag to create a word + lemma + root.
CreateWordFn = Callable[[str, str, str, str], Tuple[str, str, str]]


@dataclasses.dataclass(frozen=True)
class WinogradExample:
  source: str
  sentence: str
  option1: str
  option2: str
  label: int
  idx: Union[str, int]
  pronoun: str
  group_key: str


@dataclasses.dataclass(frozen=True)
class WinoDictAnswer:
  """Container for all the parts in WinoDict answer."""
  idx: AnswerID
  word: str
  option: str
  lemma: str
  pos: str
  tag: str
  morph: str
  definition: str
  examples: Sequence[str]
  fake_word: str  # Used for replacing inside sentences.
  fake_lemma: str  # Used for building definitions.
  fake_root: str  # Used for building k-shot instances without repetitions.

  def get_definition(self) -> str:
    return _DEFINITIONS[self.pos].format(
        definition=self.definition, lemma=self.fake_lemma)

  def get_synonym(self) -> str:
    return _SYNONYM.format(lemma=self.fake_lemma, synonym=self.lemma)

  def get_definition_synonym(self) -> str:
    return f'{self.get_definition()} {self.get_synonym()}'


@dataclasses.dataclass(frozen=True)
class WinoDictExample:
  """Container for all the parts in WinoDict example."""
  source: str
  sentence: str
  pronoun: str
  answer1: WinoDictAnswer
  answer2: WinoDictAnswer

  def get_id(self) -> Tuple[AnswerID, AnswerID]:
    return (self.answer1.idx, self.answer2.idx)


def is_candidate_new(candidate: Mapping[str, str]) -> bool:
  return all(not wordnet.synsets(candidate[tag]) for tag in ('VB', 'JJ', 'NN'))


def get_winogrande(path: str) -> Iterator[WinogradExample]:
  """Fetches the Winogrande dataset from a storage bucket."""
  with tf.io.gfile.GFile(path, 'rb') as raw:
    with zipfile.ZipFile(raw) as zf:
      with zf.open('winogrande_1.1/dev.jsonl') as f:
        for line in f:
          example = json.loads(line)
          if len(example['sentence'].split('_')) != 2:
            raise ValueError(f'Invalid example {example}.')
          yield WinogradExample(
              source='winogrande',
              sentence=example['sentence'].replace('_', _OPTION),
              option1=example['option1'],
              option2=example['option2'],
              label=int(example['answer']) - 1,
              idx=example['qID'],
              group_key=example['qID'].split('-')[0],
              pronoun='_')


def get_winograd() -> Iterator[WinogradExample]:
  """Fetches the Winograd dataset from TFDS."""
  for example in tfds.load('wsc273', split='test'):
    sentence = example['text'].numpy().decode()
    pronoun = example['pronoun_text'].numpy().decode()
    start = example['pronoun_start'].numpy()
    end = example['pronoun_end'].numpy()
    if sentence[start:end] != pronoun:
      raise ValueError(f'Invalid example {example}.')
    yield WinogradExample(
        source='winograd',
        sentence=sentence[:start] + _OPTION + sentence[end:],
        option1=example['option1'].numpy().decode(),
        option2=example['option2'].numpy().decode(),
        idx=example['idx'].numpy(),
        label=example['label'].numpy(),
        group_key='',
        pronoun=example['pronoun_text'].numpy().decode())


def _make_comparative(adj: str) -> str:
  """"Heuristic function to build comparative adjectives."""
  if adj[-1] == 'y':
    return f'{adj[:-1]}ier'
  if adj[-1] == 'e':
    return f'{adj}r'
  if re.search('[b-df-hj-np-tv-xz][aeiyou][b-df-hj-np-tvxz]$', adj):
    return f'{adj}{adj[-1]}er'
  return f'{adj}er'


def create_word(
    seed: int,
    word: str,
    lemma: str,
    tag: str,
    candidates: Sequence[Mapping[str, str]],
) -> Tuple[str, str, str]:
  """Samples a new (word, lemma, root) tuple within a log likelihood bucket."""
  morphology = random.Random(seed).choice(candidates)
  root = morphology[ROOT]
  if tag == 'VBP':
    return morphology['VB'], morphology['VB'], root
  if tag.startswith('V'):
    return morphology[tag], morphology['VB'], root
  if tag.startswith('N'):
    return morphology[tag], morphology['NN'], root
  adj = morphology['JJ']
  if tag in ('JJ', 'RB'):
    return adj, adj, root
  if tag in ('JJR', 'RBR'):
    if len(adj) > 6 or random.Random(f'{adj}:{seed}').random() < 0.25:
      return f'more {adj}', adj, root
    return _make_comparative(adj), adj, root
  raise ValueError(f'Cannot resolve {tag} for word {word}/{lemma}.')


def _build_template(
    sentence: str,
    start_idx: int,
    end_idx: int,
    marker: str,
) -> Tuple[str, str]:
  """Replaces a span in the text with a marker."""
  return sentence[:start_idx] + marker + sentence[end_idx:], sentence[
      start_idx:end_idx]


def _normalize_spaces(sentence: str) -> str:
  return ' '.join(sentence.split())


def _get_diff(
    sentence1: str,
    sentence2: str,
    marker: str,
) -> Tuple[str, Tuple[str, str]]:
  """Replaces smalles substring difference between two sentences."""
  sentence1 = _normalize_spaces(sentence1)
  sentence2 = _normalize_spaces(sentence2)
  if sentence1[-1] != sentence2[-1]:
    if sentence1[-1] != '.':
      logging.info('Adding punctuation to "%s"', sentence1)
      sentence1 = sentence1 + '.'
    if sentence2[-1] != '.':
      logging.info('Adding punctuation to "%s"', sentence2)
      sentence2 = sentence2 + '.'

  start_idx, end_idx = 0, 1
  for start in re.finditer(r'\b\w', sentence1):
    idx = start.span()[0]
    if sentence1[:idx] == sentence2[:idx]:
      start_idx = idx
  for end in reversed(list(re.finditer(r'\w\b', sentence1))):
    idx = end.span()[1] - len(sentence1)
    if sentence1[idx:] == sentence2[idx:]:
      end_idx = idx
  if end_idx >= 0:
    raise ValueError('Indexing with -0 will not work in this formulation so '
                     f'{sentence1} cannot end in a word character.')
  template1, option1 = _build_template(sentence1, start_idx, end_idx, marker)
  template2, option2 = _build_template(sentence2, start_idx, end_idx, marker)
  assert template1 == template2
  return template1, (option1, option2)


def _analyze(
    nlp: spacy.Language,
    sentence: str,
    word: str,
    option: str,
    opposite_word: str,
    create_word_fn: CreateWordFn,
) -> Mapping[str, str]:
  """Analyzes the syntactic role of a word in a sentence."""
  # The 2 versions of the same example will share the fake word.
  seed = sentence
  sentence = sentence.replace(_OPTION, option)
  tokens = {token.idx: token for token in nlp(sentence.format(word=word))}
  token = tokens.get(sentence.index(_WORD))
  if not token:
    raise ValueError('token not found in document')

  option_pos = set(t.pos_ for t in tokens.values() if t.text == option)

  pos, tag, lemma = token.pos_, token.tag_, token.lemma_
  dobj = set(child.lemma_ for child in token.children if child.dep_ == 'dobj')
  if lemma in ('more', 'less', 'few'):
    raise ValueError('quantifier')
  if lemma in ('be'):
    raise ValueError('Skipped verb')

  wordnet_pos = _POS_TAGS.get(pos)
  if not wordnet_pos:
    raise ValueError(f'{pos}:{tag} is not supported.')
  if lemma == 'lose' and not dobj:
    synsets = [wordnet.synset('lose.v.02')]  # "not winning" sense.
  elif lemma == 'keep' and opposite_word == 'lose':
    synsets = [wordnet.synset('keep.v.03')]  # Retain possession sense.
  elif lemma == 'strong' and opposite_word == 'soft':
    synsets = [wordnet.synset('hard.a.03')]  # "not soft" sense.
  elif lemma == 'soft' and opposite_word == 'louder':
    synsets = [wordnet.synset('soft.a.03')]  # Low volume sense.
  elif lemma == 'work' and opposite_word == 'unemployed':
    synsets = [wordnet.synset('work.v.02')]
  elif lemma == 'hard' and opposite_word in ('soft', 'flimsy'):
    synsets = [wordnet.synset('hard.a.03')]  # "not soft" sense.
  elif lemma == 'old' and opposite_word in ('new',):
    synsets = [wordnet.synset('old.a.02')]  # "not new" sense.
  elif lemma == 'full' and (opposite_word in ('hungry',) or
                            _PROPN in option_pos):
    synsets = [wordnet.synset('full.s.04')]
  elif lemma == 'light' and (option.lower() in ('brown',) or
                             opposite_word in ('darker',)):
    synsets = [wordnet.synset('light.a.02')]  # For colors.
  elif lemma == 'heavy' and option.lower() in ('black',):
    synsets = [wordnet.synset('dark.a.02')]  # For colors.
  elif lemma == 'high' and option.lower() in ('boots', 'the shelf'):
    synsets = [wordnet.synset('high.a.02')]  # Literal sense.
  elif lemma == 'low' and option.lower() in ('shoes',):
    synsets = [wordnet.synset('low.a.02')]  # Literal sense.
  elif lemma == 'pass':
    synsets = [wordnet.synset('pass.v.14')]
  elif lemma == 'poor':
    if _PROPN in option_pos:
      synsets = [wordnet.synset('poor.a.02')]  # not rich sense.
    else:
      synsets = [wordnet.synset('poor.s.06')]  # unsatisfactory sense.
  elif lemma == 'shallow' and _PROPN in option_pos:
    synsets = [wordnet.synset('shallow.s.03')]  # Metaphorical.
  elif lemma == 'deep':
    if _PROPN in option_pos:
      synsets = [wordnet.synset('deep.s.02')]  # Metaphorical.
    else:
      synsets = [wordnet.synset('deep.a.03')]  # Physical sense.
  elif lemma == 'short':
    if _PROPN in option_pos or option.lower() in ('tent',):
      synsets = [wordnet.synset('short.a.03')]  # Stature sense.
    elif option.lower() in ('the meeting', 'story'):
      synsets = [wordnet.synset('short.a.01')]  # Temporal sense.
    else:
      synsets = [wordnet.synset('short.a.02')]  # Spatial sense.
  elif wordnet_pos == wordnet.ADV:
    if lemma in _ADV_SYNSETS:
      synsets = [wordnet.synset(_ADV_SYNSETS[lemma])]
    else:
      # Adjective definitions are usually better descriptors.
      synsets = (
          wordnet.synsets(lemma, pos=wordnet.ADJ) +
          wordnet.synsets(lemma, pos=wordnet.ADV))
  elif lemma in _SYNSETS:
    synsets = [wordnet.synset(_SYNSETS[lemma])]
  else:
    synsets = wordnet.synsets(lemma, pos=wordnet_pos)
  if not synsets:
    raise ValueError(f'no synset found for {lemma}:{pos}.')

  definition = synsets[0].definition()
  # Remove suffixes/prefixes from definitions.
  definition = definition.split('; ;')[0]
  definition = definition.split('; --')[0]
  definition = definition.split('; e.g.')[0]
  definition = definition.split(' (sometimes used in combinations')[0]
  definition = definition.replace('(literal meaning) ', '')
  definition = definition.replace('literal meanings; ', '')
  # Remove to from the beginning on some definitions.
  if definition.startswith('to '):
    definition = definition[3:]
  fake_word, fake_lemma, fake_root = create_word_fn(seed, word, lemma, tag)
  return dict(
      pos=pos,
      tag=tag,
      morph=str(token.morph),
      lemma=lemma,
      definition=definition,
      fake_word=fake_word,
      fake_lemma=fake_lemma,
      fake_root=fake_root,
      examples=synsets[0].examples())


def get_winodict(
    lines: Sequence[WinogradExample],
    create_word_fn: CreateWordFn,
    spacy_model: str,
) -> Iterator[WinoDictExample]:
  """Builds WinoDict dataset based on Winograd examples."""
  nlp = spacy.load(spacy_model)
  # Examples are grouped by the option set and the group (original id prefix) if
  # available. This is used to identify the example pairs from the original set.
  grouped_examples = collections.defaultdict(list)
  for example in sorted(lines, key=lambda e: e.idx):
    key = (example.group_key, (example.option1, example.option2))
    grouped_examples[key].append(example)

  counter = collections.Counter()
  for (_, options), examples in sorted(grouped_examples.items()):
    if len(examples) % 2:
      counter[f'odd number of examples ({len(examples)})'] += 1
      continue
    # In WinoGrande this already identifies the pair completely, in Winograd
    # since we don't have a group key, we still rely on the order of the
    # examples to identify the pair, since they have consecutive ids. Sadly
    # id % 2 is not a good group key either since one example has length three.
    for idx in range(0, len(examples), 2):
      example1 = examples[idx]
      example2 = examples[idx + 1]
      sentence, words = _get_diff(example1.sentence, example2.sentence, _WORD)
      assert example1.pronoun == example2.pronoun

      correct_index = (example1.label, example2.label)
      if correct_index == (0, 1):
        pass
      elif correct_index == (1, 0):
        options = (options[1], options[0])
      else:
        raise ValueError(
            f'Invalid correct_index value {correct_index} for {example1}')

      if _OPTION in words[0]:
        counter['pivot contains the option'] += 1
        continue
      if len(words[0].split()) != 1 or len(words[1].split()) != 1:
        counter['pivot contains multiple tokens'] += 1
        continue

      try:
        word1_analysis = _analyze(nlp, sentence, words[0], options[0], words[1],
                                  create_word_fn)
      except ValueError as e:
        counter[str(e)] += 1
        logging.info('Skipping "%s" due to analysis error %s in %s', words[0],
                     e, example1)
        continue

      try:
        word2_analysis = _analyze(nlp, sentence, words[1], options[1], words[0],
                                  create_word_fn)
      except ValueError as e:
        counter[str(e)] += 1
        logging.info('Skipping "%s" due to analysis error %s in %s', words[1],
                     e, example2)
        continue

      yield WinoDictExample(
          source=example1.source,
          sentence=sentence,
          pronoun=example1.pronoun,
          answer1=WinoDictAnswer(
              idx=example1.idx,
              word=words[0],
              option=options[0],
              **word1_analysis),
          answer2=WinoDictAnswer(
              idx=example2.idx,
              word=words[1],
              option=options[1],
              **word2_analysis),
      )

  logging.info('Counters:\n%s',
               '\n'.join(f'{k}:{v}' for k, v in counter.most_common()))


def _examples_compatible(
    example1: WinoDictExample,
    example2: WinoDictExample,
) -> bool:
  """Assesses whether to exampels are compatible for in-context learning."""
  if example1.get_id() == example2.get_id():
    return False
  elif example1.answer1.fake_root == example2.answer1.fake_root:
    return False
  elif example1.answer2.fake_root == example2.answer2.fake_root:
    return False
  return True


def _build_few_shot_prompts(
    example: WinoDictExample,
    samples: Sequence[WinoDictExample],
    strategy: Callable[[WinoDictExample], Prompts],
) -> Prompts:
  """Builds a few-shot prompt given samples."""
  # For each sample used for ICL, we add the 2 prompts coming from both words.
  sample_prompts = ([], [])
  for sample in samples:
    prompt1, _, prompt2, _ = strategy(sample)
    # Concatenated *correct* input and output from the sample using both words.
    sample_prompt1 = ' '.join(prompt1)
    sample_prompt2 = ' '.join(prompt2)
    sample_prompts[0].append(sample_prompt1)
    sample_prompts[1].append(sample_prompt2)

  # Each sample is repeated for both the positive and negative example.
  prompt_parts = (sample_prompts[0], sample_prompts[0], sample_prompts[1],
                  sample_prompts[1])
  return tuple(('\n\n'.join(prompt_part + [part[0]]), part[1])
               for prompt_part, part in zip(prompt_parts, strategy(example)))


def merge_files(output_path: str, files: Sequence[str]) -> None:
  for ext in ('.csv', '.txt'):
    with tf.io.gfile.GFile(output_path + ext, 'wb') as merged:
      for chunk_file in files:
        with tf.io.gfile.GFile(chunk_file + ext, 'rb') as ef:
          shutil.copyfileobj(ef, merged)


def _get_answer_row(example: WinoDictExample, answer_idx: int) -> List[Any]:
  answer = example.answer2 if answer_idx else example.answer1
  return [
      answer.idx,
      answer.lemma,
      answer.fake_lemma,
      answer.pos,
      answer.tag,
      example.pronoun,
      answer.get_definition(),
      example.sentence.format(word=answer.fake_word).format(option='_'),
  ]


def _highlight(haystack: str, needle: str) -> str:
  return haystack.replace(needle, f'<strong>{needle}</strong>')


def write_text_files(dataset_name: str, dataset: Sequence[WinoDictExample],
                     id_offset: int):
  """Dumpts dataset to csv and text."""
  with tf.io.gfile.GFile(f'{dataset_name}.csv', 'w') as csv_file:
    with tf.io.gfile.GFile(f'{dataset_name}.txt', 'w') as txt_file:
      writer = csv.writer(csv_file)
      writer.writerow([
          'id', 'lemma', 'fake_lemma', 'pos', 'tag', 'pronoun', 'definition',
          'sentence', 'option1', 'option2', 'label'
      ])
      for idx, example in enumerate(dataset):
        # Randomize option order
        label = int(random.Random(example.get_id()).random() > 0.5)
        if label:
          options = [example.answer2.option, example.answer1.option]
        else:
          options = [example.answer1.option, example.answer2.option]
        row1 = _get_answer_row(example, 0)
        row2 = _get_answer_row(example, 1)
        writer.writerow(row1 + options + [label])
        writer.writerow(row2 + options + [1 - label])
        txt_file.write(
            _SURVEY_TEMPLATE.format(
                num=2 * idx + 1 + id_offset,
                definition=_highlight(row1[-2], example.answer1.fake_lemma),
                sentence=row1[-1].replace('_', example.pronoun),
                option1=options[0],
                option2=options[1],
                pronoun=example.pronoun,
            ))
        txt_file.write(
            _SURVEY_TEMPLATE.format(
                num=2 * idx + 2 + id_offset,
                definition=_highlight(row2[-2], example.answer2.fake_lemma),
                sentence=row2[-1].replace('_', example.pronoun),
                option1=options[0],
                option2=options[1],
                pronoun=example.pronoun,
            ))


def write_prompt_files(
    dataset_name: str,
    dataset: Sequence[WinoDictExample],
    aux_dataset: Sequence[WinoDictExample],
    shots: int,
    strategy: Callable[[WinoDictExample], Prompts],
    seed: int,
):
  """Creates in-context learning given a strategy to write the prompts."""
  with tf.io.gfile.GFile(f'{dataset_name}_inputs', 'w') as f_inputs:
    with tf.io.gfile.GFile(f'{dataset_name}_labels', 'w') as f_outputs:
      for example_index, example in enumerate(dataset):
        candidates = [
            e for e in aux_dataset if _examples_compatible(e, example)
        ]
        if len(candidates) < shots:
          raise ValueError(
              f'Only {len(candidates)} examples are compatible with {example}')

        samples = random.Random(f'{example.get_id()}:{seed}').sample(
            candidates, shots)
        # Corresponds to the first correct example, the second wrong example
        # then second correct example and the second wrong example
        prompts = _build_few_shot_prompts(example, samples, strategy)
        indexes = (0, 1, 0, 1)
        if isinstance(example.answer1.idx, int):
          example_ids = [example.answer1.idx] * 2 + [example.answer2.idx] * 2
        else:
          example_ids = [2 * example_index] * 2 + [2 * example_index + 1] * 2

        for (inputs, targets), index, ids in zip(prompts, indexes, example_ids):
          f_inputs.write(f'{inputs.encode()}\t{targets.encode()}\n')
          f_outputs.write(f'(({ids}, {index}), {index == 0}, 1.0)\n')


def _template_strategy(template1: str, template2: str,
                       example: WinoDictExample) -> Prompts:
  inputs1, targets1 = template1.format(
      word=example.answer1.fake_word).split(_FORMAT_OPTION)
  inputs1 += _FORMAT_OPTION
  inputs2, targets2 = template2.format(
      word=example.answer2.fake_word).split(_FORMAT_OPTION)
  inputs2 += _FORMAT_OPTION
  return ((inputs1.format(option=example.answer1.option), targets1.strip()),
          (inputs1.format(option=example.answer2.option), targets1.strip()),
          (inputs2.format(option=example.answer2.option), targets2.strip()),
          (inputs2.format(option=example.answer1.option), targets2.strip()))


def no_definition_strategy(example: WinoDictExample) -> Prompts:
  return _template_strategy(example.sentence, example.sentence, example)


def definition_first_strategy(example: WinoDictExample) -> Prompts:
  """Strategy to add the word definition to the beginning of the prompt."""
  return _template_strategy(
      f'{example.answer1.get_definition()} {example.sentence}',
      f'{example.answer2.get_definition()} {example.sentence}', example)


def definition_last_strategy(example: WinoDictExample) -> Prompts:
  """Strategy to add the word definition to the end of the prompt."""
  return _template_strategy(
      f'{example.sentence} {example.answer1.get_definition()}',
      f'{example.sentence} {example.answer2.get_definition()}', example)


def definition_synonym_first_strategy(example: WinoDictExample) -> Prompts:
  """Strategy to add the word definition to the beginning of the prompt."""
  return _template_strategy(
      f'{example.answer1.get_definition_synonym()} {example.sentence}',
      f'{example.answer2.get_definition_synonym()} {example.sentence}', example)


def definition_synonym_last_strategy(example: WinoDictExample) -> Prompts:
  """Strategy to add the word definition to the end of the prompt."""
  return _template_strategy(
      f'{example.sentence} {example.answer1.get_definition_synonym()}',
      f'{example.sentence} {example.answer2.get_definition_synonym()}', example)


def synonym_first_strategy(example: WinoDictExample) -> Prompts:
  """Strategy to add the word definition to the beginning of the prompt."""
  return _template_strategy(
      f'{example.answer1.get_synonym()} {example.sentence}',
      f'{example.answer2.get_synonym()} {example.sentence}', example)


def synonym_last_strategy(example: WinoDictExample) -> Prompts:
  """Strategy to add the word definition to the end of the prompt."""
  return _template_strategy(
      f'{example.sentence} {example.answer1.get_synonym()}',
      f'{example.sentence} {example.answer2.get_synonym()}', example)
