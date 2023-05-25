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
r"""Script to generate the WinoDict dataset.

In order to download external resources you can run:

python -m nltk.downloader omw-1.4
python -m nltk.downloader wordnet
python -m spacy download en_core_web_md-3.0.0a1

"""

import functools
import os
from typing import Sequence, Mapping

from absl import app
from absl import flags
from absl import logging
from language.wino_dict import utils
import tensorflow as tf

_WINOGRANDE_PATH = flags.DEFINE_string(
    'winogrande_path',
    'gs://ai2-mosaic/public/winogrande/winogrande_1.1.zip',
    help='Path to winogrande zipped dataset.')

_SPACY_MODEL = flags.DEFINE_string(
    'spacy_model', 'en_core_web_md-3.0.0a1', help='spacy model to use.')

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    help='Path where the datasets will be created.',
    required=True)

_WORDS_PATH = flags.DEFINE_string(
    'words_path', None, help='Path for TSV file of new words.', required=True)

_SEED = flags.DEFINE_integer(
    'seed', 42, help='Seed for building few shot examples.')

_MIN_LL = flags.DEFINE_float(
    'min_ll', -30, help='Minimum number of of log likelihood for new words.')

_MAX_LL = flags.DEFINE_float(
    'max_ll', -10, help='Maximum number of of log likelihood for new words.')

_LL_BUCKETS = flags.DEFINE_integer(
    'll_buckets', 5, help='Number of log likelihood buckets to create.')

_SHOTS = flags.DEFINE_list(
    'shots', ['0', '1', '5'],
    help='Number of examples to include in the prompt.')


def _word_candidates(idx: int) -> Sequence[Mapping[str, str]]:
  """Extracts word candidates with a certain probability bucket."""
  factor = _LL_BUCKETS.value / (_MAX_LL.value - _MIN_LL.value)
  candidates = []
  with tf.io.gfile.GFile(_WORDS_PATH.value, 'r') as f:
    for line in f:
      root, score, morph = line.split('\t')
      candidate = dict(p.split(':') for p in morph.strip().split(','))
      if (idx < (float(score) - _MIN_LL.value) * factor <= idx + 1 and
          utils.is_candidate_new(candidate)):
        candidate[utils.ROOT] = root
        candidates.append(candidate)
  # We need enough candidates to build few-shot examples without repeating.
  if len(candidates) < 2 * max(map(int, _SHOTS.value)):
    raise ValueError(f'Only {len(candidates)} candidates in bucket {idx}.')
  logging.info('%d candidates in bucket %d.', len(candidates), idx)
  return candidates


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  sources = dict(
      winogrande=list(utils.get_winogrande(_WINOGRANDE_PATH.value)),
      winograd=list(utils.get_winograd()),
  )

  create_word_fns = dict(
      unchanged=lambda seed, word, lemma, tag: (word, lemma, lemma),
  )

  for i in range(_LL_BUCKETS.value):
    create_word_fns[f'prob{i + 1}_of_{_LL_BUCKETS.value}'] = functools.partial(
        utils.create_word, candidates=_word_candidates(i))

  strategies = dict(
      no_def=utils.no_definition_strategy,
      last_def=utils.definition_last_strategy,
      first_def=utils.definition_first_strategy,
      first_syn=utils.synonym_first_strategy,
      last_syn=utils.synonym_last_strategy,
      first_def_syn=utils.definition_synonym_first_strategy,
      last_def_syn=utils.definition_synonym_last_strategy,
  )

  tf.io.gfile.makedirs(_OUTPUT_PATH.value)
  counter = 0
  total = (
      len(sources) * len(create_word_fns) * len(strategies) * len(_SHOTS.value))

  for source_name, source in sources.items():
    logging.info('Input %s examples %d.', source_name, len(source))
  for create_word_name, create_word_fn in create_word_fns.items():
    text_files = []
    for src_idx, (source_name, source) in enumerate(sorted(sources.items())):
      winodict = sorted(
          utils.get_winodict(source, create_word_fn, _SPACY_MODEL.value),
          key=lambda wd: wd.get_id())
      logging.info('Output %s_%s examples %d.', source_name, create_word_name,
                   2 * len(winodict))
      file_path = os.path.join(_OUTPUT_PATH.value,
                               f'{source_name}_{create_word_name}')
      text_files.append(file_path)
      utils.write_text_files(file_path, winodict, id_offset=src_idx * 1000)
      for strategy_name, strategy in strategies.items():
        task = f'winodict_{source_name}_{create_word_name}_{strategy_name}'
        task_path = os.path.join(_OUTPUT_PATH.value, task)
        tf.io.gfile.makedirs(task_path)
        for shots in _SHOTS.value:
          utils.write_prompt_files(
              os.path.join(task_path, f'{task}_{shots}shot'), winodict,
              winodict, int(shots), strategy, _SEED.value)
          counter += 1
          logging.info('%d/%d: Done writing %s_%sshot', counter, total, task,
                       shots)
    utils.merge_files(
        os.path.join(_OUTPUT_PATH.value, create_word_name), text_files)


if __name__ == '__main__':
  app.run(main)
