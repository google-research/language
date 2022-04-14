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
"""Export numpy embeddings to TF Embedding Projector format."""
import os


from absl import app
from absl import flags
from absl import logging
from language.serene import util
import numpy as np


FLAGS = flags.FLAGS
flags.DEFINE_string('urls_path', None, 'Path to numpy urls file')
flags.DEFINE_string(
    'sentence_ids_path', None, 'Path to numpy sentence_ids file')
flags.DEFINE_string('embeddings_path', None, 'Path to numpy embeddings')
flags.DEFINE_string(
    'out_dir', None, 'Directory to write embeddings.tsv and metadata.tsv to')


def save_as_tsv(
    *,
    embeddings, metadata, header,
    out_dir):
  """Save the embeddings and their metadata to TSV for embedding projector.

  Args:
    embeddings: Numpy 2d Array of embeddings
    metadata: List of metadata, where each row can have one or more string keys
    header: The header row of metadata, should match fields in metadata
    out_dir: Output directory to write
  """
  if embeddings.shape[0] != len(metadata):
    raise ValueError(
        f'Incompatible shapes: {embeddings.shape[0]} and {len(metadata)}')

  logging.info('Building TSV Formatted Output')
  emb_lines: List[Text] = []
  meta_lines: List[Text] = [header]
  for idx in range(embeddings.shape[0]):
    vec = embeddings[idx]
    meta = metadata[idx]
    emb_lines.append('\t'.join([str(v) for v in vec.tolist()]))
    meta_lines.append('\t'.join(meta))

  emb_content = '\n'.join(emb_lines)
  meta_content = '\n'.join(meta_lines)
  logging.info('Writing TSV embeddings and metadata to: %s', out_dir)
  with util.safe_open(os.path.join(out_dir, 'embeddings.tsv'), 'w') as f:
    f.write(emb_content)

  with util.safe_open(os.path.join(out_dir, 'metadata.tsv'), 'w') as f:
    f.write(meta_content)


def load_np_embeddings(
    *,
    urls_path, sentence_ids_path, embeddings_path):
  """Load numpy embeddings into the format expected by save_as_tsv.

  Files are read in the format used by np.save/load

  Args:
    urls_path: Path to numpy wikipedia urls file
    sentence_ids_path: Path to the sentence ids file
    embeddings_path: Path to the embeddings file

  Returns:
    Embeddings, metadata, and header for meta file
  """
  with util.log_time(f'Loading: {urls_path}'):
    with util.safe_open(urls_path, 'rb') as f:
      urls: np.ndarray = np.load(f)

  with util.log_time(f'Loading: {urls_path}'):
    with util.safe_open(sentence_ids_path, 'rb') as f:
      sentence_ids: np.ndarray = np.load(f)

  with util.log_time(f'Loading: {urls_path}'):
    with util.safe_open(embeddings_path, 'rb') as f:
      embeddings: np.ndarray = np.load(f)

  logging.info('Embedding shape: %s', embeddings.shape)

  if len({urls.shape[0], sentence_ids.shape[0], embeddings.shape[0]}) != 1:
    raise ValueError('Incompatible lengths of embedding files')

  metadata = []
  for wikipedia_url, sentence_id in zip(urls.tolist(), sentence_ids.tolist()):
    metadata.append((wikipedia_url, str(sentence_id)))

  return embeddings, metadata, 'wikipedia_url\tsentence_id'


def main(_):
  flags.mark_flag_as_required('urls_path')
  flags.mark_flag_as_required('sentence_ids_path')
  flags.mark_flag_as_required('embeddings_path')
  flags.mark_flag_as_required('out_dir')
  embeddings, metadata, header = load_np_embeddings(
      urls_path=FLAGS.urls_path,
      sentence_ids_path=FLAGS.sentence_ids_path,
      embeddings_path=FLAGS.embeddings_path,
  )
  save_as_tsv(
      embeddings=embeddings,
      metadata=metadata,
      header=header,
      out_dir=FLAGS.out_dir,
  )


if __name__ == '__main__':
  app.run(main)
