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
"""retrieve from passages."""
import json
import os.path
import time

from absl import app
from absl import flags
import h5py
import scann

flags.DEFINE_string("suffix", "", "Output file name suffix.")
flags.DEFINE_string("output_path", None, "Path to outputs.")
flags.DEFINE_string("input_path", None, "Input path to h5py files.")
flags.DEFINE_integer("num_vec_per_passage", 1, "Num of vectors per passage.")
flags.DEFINE_bool("brute_force", True,
                  "If true run exact search, otherwise approximate.")
flags.DEFINE_integer("num_leaves_to_search", 2000,
                     "SCANN parameter, used when brute_force==False.")
flags.DEFINE_integer("num_leaves", 5000,
                     "SCANN parameter, used when brute_force==False.")
flags.DEFINE_integer("num_neighbors", 1000, "Number of neighbors to retrieve.")
flags.DEFINE_bool("write_tsv", True, "Write output to tsv format.")
flags.DEFINE_bool("write_json", False, "Write output to json file.")

FLAGS = flags.FLAGS


def write_result_to_tsv(neighbors, query_ids, passage_ids, num_neighbors,
                        outfn):
  """write results to tsv file."""

  strings = []
  for i in range(len(query_ids)):
    passage_set = set()
    for j in range(num_neighbors):
      passage_idx = passage_ids[neighbors[i][j]]
      if passage_idx in passage_set:
        continue
      passage_set.add(passage_idx)
      strings.append("\t".join(
          [str(query_ids[i]),
           str(passage_idx),
           str(len(passage_set))]))
      if len(passage_set) >= FLAGS.num_neighbors:
        break

  with open(outfn, "w") as fid:
    fid.write("\n".join(strings))
  print("Write tsv output to: " + outfn)


def write_result_to_json(neighbors, query_ids, passage_ids, distances,
                         num_neighbors, outfn):
  """write results to json file."""
  results = {}
  for i in range(len(query_ids)):
    passage_set = set()
    results[str(query_ids[i])] = []
    for j in range(num_neighbors):
      passage_idx = passage_ids[neighbors[i][j]]
      if passage_idx in passage_set:
        continue
      passage_set.add(passage_idx)
      results[str(query_ids[i])].append(
          [str(passage_idx), float(distances[i][j])])
      if len(passage_set) >= FLAGS.num_neighbors:
        break

  with open(outfn, "w") as fid:
    json.dump(results, fid)
  print("Write json output to: " + outfn)


def main(_):
  query_ids_fn = FLAGS.input_path + "/queries_" + FLAGS.suffix + "_ids.h5py"
  passage_ids_fn = FLAGS.input_path + "/passage_ids.h5py"
  queries = h5py.File(query_ids_fn, "r")
  query_ids = queries["ids"][:]
  passages = h5py.File(passage_ids_fn, "r")
  passage_ids = passages["ids"][:]
  num_neighbors = FLAGS.num_vec_per_passage * FLAGS.num_neighbors
  neighbors_path = FLAGS.output_path + "/neighbors_" + FLAGS.suffix + ".h5py"
  scores_path = FLAGS.output_path + "/scores_" + FLAGS.suffix + ".h5py"
  if not os.path.isfile(neighbors_path):
    query_encoding_fn = FLAGS.input_path + "/queries_" + FLAGS.suffix + "_encodings.h5py"
    queries = h5py.File(query_encoding_fn, "r")
    query_encodings = queries["encodings"][:]
    passage_encoding_fn = FLAGS.input_path + "/passage_encodings.h5py"
    passages = h5py.File(passage_encoding_fn, "r")
    passage_encodings = passages["encodings"][:]
    print("Number of queries: " + str(query_ids.shape[0]))
    print("Number of passages: " + str(passage_ids.shape[0]))

    start = time.time()
    if FLAGS.brute_force:
      print("Start indexing (exact search)")
      searcher = scann.ScannBuilder(
          passage_encodings, num_neighbors,
          "dot_product").score_brute_force().create_pybind()
    else:
      print("Start indexing (approximate search)")
      searcher = scann.ScannBuilder(
          passage_encodings, num_neighbors, "dot_product").tree(
              num_leaves=FLAGS.num_leaves,
              num_leaves_to_search=FLAGS.num_leaves_to_search,
              training_sample_size=passage_encodings.shape[0]).score_ah(
                  2, anisotropic_quantization_threshold=0.2).reorder(
                      num_neighbors).create_pybind()

    end = time.time()
    print("Indexing Time:", end - start)
    start = time.time()
    neighbors, distances = searcher.search_batched_parallel(query_encodings)
    end = time.time()
    print("Search Time:", end - start)

    h5f = h5py.File(neighbors_path, "w")
    h5f.create_dataset("neighbors", data=neighbors)
    h5f.close()
    h5f = h5py.File(scores_path, "w")
    h5f.create_dataset("scores", data=distances)
    h5f.close()

  else:
    print("Neighbors file exists: " + neighbors_path)
    neighbors = h5py.File(neighbors_path, "r")["neighbors"][:]
    print("Scores file exists: " + scores_path)
    distances = h5py.File(scores_path, "r")["scores"][:]

  if FLAGS.write_tsv:
    output_tsv_fn = FLAGS.output_path + "/neighbors_" + FLAGS.suffix + ".tsv"
    write_result_to_tsv(neighbors, query_ids, passage_ids, num_neighbors,
                        output_tsv_fn)
  if FLAGS.write_json:
    output_json_fn = FLAGS.output_path + "/neighbors_" + FLAGS.suffix + ".json"
    write_result_to_json(neighbors, query_ids, passage_ids, distances,
                         num_neighbors, output_json_fn)


if __name__ == "__main__":
  app.run(main)
