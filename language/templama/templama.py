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
"""Script to construct the TempLAMA dataset.

This relies on the preprocessed Wikipedia and Wikidata dumps from SLING:
https://github.com/ringgaard/sling/blob/master/doc/guide/wikiflow.md
Please follow the instructions above before running this code.

This also relies on first identifying the Wikidata facts which have start/end
time qualifiers. To do that, please first run the `sling2facts.py` script
included in this directory.

Provide the extracted kb.cfacts file from the above script as `--facts_file`.
Also provide a path to the sling KB (`--sling_kb_file`) and the sling mapping
from wikidata to English wikipedia (`--sling_wiki_mapping_file`). These
are typically found at the following locations under the SLING folder:
<sling_root>/local/data/e/wiki/kb.sling
<sling_root>/local/data/e/wiki/en/mapping.sling

You can specify the date range between which the queries should be
constructed by using the `--min_year` and `--max_year` options below.
"""

import collections
import csv
import datetime
import json
import os
import random

from absl import app
from absl import flags
from absl import logging
import sling
import tensorflow as tf
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("out_dir", None, "Path to store constructed queries.")
flags.DEFINE_string(
    "facts_file", None,
    "File containing facts with qualifiers extracted from `sling2facts.py`.")
flags.DEFINE_string("sling_kb_file", None, "SLING file containing wikidata KB.")
flags.DEFINE_string(
    "sling_wiki_mapping_file", None,
    "SLING file containing mapping from QID to english wikipedia pages.")
flags.DEFINE_integer(
    "min_year", 2010,
    "Starting year to construct queries from. Only facts which have a start / "
    "end date after this will be considered.")
flags.DEFINE_integer("max_year", 2020,
                     "Ending year to construct queries up till.")
flags.DEFINE_integer(
    "max_subject_per_relation", 1000,
    "Maximum number of subjects to retain per relation. Subjects are sorted "
    "based on popularity before filtering.")
flags.DEFINE_float("train_frac", 0.2,
                   "Fraction of queries to hold out for training set.")
flags.DEFINE_float("val_frac", 0.1,
                   "Fraction of queries to hold out for validation set.")

random.seed(42)
Y_TOK = "_X_"
WIKI_PRE = "/wp/en/"


def _datetup2int(date):
  """Convert (year, month, day) to integer representation.

  Args:
    date: Tuple of (year, month, day).

  Returns:
    an int of year * 1e4 + month * 1e2 + day.
  """
  dint = date[0] * 1e4
  dint += date[1] * 1e2 if date[1] else 0
  dint += date[2] if date[2] else 0
  return dint


def date_in_interval(date, start, end):
  """Check if date is within start and end.

  Args:
    date: Tuple of (year, month, day).
    start: Start date (year, month, day).
    end: End date (year, month, day).

  Returns:
    a bool of whether start <= date <= end.
  """
  date_int = _datetup2int(date)
  start_int = _datetup2int(start) if start else 0
  end_int = _datetup2int(end) if end else 21000000
  return date_int >= start_int and date_int <= end_int


def parse_date(date_str):
  """Try to parse date from string.

  Args:
    date_str: String representation of the date.

  Returns:
    date: Tuple of (year, month, day).
  """
  date = None
  try:
    if len(date_str) == 4:
      date_obj = datetime.datetime.strptime(date_str, "%Y")
      date = (date_obj.year, None, None)
    elif len(date_str) == 6:
      date_obj = datetime.datetime.strptime(date_str, "%Y%m")
      date = (date_obj.year, date_obj.month, None)
    elif len(date_str) == 8:
      date_obj = datetime.datetime.strptime(date_str, "%Y%m%d")
      date = (date_obj.year, date_obj.month, date_obj.day)
  except ValueError:
    pass
  if date is not None and date[0] > 2100:
    # Likely an error
    date = None
  return date


def load_sling_mappings(sling_kb_file, sling_wiki_mapping_file):
  """Loads entity names, number of facts and wiki page titles from SLING.

  Args:
    sling_kb_file: kb.sling file generated from SLING wikidata processor.
    sling_wiki_mapping_file: mapping.sling file generated from SLING 'en'
      wikipedia processor.

  Returns:
    qid_names: dict mapping wikidata QIDs to canonical names.
    qid_mapping: dict mapping wikidata QIDs to wikipedia page titles.
    qid_numfacts: dict mapping wikidata QIDs to number of facts.
  """
  # Load QID names.
  logging.info("Extracting entity names and num-facts from SLING KB.")
  commons = sling.Store()
  commons.load(sling_kb_file)
  commons.freeze()
  qid_names = {}
  qid_numfacts = {}
  total = 0
  for f in commons:
    total += 1
    if "name" in f:
      if isinstance(f.name, sling.String):
        qid_names[f.id] = f.name.text()
      elif isinstance(f.name, bytes):
        qid_names[f.id] = f.name.decode("utf-8", errors="ignore")
      elif isinstance(f.name, str):
        qid_names[f.id] = f.name
      else:
        logging.warn("Could not read name of type %r", type(f.name))
    ln = len(f)
    qid_numfacts[f.id] = ln
  logging.info("Processed %d QIDs out of %d", len(qid_names), total)
  # Load QID mapping.
  logging.info("Extracting entity mapping to Wikipedia from SLING.")
  commons = sling.Store()
  commons.load(sling_wiki_mapping_file)
  commons.freeze()
  qid_mapping = {}
  for f in commons:
    try:
      if "/w/item/qid" in f:
        pg = f.id[len(WIKI_PRE):] if f.id.startswith(WIKI_PRE) else f.id
        qid_mapping[f["/w/item/qid"].id] = pg
    except UnicodeDecodeError:
      continue
  logging.info("Extracted %d mappings", len(qid_mapping))
  return qid_names, qid_mapping, qid_numfacts


def read_facts(facts_file, qid_mapping, min_year):
  """Loads facts and filters them using simple criteria.

  Args:
    facts_file: File containing wikidata facts with qualifiers.
    qid_mapping: dict mapping wikidata QIDs to wikipedia page titles.
    min_year: An int. Only facts with a start / end year greater than this will
      be kept.

  Returns:
    all_facts: list of tuples, where each tuple is a fact with
      (relation, subject, object, start, end).
  """
  logging.info("Reading facts from %s", facts_file)
  all_facts = []
  with tf.io.gfile.GFile(facts_file) as f:
    for line in tqdm(f):
      fact = line.strip().split("\t")
      # Skip boring properties.
      if not fact[0].startswith("P"):
        continue
      # Skip instance of facts.
      if fact[0] == "P31":
        continue
      # Skip facts where object is not an entity.
      if not fact[2].startswith("Q"):
        continue
      # Skip facts whose subject and objects are not wiki pages.
      if fact[1] not in qid_mapping or fact[2] not in qid_mapping:
        continue
      # Get date qualifiers.
      start, end = None, None
      for qual in fact[3:]:
        if not qual:
          continue
        elems = qual.split("=")
        # Skip inherited qualifier.
        if elems[0].endswith("*"):
          continue
        if len(elems) != 2:
          continue
        if elems[0].startswith("P580"):
          start = parse_date(elems[1])
        elif elems[0].startswith("P582"):
          end = parse_date(elems[1])
      if start is None and end is None:
        continue
      # Skip facts whose start and end are both before min_date.
      if ((start is None or start[0] < min_year) and
          (end is None or end[0] < min_year)):
        continue
      all_facts.append(fact[:3] + [start, end])
  logging.info("Loaded total %d facts", len(all_facts))
  return all_facts


def read_templates():
  """Loads relation-specific templates from `templates.csv`.

  Returns:
    a dict mapping relation IDs to string templates.
  """
  my_path = os.path.dirname(os.path.realpath(__file__))
  template_file = os.path.join(my_path, "templates.csv")
  logging.info("Reading templates from %s", template_file)
  reader = csv.reader(tf.io.gfile.GFile(template_file))
  headers = next(reader, None)
  data = collections.defaultdict(list)
  for row in reader:
    for h, v in zip(headers, row):
      data[h].append(v)
  templates = dict(zip(data["Wikidata ID"], data["Template"]))
  logging.info("\n".join("%s: %s" % (k, v) for k, v in templates.items()))
  return templates


def resolve_objects(facts):
  """Combine consecutive objects across years into one fact.

  Args:
    facts: A list of fact tuples.

  Returns:
    a list of fact tuples with consecutive facts with the same object merged.
  """

  def _datekey(fact):
    start = _datetup2int(fact[3]) if fact[3] else 0
    end = _datetup2int(fact[4]) if fact[4] else 21000000
    return (start, end)

  # First sort by start time and then by end time.
  sorted_facts = sorted(facts, key=_datekey)
  # Merge repeated objects into one.
  out_facts = [sorted_facts[0]]
  for fact in sorted_facts[1:]:
    if (fact[2] == out_facts[-1][2] and fact[3] != fact[4] and
        out_facts[-1][3] != out_facts[-1][4]):
      out_facts[-1][4] = fact[4]
    else:
      out_facts.append(fact)
  return out_facts


def _map_years_to_objects(facts, qid_numfacts, min_year, max_year):
  """Map each year between min, max to the corresponding object in facts.

  Args:
    facts: a list of facts with the same subject and relation.
    qid_numfacts: a dict mapping wikidata QIDs to number of facts.
    min_year: an int, starting year to map.
    max_year: an int, ending year to map.

  Returns:
    year2obj: a dict mapping each year between (min_year, max_year) to the
      corresponding most 'popular' object for that year.
  """
  year2obj = {}
  numfacts = lambda x: qid_numfacts.get(x, 0)
  for f in facts:
    min_ = f[3][0] if f[3] is not None else min_year
    max_ = f[4][0] if f[4] is not None else max_year
    min_ = max(min_, min_year)
    max_ = min(max_, max_year)
    for yr in range(min_, max_ + 1):
      if yr in year2obj:
        # Keep the more popular object.
        if numfacts(year2obj[yr]) < numfacts(f[2]):
          year2obj[yr] = f[2]
      else:
        year2obj[yr] = f[2]
  return year2obj


def _build_example(query):
  """Creates a tf.Example for prediction with T5 from the input query.

  Args:
    query: a dict mapping query features to their values.

  Returns:
    a tf.train.Example consisting of the query features.
  """
  # Inputs and targets.
  inp = query["query"].encode("utf-8")
  trg = query["answer"]["name"].encode("utf-8")
  # Metadata.
  id_ = query["id"].encode("utf-8")
  recent = query["most_recent_answer"]["name"].encode("utf-8")
  frequent = query["most_frequent_answer"]["name"].encode("utf-8")
  rel = query["relation"].encode("utf-8")
  # Construct TFRecord.
  feature = {
      "id":
          tf.train.Feature(bytes_list=tf.train.BytesList(value=[id_])),
      "date":
          tf.train.Feature(
              int64_list=tf.train.Int64List(value=[int(query["date"])])),
      "relation":
          tf.train.Feature(bytes_list=tf.train.BytesList(value=[rel])),
      "query":
          tf.train.Feature(bytes_list=tf.train.BytesList(value=[inp])),
      "answer":
          tf.train.Feature(bytes_list=tf.train.BytesList(value=[trg])),
      "most_frequent_answer":
          tf.train.Feature(bytes_list=tf.train.BytesList(value=[frequent])),
      "most_recent_answer":
          tf.train.Feature(bytes_list=tf.train.BytesList(value=[recent])),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def create_queries(out_dir, all_facts, templates, qid_names, qid_numfacts,
                   min_year, max_year, train_frac, val_frac,
                   max_subject_per_relation):
  """Construct queries for most popular subjects for each relation.

  Args:
    out_dir: Path to store all queries as well as yearly slices.
    all_facts: a list of facts.
    templates: a dict mapping relation IDs to templates.
    qid_names: dict mapping wikidata QIDs to canonical names.
    qid_numfacts: dict mapping wikidata QIDs to number of facts.
    min_year: an int, starting year to map.
    max_year: an int, ending year to map.
    train_frac: a float, fraction of subjects to reserve for the train set.
    val_frac: a float, fraction of subjects to reserve for the val set.
    max_subject_per_relation: number of subjects to keep per relation.
  """

  def _create_entity_obj(qid):
    return {"wikidata_id": qid, "name": qid_names[qid]}

  def _create_implicit_query(subj, tmpl):
    return tmpl.replace("<subject>", qid_names[subj]).replace("<object>", Y_TOK)

  def _most_frequent_answer(year2obj):
    counts = collections.defaultdict(int)
    for _, obj in year2obj.items():
      counts[obj] += 1
    return max(counts.items(), key=lambda x: x[1])[0]

  def _most_recent_answer(yr2obj):
    recent = max(yr2obj.keys())
    return yr2obj[recent]

  # Group by relation and by sort by subject
  logging.info("Keeping only facts with templates.")
  rel2subj = {}
  for fact in tqdm(all_facts):
    if fact[0] not in templates:
      continue
    if fact[0] not in rel2subj:
      rel2subj[fact[0]] = {}
    if fact[1] not in rel2subj[fact[0]]:
      rel2subj[fact[0]][fact[1]] = []
    rel2subj[fact[0]][fact[1]].append(fact)

  logging.info("Sorting subjects by 'popularity' resolving multiple objects.")
  sorted_rel2subj = {}
  for relation in rel2subj:
    sorted_subjs = sorted(
        rel2subj[relation].keys(),
        key=lambda x: qid_numfacts.get(x, 0),
        reverse=True)
    sorted_rel2subj[relation] = [
        (s, resolve_objects(rel2subj[relation][s])) for s in sorted_subjs
    ]

  logging.info("Keep only subjects with multiple objects.")
  total_facts = 0
  filt_rel2subj = {}
  for rel, subj2facts in sorted_rel2subj.items():
    filt_subj2facts = list(filter(lambda x: len(x[1]) > 1, subj2facts))
    if filt_subj2facts:
      filt_rel2subj[rel] = filt_subj2facts
      total_facts += sum([len(f) for _, f in filt_rel2subj[rel]])
  logging.info("# facts after filtering = %d", total_facts)

  logging.info("Keep only %d subjects per relation, split into train/val/test",
               max_subject_per_relation)
  train_queries, val_queries, test_queries = [], [], []
  tot_queries, tot_subj = 0, 0
  for relation, subj2facts in filt_rel2subj.items():
    num_subj = 0
    for subj, facts in subj2facts:
      year2obj = _map_years_to_objects(facts, qid_numfacts, min_year, max_year)
      p = random.random()  # to decide which split this subject belongs to.
      for yr, obj in year2obj.items():
        query = {
            "query":
                _create_implicit_query(subj, templates[relation]),
            "answer":
                _create_entity_obj(obj),
            "date":
                str(yr),
            "id":
                subj + "_" + relation + "_" + str(yr),
            "most_frequent_answer":
                _create_entity_obj(_most_frequent_answer(year2obj)),
            "most_recent_answer":
                _create_entity_obj(_most_recent_answer(year2obj)),
            "relation":
                relation,
        }
        if p < train_frac:
          train_queries.append(query)
        elif p < train_frac + val_frac:
          val_queries.append(query)
        else:
          test_queries.append(query)
        tot_queries += 1
      num_subj += 1
      if num_subj == max_subject_per_relation:
        break
    logging.info("%s: # subjects = %d # train = %d # val = %d # test = %d",
                 relation, len(subj2facts), len(train_queries),
                 len(val_queries), len(test_queries))
    tot_subj += num_subj

  # Save all queries as a json.
  split2qrys = {
      "train": train_queries,
      "val": val_queries,
      "test": test_queries
  }
  tf.io.gfile.makedirs(out_dir)
  logging.info("Saving all queries to %s", out_dir)
  for split in ["train", "val", "test"]:
    with tf.io.gfile.GFile(os.path.join(out_dir, f"{split}.jsonl"), "w") as f:
      for qry in split2qrys[split]:
        f.write(json.dumps(qry) + "\n")

  # Make subdirectories and store each split.
  for year in range(min_year, max_year + 1):
    subd = os.path.join(out_dir, "yearly", str(year))
    tf.io.gfile.makedirs(subd)
    logging.info("Saving queries for %d to %s", year, subd)
    counts = collections.defaultdict(int)
    for split in ["train", "val", "test"]:
      with tf.io.TFRecordWriter(os.path.join(subd, f"{split}.tf_record")) as f:
        for qry in split2qrys[split]:
          if qry["date"] == str(year):
            f.write(_build_example(qry).SerializeToString())
            counts[split] += 1


def main(_):
  # Load relation templates.
  templates = read_templates()

  # Load entity names, number of facts and wiki page titles from SLING.
  qid_names, qid_mapping, qid_numfacts = load_sling_mappings(
      FLAGS.sling_kb_file, FLAGS.sling_wiki_mapping_file)

  # Load facts with qualifiers.
  all_facts = read_facts(FLAGS.facts_file, qid_mapping, FLAGS.min_year)

  # Create queries.
  create_queries(FLAGS.out_dir, all_facts, templates, qid_names, qid_numfacts,
                 FLAGS.min_year, FLAGS.max_year, FLAGS.train_frac,
                 FLAGS.val_frac, FLAGS.max_subject_per_relation)


if __name__ == "__main__":
  app.run(main)
