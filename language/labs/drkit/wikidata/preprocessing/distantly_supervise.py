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
"""Script to distantly supervise Wikipedia with Wikidata using SLING."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import random
import time

import sling
import sling.flags as flags
import sling.task.silver as silver
import sling.task.workflow as workflow

random.seed(111)

MIN_LEN = 10
TOP_K = 5

## Required parameters
flags.define("--output", default="distant", help="Output directory.")

flags.define("--max_n", default=None, help="Number of articles to preprocess.")

flags.define(
    "--wiki_split", default=None, help="Which wiki split to preprocess (0-9).")


class SlingExtractor(object):
  """Extract passages from Wikipedia which mention a fact from Wikidata."""

  def load_kb(self, base_dir):
    """Load sling KB."""
    print("loading and indexing kb...")
    start = time.time()
    self.kb = sling.Store()
    self.kb.load(os.path.join(base_dir, "e/wiki/kb.sling"))
    self.kb.freeze()
    commons = sling.Store()
    commons.freeze()
    self.store = sling.Store(commons)
    self.cal = sling.Calendar(commons)
    self.extract_property_names()
    self.all_aliases = self.fetch_aliases(
        os.path.join(base_dir, "e/wiki/en/aliases-0000%d-of-00010.rec"))
    print("loading took", (time.time() - start), "sec")

  def fetch_aliases(self, alias_file_patterns):
    print("Pre-fetching all raw aliases...")
    all_aliases = {}
    for ii in range(10):
      fname = alias_file_patterns % ii
      print("reading from %s..." % fname)
      db = sling.RecordReader(fname)
      for aid, als in db:
        all_aliases[aid.decode("utf-8", errors="ignore")] = als
    return all_aliases

  def extract_property_names(self):
    """Store names of properties in a dict."""
    print("storing property names...")
    start = time.time()
    self.property_names = collections.defaultdict(list)
    for frame in self.kb:
      if "id" in frame and frame.id.startswith("P"):
        self.property_names[frame.id].append(frame.name)
    print("found", str(len(self.property_names)), "properties")
    print("took", (time.time() - start), "sec")

  def load_corpus(self, corpus_file):
    """Load self.corpus."""
    print("loading the corpus...")
    self.corpus = sling.Corpus(corpus_file)
    self.commons = sling.Store()
    self.docschema = sling.DocumentSchema(self.commons)
    self.commons.freeze()

  def annotate_corpus(self, unannotated_file, annotated_file):
    """Run silver annotations from SLING."""
    if os.path.exists(annotated_file):
      return
    labeler = silver.SilverWorkflow("silver")
    # labeler = entity.EntityWorkflow("wiki-label")
    unannotated = labeler.wf.resource(
        unannotated_file, format="records/document")
    annotated = labeler.wf.resource(annotated_file, format="records/document")
    labeler.silver_annotation(indocs=unannotated, outdocs=annotated)
    # labeler.label_documents(indocs=unannotated, outdocs=annotated)
    workflow.run(labeler.wf)

  def get_linked_entity(self, mention):
    """Returns the string ID of the linked entity for this mention."""
    if "evokes" not in mention.frame:
      return None
    if not isinstance(mention.frame["evokes"], sling.Frame):
      return None
    if "is" in mention.frame["evokes"]:
      if not isinstance(mention.frame["evokes"]["is"], sling.Frame):
        if ("isa" in mention.frame["evokes"] and
            mention.frame["evokes"]["isa"].id == "/w/time" and
            isinstance(mention.frame["evokes"]["is"], int)):
          return mention.frame["evokes"]["is"]
        else:
          return None
      else:
        return mention.frame["evokes"]["is"].id
    return mention.frame["evokes"].id

  def get_frame_id(self, frame):
    """Returns the WikiData identifier for a property or entity."""
    if "id" in frame:
      return frame.id
    if "is" in frame:
      if not isinstance(frame["is"], sling.Frame):
        return None
      if "id" in frame["is"]:
        return frame["is"].id
    return None

  def get_date_property(self, prop, tail):
    """Returns date if property accepts '/w/time' as target."""
    if "target" not in prop:
      return None
    if prop.target.id != "/w/time":
      return None
    prop_id = self.get_frame_id(prop)
    if isinstance(tail, int):
      return (prop_id, tail)
    elif (isinstance(tail, sling.Frame) and "is" in tail and
          isinstance(tail["is"], int)):
      return (prop_id, tail["is"])
    return None

  def get_canonical_property(self, prop, tail):
    """Returns true if the prop and tail are canonical WikiData properties."""
    if not isinstance(prop, sling.Frame) or not isinstance(tail, sling.Frame):
      return None
    prop_id = self.get_frame_id(prop)
    tail_id = self.get_frame_id(tail)
    if prop_id is None:
      return None
    if tail_id is None:
      return None
    if not prop_id.startswith("P") or not tail_id.startswith("Q"):
      return None
    return (prop_id, tail_id)

  def get_name(self, x):
    """Return name for given wikidata id."""
    if x is None:
      return None
    if isinstance(x, int):
      date = sling.Date(x)
      return self.cal.str(date)
    elif not x.startswith("Q"):
      return None
    if isinstance(self.kb[x].name, bytes):
      return self.kb[x].name.decode("utf-8", errors="ignore")
    else:
      return self.kb[x].name

  def get_aliases(self, x):
    """Return aliases for given wikidata id."""
    if isinstance(x, int):
      return {}
    aliases = collections.defaultdict(int)
    try:
      alias_raw = self.all_aliases[x]
    except KeyError:
      print("did not find alias:", x)
      return {}
    als = self.store.parse(alias_raw)
    for _, alias in als:
      al_name = alias.name
      if isinstance(al_name, bytes):
        al_name = al_name.decode("utf-8", errors="ignore")
      aliases[al_name.lower()] += alias.count
    sorted_aliases = sorted(aliases.items(), key=lambda x: x[1], reverse=True)
    return {x[0]: x[1] for x in sorted_aliases[:TOP_K]}

  def print_relation(self, relation):
    """Print the distantly supervised relation mention."""
    print(relation[2], "::", ",".join(relation[3]), "::", relation[1], "::",
          relation[0])

  def serialize_para(self, document, tok_to_char_offset, span, mentions, doc_id,
                     para_id):
    """Create JSON object for input paragraph."""
    context = " ".join([tt.word for tt in document.tokens[span[0]:span[1]]])
    my_offsets = {}
    begin = tok_to_char_offset[span[0]]
    for ix in range(span[0], span[1]):
      my_offsets[ix] = tok_to_char_offset[ix] - begin
    mention_objs = []
    for m, ent in mentions:
      mention_objs.append({
          "wikidata_id": ent,
          "name": self.get_name(ent),
          "aliases": self.get_aliases(ent),
          "start": my_offsets[m.begin],
          "text": " ".join(tt.word for tt in document.tokens[m.begin:m.end]),
      })
    return json.dumps({
        "id": para_id,
        "wikidata_id": doc_id,
        "context": context,
        "mentions": mention_objs,
    })

  def serialize_query(self, global_para_id, ment_index_to_para_index, subj_id,
                      subj_mentions, property_id, tail_id, tail_mention,
                      ctx_type):
    """Create JSON object with distantly supervised fact."""
    subject = {
        "wikidata_id": subj_id,
        "aliases": self.get_aliases(subj_id),
        "mentions": [ment_index_to_para_index[ix] for ix in subj_mentions],
    }
    if property_id not in self.property_names:
      print("did not find", property_id, "in names")
    relation = {
        "wikidata_id": property_id,
        "text": self.property_names.get(property_id, property_id)
    }
    if tail_id is None:
      obj = None
      id_ = subj_id + "_" + property_id + "_None"
    else:
      obj = {
          "wikidata_id": tail_id,
          "aliases": self.get_aliases(tail_id),
          "mention": ment_index_to_para_index[tail_mention],
      }
      id_ = subj_id + "_" + property_id + "_" + str(tail_id)
    serialized = json.dumps({
        "id": id_,
        "para_id": global_para_id,
        "subject": subject,
        "relation": relation,
        "object": obj,
        "context_type": ctx_type,
    })
    return serialized

  def serialize_fact(self, document, tok_to_char_offset, ctx_span, subj_id,
                     subj_mentions, property_id, tail_id, tail_mention,
                     ctx_type):
    """Create JSON object with distantly supervised fact."""
    context = " ".join(
        [tt.word for tt in document.tokens[ctx_span[0]:ctx_span[1]]])
    my_offsets = {}
    begin = tok_to_char_offset[ctx_span[0]]
    for ix in range(ctx_span[0], ctx_span[1]):
      my_offsets[ix] = tok_to_char_offset[ix] - begin
    all_mentions = []
    for m in subj_mentions:
      all_mentions.append({
          "start": my_offsets[m.begin],
          "text": " ".join(tt.word for tt in document.tokens[m.begin:m.end])
      })
    subject = {
        "wikidata_id": subj_id,
        "aliases": self.get_aliases(subj_id),
        "mentions": all_mentions,
    }
    if property_id not in self.property_names:
      print("did not find", property_id, "in names")
    relation = {
        "wikidata_id": property_id,
        "text": self.property_names.get(property_id, property_id)
    }
    if tail_id is None:
      obj = None
      id_ = subj_id + "_" + property_id + "_None"
    else:
      obj = {
          "wikidata_id": tail_id,
          "aliases": self.get_aliases(tail_id),
          "mention": {
              "start":
                  my_offsets[tail_mention.begin],
              "text":
                  " ".join(tt.word for tt in
                           document.tokens[tail_mention.begin:tail_mention.end])
          }
      }
      id_ = subj_id + "_" + property_id + "_" + str(tail_id)
    serialized = json.dumps({
        "id": id_,
        "context": context,
        "subject": subject,
        "relation": relation,
        "object": obj,
        "context_type": ctx_type,
    })
    return serialized

  def init_stats(self):
    """Initialize dicts to hold relation counts."""
    self.relation_stats = {
        "sentences": collections.defaultdict(int),
        "paragraphs": collections.defaultdict(int),
        "entity negatives": collections.defaultdict(int)
    }

  def link_documents(self,
                     max_n=None,
                     fact_out_file="/tmp/facts.json",
                     qry_out_file="/tmp/queries.json",
                     para_out_file="/tmp/paragraphs.json",
                     filter_subjects=None,
                     exclude_subjects=None):
    """Load n documents and link them to facts."""
    start = time.time()
    fout = open(fact_out_file, "w")
    fq_out, fp_out = open(qry_out_file, "w"), open(para_out_file, "w")
    seen_articles = set()
    total_paras = 0
    for n, (doc_id, doc_raw) in enumerate(self.corpus.input):
      doc_id = str(doc_id, "utf-8")
      if n % 1000 == 0:
        print("processed", n, "items in %.1f" % (time.time() - start), "sec")
      if max_n is not None and random.uniform(0, 1) > (float(max_n) / 550000):
        continue
      if filter_subjects is not None and doc_id not in filter_subjects:
        continue
      if exclude_subjects is not None and doc_id in exclude_subjects:
        continue
      # get kb items
      seen_articles.add(doc_id)
      kb_item = self.kb[doc_id]
      tail_entities = {}
      all_properties = []
      for prop, tail in kb_item:
        tup = self.get_canonical_property(prop, tail)
        if tup is None:
          tup = self.get_date_property(prop, tail)
          if tup is None:
            continue
        tail_entities[tup[1]] = tup[0]
        all_properties.append(tup[0])
      store = sling.Store(self.commons)
      document = sling.Document(store.parse(doc_raw), store, self.docschema)
      if not document.tokens:
        print("Skipping %s No tokens." % (doc_id))
        continue
      # build token maps
      tok_to_sent_id, tok_to_para_id, sent_to_span, para_to_span = {}, {}, {}, {}
      tok_to_char_offset = {}
      offset = 0
      sent_begin = para_begin = 0
      sent_id = para_id = 0
      for ii, token in enumerate(document.tokens):
        if ii > 0 and token.brk == 4:
          para_to_span[para_id] = (para_begin, ii)
          sent_to_span[sent_id] = (sent_begin, ii)
          para_id += 1
          sent_id += 1
          sent_begin = para_begin = ii
        elif ii > 0 and token.brk == 3:
          sent_to_span[sent_id] = (sent_begin, ii)
          sent_id += 1
          sent_begin = ii
        tok_to_sent_id[ii] = sent_id
        tok_to_para_id[ii] = para_id
        tok_to_char_offset[ii] = offset
        offset += len(token.word) + 1
      para_to_span[para_id] = (para_begin, len(document.tokens))
      sent_to_span[sent_id] = (sent_begin, len(document.tokens))
      # find subjects
      sent_to_subj, para_to_subj = (collections.defaultdict(list),
                                    collections.defaultdict(list))
      para_to_ment = collections.defaultdict(list)
      ment_to_para_index = {}
      mentid_to_linked_entity = {}
      sorted_mentions = sorted(document.mentions, key=lambda m: m.begin)
      for ii, mention in enumerate(sorted_mentions):
        if tok_to_sent_id[mention.begin] != tok_to_sent_id[mention.end - 1]:
          continue
        linked_entity = self.get_linked_entity(mention)
        mentid_to_linked_entity[ii] = linked_entity
        para_id = tok_to_para_id[mention.begin]
        para_to_ment[para_id].append((mention, linked_entity))
        ment_to_para_index[ii] = len(para_to_ment[para_id]) - 1
        if linked_entity == doc_id:
          sent_id = tok_to_sent_id[mention.begin]
          sent_to_subj[sent_id].append(ii)
          para_to_subj[para_id].append(ii)

      # save paragraphs
      local_to_global_para = {}
      for para_id, para_span in para_to_span.items():
        if para_span[1] - para_span[0] < MIN_LEN:
          continue
        if len(para_to_ment[para_id]) <= 1:
          continue
        local_to_global_para[para_id] = total_paras
        fp_out.write(
            self.serialize_para(document, tok_to_char_offset, para_span,
                                para_to_ment[para_id], doc_id, total_paras) +
            "\n")
        total_paras += 1

      # find tails
      seen_properties = {}
      for ii, mention in enumerate(sorted_mentions):
        # first look for sentence matches
        linked_entity = mentid_to_linked_entity[ii]
        if linked_entity == doc_id:
          continue
        if linked_entity in tail_entities:
          if tail_entities[linked_entity] in seen_properties:
            continue
          my_sent = tok_to_sent_id[mention.begin]
          my_para = tok_to_para_id[mention.begin]
          para_span = para_to_span[my_para]
          if my_para not in local_to_global_para:
            continue
          if my_sent in sent_to_subj:
            # sent_span = sent_to_span[my_sent]
            fq_out.write(
                self.serialize_query(
                    local_to_global_para[my_para], ment_to_para_index, doc_id,
                    sent_to_subj[my_sent], tail_entities[linked_entity],
                    linked_entity, ii, "sentence") + "\n")
            subj_mentions = [
                para_to_ment[my_para][ment_to_para_index[mm]][0]
                for mm in sent_to_subj[my_sent]
            ]
            fout.write(
                self.serialize_fact(document, tok_to_char_offset, para_span,
                                    doc_id, subj_mentions,
                                    tail_entities[linked_entity], linked_entity,
                                    mention, "sentence") + "\n")
            seen_properties[tail_entities[linked_entity]] = my_para
            self.relation_stats["sentences"][tail_entities[linked_entity]] += 1

      for ii, mention in enumerate(sorted_mentions):
        # next look for paragraph matches
        linked_entity = mentid_to_linked_entity[ii]
        if linked_entity == doc_id:
          continue
        if linked_entity in tail_entities:
          if tail_entities[linked_entity] in seen_properties:
            continue
          my_para = tok_to_para_id[mention.begin]
          para_span = para_to_span[my_para]
          if my_para not in local_to_global_para:
            continue
          if my_para in para_to_subj:
            fq_out.write(
                self.serialize_query(
                    local_to_global_para[my_para], ment_to_para_index, doc_id,
                    para_to_subj[my_para], tail_entities[linked_entity],
                    linked_entity, ii, "paragraph") + "\n")
            subj_mentions = [
                para_to_ment[my_para][ment_to_para_index[mm]][0]
                for mm in para_to_subj[my_para]
            ]
            fout.write(
                self.serialize_fact(document, tok_to_char_offset, para_span,
                                    doc_id, subj_mentions,
                                    tail_entities[linked_entity], linked_entity,
                                    mention, "paragraph") + "\n")
            seen_properties[tail_entities[linked_entity]] = my_para
            self.relation_stats["paragraphs"][tail_entities[linked_entity]] += 1

      # add negatives
      max_neg = len(seen_properties)
      num_neg = 0
      all_para_id = list(para_to_subj.keys())
      if not all_para_id:
        continue
      for tail, prop in tail_entities.items():
        if num_neg == max_neg:
          break
        if prop in seen_properties:
          continue
        random_para_id = random.choice(all_para_id)
        random_para_span = para_to_span[random_para_id]
        subj_mentions = [
            para_to_ment[random_para_id][ment_to_para_index[mm]][0]
            for mm in para_to_subj[random_para_id]
        ]
        fout.write(
            self.serialize_fact(document, tok_to_char_offset, random_para_span,
                                doc_id, subj_mentions, prop, None, None,
                                "entity negative") + "\n")
        num_neg += 1
        seen_properties[prop] = None
        self.relation_stats["entity negatives"][prop] += 1

    fout.close()
    fq_out.close()
    fp_out.close()
    print("Sentences -- ", sum(self.relation_stats["sentences"].values()))
    print(" :: ".join(
        "%s:%d" % (k, v) for k, v in self.relation_stats["sentences"].items()))
    print("Paragraphs -- ", sum(self.relation_stats["paragraphs"].values()))
    print(" :: ".join(
        "%s:%d" % (k, v) for k, v in self.relation_stats["paragraphs"].items()))


if __name__ == "__main__":
  flags.parse()
  # workflow.startup()
  s = SlingExtractor()
  s.load_kb(flags.arg.data)
  s.init_stats()
  if flags.arg.wiki_split is None:
    rn = range(0, 10)
  else:
    rn = [flags.arg.wiki_split]
  for i in rn:
    print("Processing shard %d" % i)
    t_st = time.time()
    # s.annotate_corpus(
    #     unannotated_file=os.path.join(
    #         flags.arg.data,
    #         "e/wiki/en/documents-0000%d-of-00010.rec" % i),
    #     annotated_file=os.path.join(
    #         flags.arg.data,
    #         "e/wiki/en/labeled-documents-0000%d-of-00010.rec" % i))
    s.load_corpus(
        corpus_file=os.path.join(flags.arg.data,
                                 "e/silver/en/silver-0000%d-of-00010.rec" % i))
    s.link_documents(
        fact_out_file=os.path.join(flags.arg.data, flags.arg.output,
                                   "facts-0000%d-of-00010.json" % i),
        para_out_file=os.path.join(flags.arg.data, flags.arg.output,
                                   "paragraphs-0000%d-of-00010.json" % i),
        qry_out_file=os.path.join(flags.arg.data, flags.arg.output,
                                  "queries-0000%d-of-00010.json" % i),
        max_n=flags.arg.max_n)
    print("Processing took %.1f secs" % (time.time() - t_st))
  # workflow.shutdown()
