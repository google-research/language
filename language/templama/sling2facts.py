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
"""Extract a TSV file of facts from the sling KB.

This relies on the preprocessed Wikipedia and Wikidata dumps from SLING:
https://github.com/ringgaard/sling/blob/master/doc/guide/wikiflow.md

Please make sure you have installed sling and run the wikipedia
preprocessing pipeline by following the instructions above before running this
code.

Also includes code to reformulate the KB in some non-trivial ways.  Background:
relations in wikidata can be direct or qualified.  A "direct" relation r between
subject x and object y is encoded as a slot named r in the framre for x pointing
to y, where y is some Python primitive, or a frame for another wikidata entity,
like

  P158[seal image] Q1342[Pittsburgh]  'Seal of the City of Pittsburgh.svg'
  P571[inception]  Q1342[Pittsburgh]   17581127
  P421[located in time zone] Q1342[Pittsburgh] Q941023[Eastern Time Zone]

In an "indirect relation", y is a special frame marked by having a null id, and
a slot named 'is', which points to a z that is the real object.  Here y is
similar to a freebase CVT node. Other slots in y are "qualifiers" - eg slot P6
for Pittsburgh has a filler

  P6[head of government]        Q1342[Pittsburgh]       Q4910457[Bill Peduto]

with qualifiers P580[start time]=20140106 and P102[member of political
party]=Q29552[Democratic Party].  Just like in a direct relation, z can be a
frame for a wikidata entity or a Python primitive, as can the qualifiers.
(Qualifiers can also be indirect relations but this code doesn't support that.)

The core code here processes frames and extracts triples [r, x, y] where y is
EITHER the object of a direct relation OR the object of an indirect relation.
The output can either include the qualifiers, in which case they are additional
columns with values like "P580=20140106" or "P102=Q29552", or not.  There are
also a number of filtering options, that require Wikidata-specific logic to
implement.

 - keep_only_numeric_slots: only keep slots and qualifiers with numeric values,
  and discard the rest (for Einat's comparison project)

 - keep_only_date_slots: only keep slots and qualifiers with date values
  which can be parsed using the datetime module

 - skip_identifier_rels: skip (don't output) slots with values that are external
   identifiers, like Freebase Ids.

 - skip_nonentity_objects: skip triples with object entities that are not
  wikidata entities

 - skip_empty_objects: skip triples with object entities that are neither
  wikidata entities nor numbers

 - skip_categories: skip triples with objects that are Wikidata Categories.

 - skip_nonentity_qualifiers: skip qualifier pairs with values that are
  not wikidata entities, eg "P580[start time]=20140106"

 - skip_qualifiers: don't output any qualifiers (so output has only three
  columns: rel, subject and object)

 - skip_nonenglish: skip triples with subjects that are not English language

 - close_types_with_inheritance: Wikidata has an 'instance_of' relation, which
  connects x to types y that x is in, but there is also a type hierarchy defined
  by a subclass_of relation.  Logically if instance_of(x,y) and subclass_of(y,z)
  then instance_of(x,z) is true, but that's not explicitly stored.  When this is
  set these triples are added to the graph, performing the transitive closure
  operation max_closure_depth times.

  - close_locations_with_containment: there's a similar story with the
  'located_in' relation: the relation located_in(Q1342[Pittsburgh],
  Q156291[Allegheny County]) is stored explicitly in the frame for Pittsburgh,
  but not located_in(Q1342[Pittsburgh],
  Q1400[Pennsylvania])

  - inherit_props_from_entity: in some cases we want to attach a property about
  the entity to its facts as additional qualifiers. eg. if an entity has a
  start-time, we can reasonably assume that its facts also have the same
  start-time themselves. This flag accepts a list of properties which we wish
  to inherit, and these will be added as additional qualifiers to its facts.
  The additional qualifiers are marked with a `*` to indicate that these are
  inherited, e.g. "P585*=20161108".

Filtering logic that doesn't need Wikidata schema information should be placed
in prepare_data rather than here.
"""

import collections
import datetime
import numbers
import os

from absl import app
from absl import flags
from absl import logging
import sling

FLAGS = flags.FLAGS
############################################################
# where to find sling and where to put stuff
############################################################
flags.DEFINE_string(
    'basedir',
    None,
    'a place to put files')
flags.DEFINE_string(
    'sling_kb_file',
    None,
    'where to find sling kb')
############################################################
# actions
############################################################
flags.DEFINE_enum(
    'action',
    'make_kb',
    ['echo_frame', 'make_kb', 'make_freebase_links'],
    'action to perform')
# opts for the make_* actions
flags.DEFINE_boolean(
    'quick_test',
    False,
    'stop after a few thousand frames for testing')
############################################################
# opts for echo_frame action
############################################################
flags.DEFINE_string(
    'frame_to_echo',
    'Q1342',
    'id of frame to echo with echo_frame action')
############################################################
# opts for make_freebase_links action
############################################################
flags.DEFINE_string(
    'freebase_id_file',
    'wikidata2freebase.tsv',
    'file mapping qids to mids')
############################################################
# opts for write_kb action - skip means "don't write"
############################################################
flags.DEFINE_string(
    'kb_file',
    'kb.cfacts',
    'file holding the NQL KB')
flags.DEFINE_boolean(
    'keep_only_numeric_slots',
    False,
    'keep only numeric slot values')
flags.DEFINE_boolean(
    'keep_only_date_slots',
    False,
    'keep only date slot values (this has no effect if '
    '`keep_only_numeric_slots` is also true, since dates are numeric)')
flags.DEFINE_boolean(
    'skip_identifier_rels',
    True,
    'skip relations that name an external identifier, eg FreeBase ID')
flags.DEFINE_boolean(
    'skip_empty_objects',
    True,
    'skip slot values that are neither QIDs nor numbers')
flags.DEFINE_boolean(
    'skip_nonentity_objects',
    True,
    'skip slot values that are not QIDs')
flags.DEFINE_boolean(
    'skip_nonenglish',
    True,
    'skip slots that are not marked English language')
flags.DEFINE_boolean(
    'skip_nonentity_qualifiers',
    True,
    'skip slot qualifiers that are not QIDd')
flags.DEFINE_boolean(
    'skip_qualifiers',
    True,
    'skip all slot qualifiers')
flags.DEFINE_boolean(
    'skip_categories',
    True,
    'skip all slot values that are Wikimedia Categories')
flags.DEFINE_boolean(
    'show_names',
    False,
    'in triples output id[readable name] instead of just the id')
flags.DEFINE_boolean(
    'close_types_with_inheritance',
    True,
    'output not just the instance-of filler but also superclasses')
flags.DEFINE_boolean(
    'close_locations_with_containment',
    True,
    'output locations that indirectly contain the subject')
flags.DEFINE_integer(
    'max_closure_depth',
    3,
    'number of iterations in closing types and locations')
flags.DEFINE_list(
    'inherit_props_from_entity',
    '',
    'comma-separated list of propery ids to inherit from entity to facts')

SUBCLASS_OF_ID = 'P279'
INSTANCE_OF_ID = 'P31'
IDENTIFIER_TYPE_ID = 'Q19847637'
WIKIMEDIA_CATEGORY_TYPE_ID = 'Q4167836'
LOCATED_IN_ID = 'P131'
FREEBASE_IDREL_ID = 'P646'


class SlotCollection(object):
  """Encode a frame with slot-fillers, grouped by slot."""

  def __init__(self, frame):
    self.subject = frame
    self.value_lists = collections.defaultdict(list)
    self.slot_list = []
    for key, val in frame:
      if key not in self.value_lists:
        self.slot_list.append(key)
      self.value_lists[key].append(val)


class SlingExtractor(object):
  """Extract data from Wikidata with Sling."""

  def __init__(self):
    logging.info('loading and indexing kb...')
    self.kb = sling.Store()
    self.kb.load(FLAGS.sling_kb_file)
    self.kb.freeze()
    logging.info('loaded')
    self.ancestor_type_ids_cache = {}
    self.inherit_props_from_entity = set(FLAGS.inherit_props_from_entity)

  def frames(self, filter_english=True):
    """Iterate over all frames, or maybe just english-language ones."""

    for n, f in enumerate(self.kb):
      is_english = 'lang' in f and f['lang'].id == '/lang/en'
      if filter_english is None or is_english == filter_english:
        yield f
      if n > 0 and not n % 500000:
        logging.info('processed %d frames', n)
        if FLAGS.quick_test:
          logging.info('stopping since this is a quick_test')
          return

  def write_freebase_ids(self, filename):
    """Write a file mapping wikidata qids to freebase mids."""
    with open(filename, 'w') as fp:
      for f in self.frames(filter_english=None):
        if FREEBASE_IDREL_ID in f:
          fp.write(f.id + '\t' + str(f[FREEBASE_IDREL_ID]) + '\n')

  def write_kb(self, filename):
    """Write out all triples rel/subject/object, perhaos with qualifiers."""
    with open(filename, 'w') as fp:
      for f in self.frames(filter_english=FLAGS.skip_nonenglish):
        for t in self.as_triples(SlotCollection(f)):
          fp.write(t + '\n')

  def echo_frame(self):
    """For debug, log triples for a subject with given string id."""
    frame = self.kb[FLAGS.frame_to_echo]
    for triple in self.as_triples(SlotCollection(frame)):
      logging.info('triple: %s', triple)

  def ancestor_types(self, x):
    """Find all types frame x belongs to (explicitly or implicitly)."""
    types = set()
    for key, val in x:
      if isinstance(key, sling.Frame) and key.id == INSTANCE_OF_ID:
        types.add(val)
    return self.transitively_close(types, SUBCLASS_OF_ID)

  def transitively_close(self, xs, inheritance_key_id):
    """Find transitive closure under given relation of the set of frames xs."""
    ancestors = set(xs)
    self._collect_ancestors(ancestors, xs, inheritance_key_id, 0)
    return ancestors

  def _collect_ancestors(self, ancestors, types, inheritance_key_id, depth):
    """Augment the set 'ancestors' with superclasses of elements of types."""
    if depth > FLAGS.max_closure_depth:
      return
    new_types = set()
    for t in types:
      for key, val in t:
        if key.id == inheritance_key_id:
          if val not in ancestors:
            new_types.add(val)
            ancestors.add(val)
    if new_types:
      self._collect_ancestors(
          ancestors, new_types, inheritance_key_id, depth + 1)

  def get_name(self, x):
    """Robustly get a string naming frame x."""
    if x is None:
      return None
    elif isinstance(x, sling.Frame):
      return x['name']
    else:
      return 'repr:%r' % x

  def is_id_slot(self, slot):
    """Test if the slot is for an external identifier."""
    if slot.id not in self.ancestor_type_ids_cache:
      self.ancestor_type_ids_cache[slot.id] = [
          x.id for x in self.ancestor_types(slot)]
    anc_ids = self.ancestor_type_ids_cache[slot.id]
    return IDENTIFIER_TYPE_ID in anc_ids

  def get_object(self, _, val):
    """Get the object of a slot - the value or primary slot filler."""
    if not self.is_nested_frame(val):
      return val
    elif self.is_qualified_relation_frame(val):
      return self.qualified_relation_object(val)

  def get_qualifiers(self, _, val):
    """Get the a list of (slot, filler) pairs that qualify a primary slot filler."""
    if not self.is_nested_frame(val):
      return []
    else:
      result = []
      for key1, val1 in val:
        if key1.id != 'is':
          result.append((key1, val1))
      return result

  def is_wikidata_entity(self, x):
    """Check if this is wikidata entity."""
    return isinstance(x, sling.Frame)

  def is_nested_frame(self, x):
    """Test if x is a CVT-node-like frame, for an indirect relation."""
    return isinstance(x, sling.Frame) and not x['id']

  def is_qualified_relation_frame(self, x):
    """Test if x is a CVT-node-like frame with a main object."""
    if not self.is_nested_frame(x):
      return False
    if isinstance(x['is'], (sling.Frame, numbers.Number)):
      return True
    return False

  def qualified_relation_object(self, x):
    """Get the main object of x, if x is a CVT-node-like frame."""
    if isinstance(x['is'], (sling.Frame, numbers.Number)):
      return x['is']
    else:
      return None

  def as_string(self, x):
    """Return a string based on x.id or x.id[x.name]."""
    x_name = self.get_name(x)
    if x is None:
      x_str = 'None'
    elif isinstance(x, sling.Frame) and 'id' in x:
      x_str = x.id
    else:
      try:
        x_str = str(x)
      except UnicodeDecodeError:
        x_str = 'None'
    if FLAGS.show_names:
      return ('%s[%s]' % (x_str, x_name)) if x_name else ('%s[]' % x_str)
    else:
      return x_str

  def valid_slot(self, slot):
    """Should this slot be turned into a triple."""
    return not (self.is_id_slot(slot) and FLAGS.skip_identifier_rels)

  def valid_object(self, obj):
    """Should a slot with this object be turned into a triple."""
    if obj is None and FLAGS.skip_empty_objects:
      return False
    elif not self.is_wikidata_entity(obj) and FLAGS.skip_nonentity_objects:
      return False
    elif FLAGS.skip_categories and isinstance(obj, sling.Frame):
      if obj.id not in self.ancestor_type_ids_cache:
        self.ancestor_type_ids_cache[obj.id] = [
            x.id for x in self.ancestor_types(obj)]
      anc_ids = self.ancestor_type_ids_cache[obj.id]
      if WIKIMEDIA_CATEGORY_TYPE_ID in anc_ids:
        return False
    return True

  def parse_date(self, date_str):
    """Try to parse date from string."""
    date = None
    try:
      if len(date_str) == 4:
        date = datetime.datetime.strptime(date_str, '%Y')
      elif len(date_str) == 6:
        date = datetime.datetime.strptime(date_str, '%Y%m')
      elif len(date_str) == 8:
        date = datetime.datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
      pass
    if date is not None and date.year > 2100:
      # Likely an error
      date = None
    return date

  def valid_slot_obj(self, obj, quals):
    """Do these object and qualifiers satisfy numeric or date conditions."""
    if not FLAGS.keep_only_numeric_slots and not FLAGS.keep_only_date_slots:
      return True
    elif FLAGS.keep_only_numeric_slots:
      if isinstance(obj, numbers.Number):
        return True
      for _, qval in quals:
        if isinstance(qval, numbers.Number):
          return True
    else:
      if isinstance(obj, numbers.Number):
        if self.parse_date(str(obj)) is not None:
          return True
      for _, qval in quals:
        if isinstance(qval, numbers.Number):
          if self.parse_date(str(qval)) is not None:
            return True
    return False

  def as_triples(self, slot_collection):
    """Print qualified triples for every slot filler."""
    # Collect inherited attributes attached to the entity.
    inherit = {}
    if self.inherit_props_from_entity:
      for slot in slot_collection.slot_list:
        if not isinstance(slot, sling.Frame):
          continue
        if slot.id in self.inherit_props_from_entity:
          inherit[slot] = slot_collection.value_lists[slot]

    triples = []
    for slot in slot_collection.slot_list:
      if not isinstance(slot, sling.Frame):
        continue
      if self.valid_slot(slot):
        values = slot_collection.value_lists[slot]
        if slot.id == INSTANCE_OF_ID and FLAGS.close_types_with_inheritance:
          values = self.transitively_close(values, SUBCLASS_OF_ID)
        if slot.id == LOCATED_IN_ID and FLAGS.close_locations_with_containment:
          values = self.transitively_close(values, LOCATED_IN_ID)
        for val in values:
          obj = self.get_object(slot, val)
          if self.valid_object(obj):
            if FLAGS.skip_qualifiers:
              if self.valid_slot_obj(obj, []):
                triples.append('\t'.join([
                    self.as_string(slot),
                    self.as_string(slot_collection.subject),
                    self.as_string(obj)]))
            else:
              quals = self.get_qualifiers(slot, val)
              # Original qualifiers.
              filtered_quals = []
              filtered_qrels = set()
              for (qrel, qval) in quals:
                if self.valid_object(qval):
                  filtered_qrels.add(qrel)
                  filtered_quals.append((qrel, qval))
              # Inherited qualifiers.
              inherited_quals = []
              if slot.id not in self.inherit_props_from_entity:
                for e_rel, e_vals in inherit.items():
                  if e_rel not in filtered_qrels:
                    for e_val in e_vals:
                      if self.valid_object(e_val):
                        inherited_quals.append((e_rel, e_val))
              if self.valid_slot_obj(obj, filtered_quals + inherited_quals):
                qual_string = '\t'.join([
                    '%s=%s' % (self.as_string(qrel), self.as_string(qval))
                    for qrel, qval in filtered_quals])
                if inherited_quals:
                  qual_string += '\t' + '\t'.join([
                      '%s*=%s' % (self.as_string(irel), self.as_string(ival))
                      for irel, ival in inherited_quals])
                triples.append('\t'.join([
                    self.as_string(slot),
                    self.as_string(slot_collection.subject),
                    self.as_string(obj),
                    qual_string]))
    return triples


def main(_):
  logging.set_verbosity(logging.INFO)
  s = SlingExtractor()
  if FLAGS.action == 'echo_frame':
    s.echo_frame()
  elif FLAGS.action == 'make_freebase_links':
    filename = os.path.join(FLAGS.basedir, FLAGS.freebase_id_file)
    s.write_freebase_ids(filename)
  elif FLAGS.action == 'make_kb':
    filename = os.path.join(FLAGS.basedir, FLAGS.kb_file)
    s.write_kb(filename)
  else:
    logging.info('action %s not implemented', FLAGS.action)

if __name__ == '__main__':
  app.run(main)
