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
"""A parser that applies all transformations for CFQ."""

import collections
import re

from language.compir.dataset_parsers import dataset_parser


class CfqParser(dataset_parser.DatasetParserInterface):
  """A parser that applies all transformations for CFQ."""

  def __init__(self, train_examples_raw,
               test_examples_raw):
    super().__init__(train_examples_raw, test_examples_raw)
    self.trimmed_relations = self._get_trimmed_relations()
    self.trimmed_relations_inv = {
        rel_short: rel_long
        for rel_long, rel_short in self.trimmed_relations.items()
    }

  def preprocess_program(self, program):
    """Switches OOV T5 tokens to in-vocabulary tokens."""
    program_processed = str(program)
    program_processed = program_processed.replace("{", "lb")
    program_processed = program_processed.replace("}", "rb")
    program_processed = program_processed.replace("^", "#")
    return program_processed

  def _get_program_parts(self, program):
    """Parses a SPARQL program into a prefix and conjuncts."""
    # Remove the closing bracket and split on opening bracket.
    if not program.endswith(" rb"):
      raise ValueError("Wrong program format.")
    program_no_closing = program[:-3]
    parts = program_no_closing.split(" lb ")
    if len(parts) != 2:
      raise ValueError("Wrong program format.")
    prefix = parts[0]
    conjuncts_str = parts[1]
    conjuncts = conjuncts_str.split(" . ")
    return prefix, conjuncts

  def _get_trimmed_relations(self):
    """Gets a mapping between relations and their short naming."""
    relations = set()
    for example in self.train_examples:
      tokens = example.program.split()
      for token in tokens:
        if "." in token and token != ".":  # Identify tokens that are relations.
          relations.add(token)

    trimmed_relations = {}
    relations_original = set(relations)
    for relation in relations_original:
      # Trim relations that contain "ns:".
      if "ns:" in relation and not relation.startswith("#"):
        relation_trimmed = relation.split("ns:")[-1]
        if relation_trimmed in relations:
          raise RuntimeError(
              "The trimmed relation {} is not unique!".format(relation_trimmed))
        relations.add(relation_trimmed)
        trimmed_relations[relation] = relation_trimmed
    return trimmed_relations

  def f_reversible(self, example):
    """Transforms a single program to its reversible IR."""

    def get_subj_rel_to_objects(conjuncts):
      """Merges conjuncts that share the same subject and relation."""
      subj_rel_to_objects = collections.OrderedDict()
      for conjunct in conjuncts:
        if conjunct.startswith("FILTER"):  # Keep FILTER conjunts as is.
          subj_rel_to_objects[conjunct] = None
        else:
          subj, rel, obj = conjunct.split()  # A (subj, rel, obj) triple.
          if rel == "a":  # Keep unary conjuncts as is.
            subj_rel_to_objects[conjunct] = None
          else:  # Handle a binary conjunct.
            subj_rel = (subj, rel)
            if subj_rel not in subj_rel_to_objects:
              subj_rel_to_objects[subj_rel] = []
            subj_rel_to_objects[subj_rel].append(obj)
      return subj_rel_to_objects

    def get_conjuncts_reversible(
        subj_rel_to_objects,
        subj_objs_to_rels):
      """Generates conjuncts in their reversible intermediate representation."""
      conjuncts_reversible = []
      added_subj_objs = []
      for subj_rel, objects in subj_rel_to_objects.items():
        if objects is None:  # Only wrap the conjunct with parentheses.
          conjuncts_reversible.append("( {} )".format(subj_rel))
        else:
          # Prepare conjunct in the form of (s , (r_1 r_2 ...) (o_1 , o_2 ...)).
          subj, _ = subj_rel
          objects_tup = tuple(objects)
          if (objects_tup, subj) in added_subj_objs:
            # Already handled the conjuncts with this subject and objects list.
            continue
          else:
            added_subj_objs.append((objects_tup, subj))
          conjunct_reversible = "( {} ( {} ) ( {} ) )".format(
              subj, " , ".join(subj_objs_to_rels[(objects_tup, subj)]),
              " , ".join(objects))
          conjuncts_reversible.append(conjunct_reversible)
      return conjuncts_reversible

    # Trim long relations.
    program_trimmed_rel = str(example.program)
    for relation, relation_trimmed in self.trimmed_relations.items():
      program_trimmed_rel = program_trimmed_rel.replace(relation,
                                                        relation_trimmed)

    prefix, conjuncts = self._get_program_parts(program_trimmed_rel)

    # Prepare subject-relation to objects map.
    subj_rel_to_objects = get_subj_rel_to_objects(conjuncts)

    # Prepare subject-objects to relations map.
    subj_objs_to_rels = collections.defaultdict(list)
    for subj_rel, objects in subj_rel_to_objects.items():
      if objects is not None:
        objects_tuple = tuple(objects)
        subj, rel = subj_rel
        key = (objects_tuple, subj)
        subj_objs_to_rels[key].append(rel)

    conjuncts_reversible = get_conjuncts_reversible(subj_rel_to_objects,
                                                    subj_objs_to_rels)
    reversible_ir = "{} lb {} rb".format(prefix,
                                         " . ".join(conjuncts_reversible))
    return reversible_ir

  def f_lossy(self, program, is_rir):
    """Transforms a single program to its lossy IR."""
    var_token = "var"
    lir = str(program)
    if not is_rir:  # Anonymize each token of an entity or variable.
      entities = ["M" + str(i) for i in range(10)]
      variables = ["?x" + str(i) for i in range(10)]
      for entity in entities:  # Anonymize entities.
        lir = lir.replace(entity, var_token)
      for variable in variables:  # Anonymize variables.
        lir = lir.replace(variable, var_token)
      return lir
    else:
      # Anonymize entities and variables, which are possibly merged, to a single
      # "var" token, e.g., "( M1, M2 )" -> "( var )".
      to_be_masked = []  # Strings for entities and variables to anonymize.
      # Parts that should be anonymized are inside parentheses.
      regex = (r"\( ([^ ]+?) a [^ ]+? \)|\( FILTER \( ([^ ]+) != ([^ ]+) \) "
               r"\)|\( ([^ ]+?) \( .+? \) \( (.+?) \) \)")
      matches = re.findall(regex, program)
      for match in matches:
        # Each entry in 'match' corresponds to a different parentheses in regex.
        for group in match:
          # A group contains comma separated entities/variables, e.g., "M1 , M2"
          if group:
            to_be_masked.append(group)
      lir = str(program)
      # Sort to replace larger groups first.
      to_be_masked.sort(key=len, reverse=True)

      for group in to_be_masked:
        lir = lir.replace(group, var_token)
      return lir

  def postprocess_program(self, program):
    """Postprocesses a predicted SPARQL program."""
    prefix, conjuncts = self._get_program_parts(program)
    # Take unique conjuncts and sort them alphabetically. FILTER conjuncts can
    # have duplicates, so these are not turned into a set.
    conjuncts_unique = list(
        set([
            conjunct for conjunct in conjuncts
            if not conjunct.startswith("FILTER")
        ])) + [
            conjunct for conjunct in conjuncts if conjunct.startswith("FILTER")
        ]
    conjuncts_ordered = sorted(list(conjuncts_unique))

    program_processed = "{} {{ {} }}".format(prefix,
                                             " . ".join(conjuncts_ordered))
    # Replace back T5 OOV tokens.
    program_processed = program_processed.replace("lb", "{")
    program_processed = program_processed.replace("rb", "}")
    program_processed = program_processed.replace("#", "^")
    return program_processed

  def _to_orig_relation(self, relation):
    """Converts a relation back to its original name."""
    return self.trimmed_relations_inv.get(relation, relation)

  def _invert_unary_conjunct(self, subject, relation):
    """Inverts an unary conjunct for an IR to its original representation."""
    return "{} a {}".format(subject, self._to_orig_relation(relation))

  def _invert_binary_conjunct(self, subject, relations,
                              objects):
    """Inverts a binary conjunct for an IR to its original representation."""
    inverted_conjuncts = []
    # Get relations to their original names.
    relations_inverted = [
        self._to_orig_relation(relation) for relation in relations.split(" , ")
    ]
    objects_separated = objects.split(" , ")
    # Generate conjuncts in their original distributed form.
    for relation in relations_inverted:
      for object_triple in objects_separated:
        inverted_conjuncts.append("{} {} {}".format(subject, relation,
                                                    object_triple))
    return inverted_conjuncts

  def f_reversible_inverse(self, program):
    """Invert a SPARQL program from its IR back to its original representation."""
    prefix, conjuncts = self._get_program_parts(program)
    regex_binary = r"\( (.*?) \( (.*?) \) \( (.*?) \) \)"
    regex_unary = r"\( (.*?) a (.*?) \)"
    conjuncts_original_form = []
    for conjunct in conjuncts:
      if "FILTER" in conjunct:
        # Only remove brackets for FILTER conjuncts.
        conjuncts_original_form.append(conjunct[1:-1].strip())
      else:  # Handle unary and binary conjuncts.
        unary_parse = re.findall(regex_unary, conjunct)
        binary_parse = re.findall(regex_binary, conjunct)
        if unary_parse:  # Parse as an unary conjunct.
          conjuncts_original_form.append(
              self._invert_unary_conjunct(unary_parse[0][0], unary_parse[0][1]))
        elif binary_parse:  # Parse as a binary conjunct.
          conjuncts_original_form.extend(
              self._invert_binary_conjunct(binary_parse[0][0],
                                           binary_parse[0][1],
                                           binary_parse[0][2]))
        else:
          raise ValueError("Wrong prediction format.")
    return "{} lb {} rb".format(prefix, " . ".join(conjuncts_original_form))
