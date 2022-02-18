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
"""The grammar class."""

import collections
import itertools

from GroundedScan import grammar
from GroundedScan import world

# Represents the non-terminal symbol `NT` with linked index i
NTParent = collections.namedtuple(
    "Nonterminal", ("name", "index"), defaults=[None, 0])
TParent = collections.namedtuple("Terminal", "name")


# Define sub-class to override __str__ and __repr__ for easier debugging.
class Nonterminal(NTParent):
  """Nonterminal symbol."""

  def __str__(self):
    return "%s (%d)" % (self.name, self.index)

  def __repr__(self):
    return str(self)

  # Override the instance type for consistency.
  @property
  def __class__(self):
    return grammar.Nonterminal


# Define sub-class to override __str__ and __repr__ for easier debugging.
class Terminal(TParent):
  """Terminal symbol."""

  def __str__(self):
    return "'%s'" % self.name

  def __repr__(self):
    return str(self)

  # Override the instance type for consistency.
  @property
  def __class__(self):
    return grammar.Terminal


ROOT = Nonterminal("ROOT")
VP = Nonterminal("VP")
VV_intransitive = Nonterminal("VV_intransitive")
VV_transitive = Nonterminal("VV_transitive")
RB = Nonterminal("RB")
DP = Nonterminal("DP")
NP = Nonterminal("NP")
NN = Nonterminal("NN")
JJ = Nonterminal("JJ")
PP = Nonterminal("PP")
LOC = Nonterminal("LOC")

LOCATION = world.SemType("location")
fields = ("action", "is_transitive", "manner", "adjective_type", "noun",
          "location")
Weights = collections.namedtuple(
    "Weights", fields, defaults=[None] * len(fields))


class LogicalForm(world.LogicalForm):
  """Logical form class supports object relation."""

  def split_terms_on_location(self):
    is_loc = [True if term.specs.location else False for term in self.terms]
    split_index = is_loc.index(True) if True in is_loc else -1
    target_terms = self.terms[split_index + 1:]
    ref_terms = self.terms[:split_index + 1]
    return target_terms, ref_terms

  def to_predicate(self, return_ref_predicate=False):
    """Similar to the parent's function but allow returning ref predicate."""
    assert len(self.variables) == 1
    target_predicate = {"noun": "", "size": "", "color": ""}
    ref_predicate = {"noun": "", "size": "", "color": "", "location": ""}
    target_terms, ref_terms = self.split_terms_on_location()
    for term in target_terms:
      term.to_predicate(target_predicate)
    for term in ref_terms:
      term.to_predicate(ref_predicate)
    object_str = ""
    if target_predicate["color"]:
      object_str += " " + target_predicate["color"]
    object_str += " " + target_predicate["noun"]
    object_str = object_str.strip()
    if return_ref_predicate:
      return object_str, target_predicate, ref_predicate
    else:
      return object_str, target_predicate


class Term(world.Term):
  """Term class that supports location in predicate."""

  def replace(self, var_to_find, replace_by_var):
    """Find a variable `var_to_find` the arguments and replace it by `replace_by_var`."""
    return Term(
        function=self.function,
        args=tuple(replace_by_var if variable == var_to_find else variable
                   for variable in self.arguments),
        specs=self.specs,
        meta=self.meta)

  def to_predicate(self, predicate):
    output = self.function
    if self.specs.location:
      predicate["location"] = output
    else:
      super().to_predicate(predicate)


class Rule(object):
  """Rule class of form LHS -> RHS with method instantiate defines its meaning.

  The rule is similar to original gSCAN grammar but supoorts indexing. See
  https://github.com/LauraRuis/groundedSCAN/blob/master/GroundedScan/grammar.py
  for more details.
  """

  def __init__(self, lhs, rhs, max_recursion=2):
    self.lhs = lhs
    self.rhs = rhs
    self.sem_type = None
    self.max_recursion = max_recursion

  def instantiate(self, *args, **kwargs):
    raise NotImplementedError()

  def __repr__(self):
    rhs = " ".join([str(rhs) for rhs in self.rhs])
    return f"{self.lhs} -> {rhs}"


class LexicalRule(Rule):
  """Rule of form Non-Terminal -> Terminal."""

  def __init__(self, lhs, word, specs, sem_type):
    super().__init__(lhs=lhs, rhs=[Terminal(word)], max_recursion=1)
    self.name = word
    self.sem_type = sem_type
    self.specs = specs

  def instantiate(self, meta=None, **kwargs):
    var = grammar.free_var(self.sem_type)
    return LogicalForm(
        variables=(var,),
        terms=(Term(self.name, (var,), specs=self.specs, meta=meta),))

  def __repr__(self):
    return f"{self.lhs.name} -> {self.rhs[0].name}"


class Root(Rule):
  """Root rule."""

  def __init__(self):
    super().__init__(lhs=ROOT, rhs=[VP])

  def instantiate(self, child, **kwargs):
    return child


class VpWrapper(Rule):
  """VP Wrapper rule."""

  def __init__(self, max_recursion=0):
    super().__init__(lhs=VP, rhs=[VP, RB], max_recursion=max_recursion)

  def instantiate(self, rb, vp, unused_meta, **kwargs):
    bound = rb.bind(vp.head)
    assert bound.variables[0] == vp.head
    return LogicalForm(
        variables=vp.variables + bound.variables[1:],
        terms=vp.terms + bound.terms)


class VpIntransitive(Rule):
  """Intransitive VP rule."""

  def __init__(self):
    super().__init__(lhs=VP, rhs=[VV_intransitive, Terminal("to"), DP])

  def instantiate(self, vv, dp, meta, **kwargs):
    role = Term("patient", (vv.head, dp.head))
    meta["arguments"].append(dp)
    return LogicalForm(
        variables=vv.variables + dp.variables,
        terms=vv.terms + dp.terms + (role,))


class VpTransitive(Rule):
  """Transitive VP rule."""

  def __init__(self):
    super().__init__(lhs=VP, rhs=[VV_transitive, DP])

  def instantiate(self, vv, dp, meta, **kwargs):
    role = Term("patient", (vv.head, dp.head))
    meta["arguments"].append(dp)
    return LogicalForm(
        variables=vv.variables + dp.variables,
        terms=vv.terms + dp.terms + (role,))


class Dp(Rule):
  """DP rule."""

  def __init__(self, l_ind=0, r_inds=(0, 0)):
    super().__init__(
        lhs=Nonterminal("DP", l_ind),
        rhs=[Terminal("a"), Nonterminal("NP", r_inds[1])])

  def instantiate(self, noun_p, **kwargs):
    return noun_p


class NpWrapper(Rule):
  """NP Wrapper rule."""

  def __init__(self, max_recursion=0, l_ind=0, r_inds=(0, 0)):
    super().__init__(
        lhs=Nonterminal("NP", l_ind),
        rhs=[Nonterminal("JJ", r_inds[0]),
             Nonterminal("NP", r_inds[1])],
        max_recursion=max_recursion)

  def instantiate(self, jj, noun_p, unused_meta=None, **kwargs):
    bound = jj.bind(noun_p.head)
    assert bound.variables[0] == noun_p.head
    return LogicalForm(
        variables=noun_p.variables + bound.variables[1:],
        terms=noun_p.terms + bound.terms)


class Np(Rule):
  """NP rule."""

  def __init__(self, l_ind=0, r_inds=(0,)):
    super().__init__(
        lhs=Nonterminal("NP", l_ind), rhs=[Nonterminal("NN", r_inds[0])])

  def instantiate(self, nn, **kwargs):
    return nn


class NpPpWrapper(Rule):
  """NP PP Wrapper rule."""

  def __init__(self, max_recursion=0):
    super().__init__(lhs=NP, rhs=[NP, PP], max_recursion=max_recursion)

  def instantiate(self, noun_p, pp, unused_meta=None, **kwargs):
    bound = noun_p.bind(pp.head)
    assert bound.variables[0] == pp.head
    return LogicalForm(
        variables=pp.variables + bound.variables[1:],
        terms=pp.terms + bound.terms)


class PpWrapper(Rule):
  """PP Wrapper rule."""

  def __init__(self, max_recursion=0, l_ind=0, r_inds=(0, 0)):
    super().__init__(
        lhs=Nonterminal("PP", l_ind),
        rhs=[Nonterminal("LOC", r_inds[0]),
             Nonterminal("DP", r_inds[1])],
        max_recursion=max_recursion)

  def instantiate(self, loc, dp, unused_meta=None, **kwargs):
    bound = loc.bind(dp.head)
    assert bound.variables[0] == dp.head
    return LogicalForm(
        variables=dp.variables + bound.variables[1:],
        terms=dp.terms + bound.terms)


class Derivation(grammar.Derivation):
  """Holds a constituency tree that makes up a sentence."""

  # Override the instance type for consistency.
  @property
  def __class__(self):
    return grammar.Derivation

  @classmethod
  def from_rules(cls, rules, symbol=ROOT, lexicon=None):
    """Recursively form a derivation from a rule list."""

    # If the current symbol is a Terminal, close current branch and return.
    if isinstance(symbol, grammar.Terminal):
      return symbol
    if symbol not in lexicon.keys():
      next_rule = rules.pop()
    else:
      next_rule = lexicon[symbol].pop()

    return Derivation(
        next_rule,
        tuple(
            cls.from_rules(rules, symbol=next_symbol, lexicon=lexicon)
            for next_symbol in next_rule.rhs))

  def to_rules(self, rules, lexicon):
    """In-order travesal for the constituency tree."""
    if isinstance(self.rule, LexicalRule):
      if self.rule.lhs not in lexicon:
        lexicon[self.rule.lhs] = [self.rule]
      else:
        lexicon[self.rule.lhs] = [self.rule] + lexicon[self.rule.lhs]
    else:
      rules.insert(0, self.rule)
    for child in self.children:
      if isinstance(child, Derivation):
        child.to_rules(rules, lexicon)
      else:
        lexicon[child] = [child]
    return


class RelationGrammar(grammar.Grammar):
  """The grammar class that supports new rules."""

  BASE_RULES = [Root(), Dp(), Np()]
  RELATION_RULES = [
      NpPpWrapper(max_recursion=1),
      PpWrapper(r_inds=[0, 1]),
      Dp(l_ind=1, r_inds=[0, 1]),
      NpWrapper(max_recursion=2, l_ind=1, r_inds=[0, 1]),
      Np(l_ind=1, r_inds=[0])
  ]
  RULES = {}
  RULES["simple_trans"] = BASE_RULES.copy() + [
      VpTransitive(), NpWrapper(max_recursion=1)
  ]
  RULES["simple_intrans"] = BASE_RULES.copy() + [
      VpIntransitive(), NpWrapper(max_recursion=1)
  ]
  RULES["normal"] = BASE_RULES.copy() + [
      VpIntransitive(),
      VpTransitive(),
      NpWrapper(max_recursion=2)
  ]
  RULES["adverb"] = RULES["normal"].copy() + [VpWrapper()]
  # Add rules support spatial relations.
  for rule_name in set(RULES):
    RULES[f"relation_{rule_name}"] = RULES[rule_name].copy() + RELATION_RULES

  def lexical_rules(self, verbs_intrans, verbs_trans, adverbs, nouns,
                    color_adjectives, size_adjectives, location_preps):
    """Instantiate the lexical rules using new LexicalRule class."""
    assert size_adjectives or color_adjectives, (
        "Please specify words for at least one of size_adjectives or "
        "color_adjectives.")
    all_rules = []
    for verb in verbs_intrans:
      vv_intrans_rule = LexicalRule(
          lhs=VV_intransitive,
          word=verb,
          sem_type=world.EVENT,
          specs=Weights(action=verb, is_transitive=False))
      all_rules.append(vv_intrans_rule)
    if self.type_grammar != "simple":
      for verb in verbs_trans:
        vv_trans_rule = LexicalRule(
            lhs=VV_transitive,
            word=verb,
            sem_type=world.EVENT,
            specs=Weights(action=verb, is_transitive=True))
        all_rules.append(vv_trans_rule)
    if self.type_grammar.endswith("adverb") or self.type_grammar == "full":
      for word in adverbs:
        rb_rule = LexicalRule(
            lhs=RB, word=word, sem_type=world.EVENT, specs=Weights(manner=word))
        all_rules.append(rb_rule)
    for word in nouns:
      nn_rule = LexicalRule(
          lhs=NN, word=word, sem_type=world.ENTITY, specs=Weights(noun=word))
      all_rules.append(nn_rule)
    if color_adjectives:
      for word in color_adjectives:
        jj_rule = LexicalRule(
            lhs=JJ,
            word=word,
            sem_type=world.ENTITY,
            specs=Weights(adjective_type=world.COLOR))
        all_rules.append(jj_rule)
    if size_adjectives:
      for word in size_adjectives:
        jj_rule = LexicalRule(
            lhs=JJ,
            word=word,
            sem_type=world.ENTITY,
            specs=Weights(adjective_type=world.SIZE))
        all_rules.append(jj_rule)
    if self.type_grammar.startswith("relation"):
      for word in location_preps:
        loc_rule = LexicalRule(
            lhs=LOC, word=word, sem_type=LOCATION, specs=Weights(location=word))
        all_rules.append(loc_rule)
    return all_rules

  def __init__(self, vocabulary, max_recursion=1, type_grammar="normal"):
    """Defines a grammar of NT -> NT rules and NT -> T rules."""
    if type_grammar not in self.RULES:
      raise ValueError(f"Specified unsupported type grammar {type_grammar}")
    self.type_grammar = type_grammar
    if (type_grammar == "simple_intrans" and
        not vocabulary.get_intransitive_verbs()):
      raise ValueError("Please specify intransitive verbs.")
    elif (type_grammar == "simple_trans" and
          not vocabulary.get_transitive_verbs()):
      raise ValueError("Please specify transitive verbs.")
    self.rule_list = self.RULES[type_grammar] + self.lexical_rules(
        vocabulary.get_intransitive_verbs(), vocabulary.get_transitive_verbs(),
        vocabulary.get_adverbs(), vocabulary.get_nouns(),
        vocabulary.get_color_adjectives(), vocabulary.get_size_adjectives(),
        vocabulary.get_location_preps())
    nonterminals = {rule.lhs for rule in self.rule_list}
    self.rules = {nonterminal: [] for nonterminal in nonterminals}
    self.nonterminals = {nt.name: nt for nt in nonterminals}
    self.terminals = {}

    self.vocabulary = vocabulary
    self.rule_str_to_rules = {}
    for rule in self.rule_list:
      self.rules[rule.lhs].append(rule)
      self.rule_str_to_rules[str(rule)] = rule
    self.expandables = set(
        rule.lhs
        for rule in self.rule_list
        if not isinstance(rule, LexicalRule))
    self.categories = {
        "manner": set(vocabulary.get_adverbs()),
        "shape": set(vocabulary.get_nouns()),
        "color": set(vocabulary.get_color_adjectives()),
        "size": set(vocabulary.get_size_adjectives()),
        "location": set(vocabulary.get_location_preps()),
    }
    self.word_to_category = {}
    for category, words in self.categories.items():
      for word in words:
        self.word_to_category[word] = category

    self.max_recursion = max_recursion
    self.all_templates = []
    self.all_derivations = {}
    self.command_statistics = self.empty_command_statistics()

  @staticmethod
  def empty_command_statistics():
    return {
        VV_intransitive: {},
        VV_transitive: {},
        NN: {},
        JJ: {},
        RB: {},
        LOC: {}
    }

  def generate_all_commands(self, exclude_templates=None):
    """Generate all commands but allow excluding unused templates."""

    # Generate all possible templates from the grammar.
    initial_template = grammar.Template()
    initial_template.add_value(value=ROOT, expandable=True)
    self.generate_all(
        current_template=initial_template,
        all_templates=self.all_templates,
        rule_use_counter={})
    # Remove duplicate templates due to ambiguous PP attachment.
    self.remove_duplicate_templates()
    if exclude_templates:
      self.remove_exclude_templates(exclude_templates)
    # For each template, form all possible commands
    # by combining it with the lexicon.
    for i, (derivation_template,
            derivation_rules) in enumerate(self.all_templates):
      derivations = self.form_commands_from_template(derivation_template,
                                                     derivation_rules)
      self.all_derivations[i] = derivations

  def form_commands_from_template(self, derivation_template, derivation_rules):
    """Similar to parent's function but use new Derivation class."""

    # Replace each lexical rule with the possible words from the lexicon.
    replaced_template = []
    previous_symbol = None
    lexicon = {}
    for symbol in derivation_template:
      if isinstance(symbol, grammar.Nonterminal):
        # pytype: disable=attribute-error
        possible_words = [s.name for s in self.rules[symbol]]
        for rule in self.rules[symbol]:
          lexicon[rule.name] = rule
        if previous_symbol == symbol:
          previous_words = replaced_template.pop()
          first_words, second_words = self.split_on_category(previous_words)
          replaced_template.append(first_words)
          replaced_template.append(second_words)
        else:
          replaced_template.append(possible_words)
      else:
        lexicon[symbol.name] = symbol
        replaced_template.append([symbol.name])
      previous_symbol = symbol

    # Generate all possible commands from the templates.
    all_commands = list(itertools.product(*replaced_template))
    all_derivations = []
    for command in all_commands:
      command_lexicon = {}
      for word, symbol in zip(command, derivation_template):
        if symbol not in command_lexicon:
          command_lexicon[symbol] = [lexicon[word]]
        else:
          command_lexicon[symbol] = [lexicon[word]] + command_lexicon[symbol]
        if isinstance(symbol, grammar.Nonterminal):
          if word not in self.command_statistics[symbol].keys():
            self.command_statistics[symbol][word] = 1
          else:
            self.command_statistics[symbol][word] += 1
      derivation = Derivation.from_rules(
          derivation_rules.copy(), symbol=ROOT, lexicon=command_lexicon)
      if " ".join(derivation.words()) != " ".join(command):
        raise ValueError("Derivation and command not the same.")
      # pytype: enable=attribute-error
      all_derivations.append(derivation)
    return all_derivations

  def remove_duplicate_templates(self):
    """Remove duplicate templates from the grammar."""
    all_templates = []
    current_templates = []
    for template, rules in self.all_templates:
      if template not in current_templates:
        all_templates.append((template, rules))
        current_templates.append(template)
    self.all_templates = all_templates

  def remove_exclude_templates(self, exclude_templates):
    """Remove specified exclude templates from the grammar."""
    all_templates = []
    exclude_templates = [(t[0], str(t[1])) for t in exclude_templates]
    for template, rules in self.all_templates:
      if (template, str(rules)) not in exclude_templates:
        all_templates.append((template, rules))
    self.all_templates = all_templates
