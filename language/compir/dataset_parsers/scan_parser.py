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
"""A parser that applies all transformations for SCAN."""

import re

from language.compir.dataset_parsers import dataset_parser


class ScanParser(dataset_parser.DatasetParserInterface):
  """A parser that applies all transformations for SCAN."""

  def __init__(self, train_examples_raw,
               test_examples_raw):
    # super(dataset_parser.DatasetParserInterface, self).__init__()
    super().__init__(train_examples_raw, test_examples_raw)
    self.atom_map = self._get_atom_map()

  def preprocess_program(self, program):
    """No preprocessing for SCAN programs."""
    return program

  def _bracket_atomic_action(self, utterance, program):
    """Wraps atomic actions (without "and"/"after") with parentheses.

    Args:
      utterance: the question.
      program: a program where "twice" occurs exactly once, "thrice" occurs
        exactly once, or neither "twice" nor "thrice" occurs.

    Returns:
      The program with repeated actions (that repeat twice or thrice) wrapped
      with parenthesis.
    """

    program_tokens = program.split()
    num_program_tokens = len(program_tokens)
    if "twice" in utterance:
      # The repeated action consists of 1/2 of the tokens in the program.
      action_repeated = " ".join(program_tokens[:int(num_program_tokens / 2)])
      program_bracketed = "( {action} ) ( {action} )".format(
          action=action_repeated)
    elif "thrice" in utterance:
      # The repeated action consists of 1/3 of the tokens in the program.
      action_repeated = " ".join(program_tokens[:int(num_program_tokens / 3)])
      program_bracketed = "( {action} ) ( {action} ) ( {action} )".format(
          action=action_repeated)
    else:
      # There is no repeated action.
      program_bracketed = str(program)
    return program_bracketed

  def _bracket_complex_action(self, utterance, category):
    """Wraps complex actions (separated by "and"/"after") with parentheses."""
    utterance_parts = utterance.split(" {} ".format(category))
    program_parts_bracketed = [
        self._bracket_atomic_action(utterance_part,
                                    self.atom_map[utterance_part])
        for utterance_part in utterance_parts
    ]
    if category == "and":
      return "( {} ) ( {} )".format(program_parts_bracketed[0],
                                    program_parts_bracketed[1])
    else:
      return "( {} ) ( {} )".format(program_parts_bracketed[1],
                                    program_parts_bracketed[0])

  def f_reversible(self, example):
    """Transforms a single program to its reversible IR."""

    if "and" in example.utterance:
      return self._bracket_complex_action(example.utterance, "and")
    elif "after" in example.utterance:
      return self._bracket_complex_action(example.utterance, "after")
    else:
      return "( {} )".format(
          self._bracket_atomic_action(example.utterance, example.program))

  def _should_anonymize(self, token, last_token):
    return last_token is not None and token == last_token and (
        token == "I_TURN_LEFT" or token == "I_TURN_RIGHT")

  def f_lossy(self, program, is_rir):
    """Transforms a single program to its lossy IR."""
    lir_tokens = []
    last_token = None
    for token in program.split():
      if not self._should_anonymize(token, last_token):
        lir_tokens.append(token)
        last_token = token
      else:
        # In this case a repeated "left"/"right" action is encountered, and thus
        # is anonymized.
        lir_tokens.append("I_TURN_DIRECT")
    return " ".join(lir_tokens)

  def _get_atom_map(self):
    """Gets (utterance, program) pairs, used for adding brackets to programs."""
    atoms = [
        ("turn left", "I_TURN_LEFT"),
        ("turn right", "I_TURN_RIGHT"),
        ("walk left", "I_TURN_LEFT I_WALK"),
        ("run left", "I_TURN_LEFT I_RUN"),
        ("look left", "I_TURN_LEFT I_LOOK"),
        ("jump left", "I_TURN_LEFT I_JUMP"),
        ("walk right", "I_TURN_RIGHT I_WALK"),
        ("run right", "I_TURN_RIGHT I_RUN"),
        ("look right", "I_TURN_RIGHT I_LOOK"),
        ("jump right", "I_TURN_RIGHT I_JUMP"),
        ("walk", "I_WALK"),
        ("run", "I_RUN"),
        ("look", "I_LOOK"),
        ("jump", "I_JUMP"),
        ("turn opposite left", "I_TURN_LEFT I_TURN_LEFT"),
        ("turn opposite right", "I_TURN_RIGHT I_TURN_RIGHT"),
        ("walk opposite left", "I_TURN_LEFT I_TURN_LEFT I_WALK"),
        ("walk opposite right", "I_TURN_RIGHT I_TURN_RIGHT I_WALK"),
        ("run opposite left", "I_TURN_LEFT I_TURN_LEFT I_RUN"),
        ("run opposite right", "I_TURN_RIGHT I_TURN_RIGHT I_RUN"),
        ("look opposite left", "I_TURN_LEFT I_TURN_LEFT I_LOOK"),
        ("look opposite right", "I_TURN_RIGHT I_TURN_RIGHT I_LOOK"),
        ("jump opposite left", "I_TURN_LEFT I_TURN_LEFT I_JUMP"),
        ("jump opposite right", "I_TURN_RIGHT I_TURN_RIGHT I_JUMP"),
        ("turn around left", "I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT"),
        ("turn around right",
         "I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT"),
        ("walk around left",
         "I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT"
         " I_WALK"),
        ("walk around right",
         "I_TURN_RIGHT I_WALK I_TURN_RIGHT I_WALK I_TURN_RIGHT I_WALK "
         "I_TURN_RIGHT I_WALK"),
        ("run around left",
         "I_TURN_LEFT I_RUN I_TURN_LEFT I_RUN I_TURN_LEFT I_RUN I_TURN_LEFT I_RUN"
        ),
        ("run around right",
         "I_TURN_RIGHT I_RUN I_TURN_RIGHT I_RUN I_TURN_RIGHT I_RUN "
         "I_TURN_RIGHT I_RUN"),
        ("look around left",
         "I_TURN_LEFT I_LOOK I_TURN_LEFT I_LOOK I_TURN_LEFT I_LOOK I_TURN_LEFT"
         " I_LOOK"),
        ("look around right", "I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK "
         "I_TURN_RIGHT I_LOOK "
         "I_TURN_RIGHT I_LOOK"),
        ("jump around left",
         "I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP I_TURN_LEFT"
         ""
         " I_JUMP"),
        ("jump around right",
         "I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP "
         "I_TURN_RIGHT I_JUMP"),
    ]
    atom_map = {atom[0]: atom[1] for atom in atoms}
    # Adds "X twice" actions.
    atom_map.update({
        "{} twice".format(atom[0]): "{} {}".format(atom[1], atom[1])
        for atom in atoms
    })
    # Adds "X thrice" actions.
    atom_map.update({
        "{} thrice".format(atom[0]):
        "{} {} {}".format(atom[1], atom[1], atom[1]) for atom in atoms
    })
    return atom_map

  def postprocess_program(self, program):
    """No postprocessing for SCAN programs."""
    return program

  def f_reversible_inverse(self, program):
    """Removes all parenthesis from the program."""
    program_no_brackets = program.replace("(", "").replace(")", "")
    program_single_space = re.sub(" +", " ", program_no_brackets).strip()
    return program_single_space
