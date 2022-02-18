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
"""The dataset class."""

import collections
import functools
import itertools
import json
import os
import random

from absl import logging
from GroundedScan import dataset
from GroundedScan import world as world_lib
from language.gscan.data import grammar
from language.gscan.data import vocabulary
from language.gscan.data import world
import tensorflow as tf


class RelationGroundedScan(dataset.GroundedScan):
  """The groundedScan dataset for systematic generalization.

  The dataset is similar to original one but supports spatial relation between
  objects in the visual environment. See
  https://github.com/LauraRuis/groundedSCAN/blob/master/GroundedScan/dataset.py
  for more details.
  """

  def __init__(self,
               intransitive_verbs,
               transitive_verbs,
               adverbs,
               nouns,
               color_adjectives,
               size_adjectives,
               location_preps,
               grid_size,
               min_object_size,
               max_object_size,
               type_grammar,
               sample_vocabulary,
               percentage_train,
               percentage_dev=0.01,
               save_directory=os.getcwd(),
               max_recursion=1,
               exclude_type_grammar=""):
    sub_type_grammar = type_grammar.split("relation_")[-1]
    super().__init__(
        intransitive_verbs,
        transitive_verbs,
        adverbs,
        nouns,
        color_adjectives,
        size_adjectives,
        grid_size,
        min_object_size,
        max_object_size,
        sub_type_grammar,
        sample_vocabulary,
        percentage_train,
        percentage_dev=percentage_dev,
        save_directory=save_directory,
        max_recursion=max_recursion)
    # Override the vocabulary.
    if sample_vocabulary == "load":
      if not isinstance(location_preps, dict):
        raise ValueError("The type of location prepositions should be dict.")
      self._vocabulary = vocabulary.RelationVocabulary(
          intransitive_verbs=intransitive_verbs,
          transitive_verbs=transitive_verbs,
          adverbs=adverbs,
          nouns=nouns,
          color_adjectives=color_adjectives,
          size_adjectives=size_adjectives,
          location_preps=location_preps)
    elif sample_vocabulary == "default":
      if not isinstance(location_preps, list):
        raise ValueError("The type of location prepositions should be list.")
      self._vocabulary = vocabulary.RelationVocabulary.initialize(
          intransitive_verbs=intransitive_verbs,
          transitive_verbs=transitive_verbs,
          adverbs=adverbs,
          nouns=nouns,
          color_adjectives=color_adjectives,
          size_adjectives=size_adjectives,
          location_preps=location_preps)
    else:
      raise ValueError(
          f"Unknown value specified for sample_vocabulary: {sample_vocabulary}")
    # Override the world.
    self._world = world.RelationWorld(
        grid_size=grid_size,
        colors=self._vocabulary.get_semantic_colors(),
        object_vocabulary=self._object_vocabulary,
        shapes=self._vocabulary.get_semantic_shapes(),
        save_directory=self.save_directory)

    # Override the grammar.
    self._type_grammar = type_grammar
    self._grammar = grammar.RelationGrammar(
        vocabulary=self._vocabulary,
        type_grammar=type_grammar,
        max_recursion=max_recursion)
    self._exclude_type_grammar = exclude_type_grammar
    if exclude_type_grammar:
      self._exclude_grammar = grammar.RelationGrammar(
          vocabulary=self._vocabulary,
          type_grammar=exclude_type_grammar,
          max_recursion=max_recursion)
    else:
      self._exclude_grammar = None

    self._possible_splits += [
        "relation", "referent", "relative_position_1", "relative_position_2"
    ]
    self._data_pairs = self.get_empty_split_dict()
    self._template_identifiers = self.get_empty_split_dict()
    self._data_statistics = {
        split: self.get_empty_data_statistics()
        for split in self._possible_splits
    }
    self._coverage_commands = {split: {} for split in self._possible_splits}
    self._coverage_worlds = {split: {} for split in self._possible_splits}
    self._coverage_full = {split: {} for split in self._possible_splits}

  def parse_derivation_repr(self, derivation_repr):
    """Parse new Derivation class from string."""
    command_rules, command_lexicon = derivation_repr.split(";")
    return grammar.Derivation.from_str(command_rules, command_lexicon,
                                       self._grammar)

  def get_example_info(self, example):
    command = example["command"]
    target_command = example["target_commands"]
    row = example["situation"]["target_object"]["position"]["row"]
    column = example["situation"]["target_object"]["position"]["column"]
    return command, target_command, row, column

  def count_equivalent_examples(self, split_1="train", split_2="test"):
    """Faster version to count the number of equivalent examples in two splits."""
    return len(self.get_equivalent_ids(split_1, split_2))

  def discard_equivalent_examples(self, split="test"):
    """Faster version to discard equivalent examples that already in training set."""
    to_delete = self.get_equivalent_ids(split_1="train", split_2=split)
    equivalent_examples = len(to_delete)
    for i_to_delete in sorted(to_delete, reverse=True):
      del self._data_pairs[split][i_to_delete]
      del self._template_identifiers[split][i_to_delete]
    return equivalent_examples

  def get_equivalent_ids(self, split_1="train", split_2="test"):
    """Return a list of equivalent ids between two splits."""
    equivalent_ids = []
    example_map = collections.defaultdict(int)
    for i, example in enumerate(self._data_pairs[split_1]):
      template_identifier = self._template_identifiers[split_1][i]
      command, target_command, row, column = self.get_example_info(example)
      key = (template_identifier, command, target_command, row, column)
      example_map[key] += 1
    for i, example in enumerate(self._data_pairs[split_2]):
      template_identifier = self._template_identifiers[split_2][i]
      command, target_command, row, column = self.get_example_info(example)
      key = (template_identifier, command, target_command, row, column)
      if example_map[key] > 0:
        equivalent_ids.append(i)
    return equivalent_ids

  def fill_example(self, command, derivation, situation, target_commands,
                   verb_in_command, target_predicate, visualize, adverb,
                   location_prep, splits):
    """Add an example to the list of examples for the specified split."""
    example = super().fill_example(command, derivation, situation,
                                   target_commands, verb_in_command,
                                   target_predicate, visualize, adverb, splits)
    example["location_prep"] = location_prep
    return example

  def get_empty_data_statistics(self):
    empty_dict = super().get_empty_data_statistics()
    empty_dict["location_preps_in_command"] = collections.defaultdict(int)
    return empty_dict

  def update_data_statistics(self, data_example, split="train"):
    super().update_data_statistics(data_example, split)
    loc_prep = data_example.get("location_prep")
    self._data_statistics[split]["location_preps_in_command"][loc_prep] += 1

  def save_dataset(self, file_name):
    """Saves the current generated data to a file."""
    assert self._data_pairs, "No data to save, call .get_data_pairs()"
    output_path = os.path.join(self.save_directory, file_name)
    with tf.io.gfile.GFile(output_path, "w") as outfile:
      dataset_representation = {
          "grid_size": self._world.grid_size,
          "type_grammar": self._type_grammar,
          "exclude_type_grammar": self._exclude_type_grammar,
          "grammar": self._grammar.__str__(),
          "min_object_size": self._object_vocabulary.smallest_size,
          "max_object_size": self._object_vocabulary.largest_size,
          "max_recursion": self.max_recursion,
          "percentage_train": self._percentage_train,
          "examples": {key: values for key, values in self._data_pairs.items()}
      }
      dataset_representation.update(self._vocabulary.to_representation())
      if self._type_grammar.endswith("simple_intrans"):
        dataset_representation["transitive_verbs"] = {}
      if self._type_grammar.endswith("simple_trans"):
        dataset_representation["intransitive_verbs"] = {}
      if not (self._type_grammar.endswith("adverb") or
              self._type_grammar == "conjunction"):
        dataset_representation["adverbs"] = {}
      if self._exclude_type_grammar:
        dataset_representation.update(
            {"exclude_grammar": self._exclude_grammar.__str__()})
      json.dump(dataset_representation, outfile, indent=4)
    logging.info("Saved dataset to %s", output_path)
    return output_path

  @classmethod
  def load_dataset_from_file(cls, file_path, save_directory, k=0):
    tf.io.gfile.makedirs(save_directory)
    with tf.io.gfile.GFile(file_path, "r") as infile:
      all_data = json.load(infile)
    percentage_train = all_data.get("percentage_train")
    if not percentage_train:
      percentage_train = 0.8
    if not all_data.get("location_preps"):
      all_data["location_preps"] = {"": ""}
    gscan_dataset = cls(
        all_data["intransitive_verbs"],
        all_data["transitive_verbs"],
        all_data["adverbs"],
        all_data["nouns"],
        all_data["color_adjectives"],
        all_data["size_adjectives"],
        all_data["location_preps"],
        all_data["grid_size"],
        all_data["min_object_size"],
        all_data["max_object_size"],
        type_grammar=all_data["type_grammar"],
        save_directory=save_directory,
        percentage_train=percentage_train,
        max_recursion=all_data["max_recursion"],
        sample_vocabulary="load")
    for split, examples in all_data["examples"].items():
      if split == "adverb_1":
        num_examples = len(examples)
        k_random_indices = random.sample(range(0, num_examples), k=k)
      else:
        k_random_indices = []
      # pylint: disable=protected-access
      for i, example in enumerate(examples):
        if i in k_random_indices:
          gscan_dataset._data_pairs["train"].append(example)
          gscan_dataset.update_data_statistics(example, "train")
          gscan_dataset._data_pairs["dev"].append(example)
          gscan_dataset.update_data_statistics(example, "dev")
        else:
          gscan_dataset._data_pairs[split].append(example)
          gscan_dataset.update_data_statistics(example, split)
    return gscan_dataset

  def generate_all_commands(self):
    exclude_templates = None
    if self._exclude_type_grammar:
      self._exclude_grammar.generate_all_commands()
      exclude_templates = self._exclude_grammar.all_templates
    self._grammar.generate_all_commands(exclude_templates)

  def visualize_data_example(self, data_example, parent_save_dir=""):
    """Visualize a data exmaple."""
    (command, meaning, _, situation, actual_target_commands,
     target_demonstration, _) = self.parse_example(data_example)
    command_str = " ".join(command)
    meaning_str = " ".join(meaning)
    actual_target_commands_str = " ".join(actual_target_commands)
    mission = (f"Command: {command_str}, \nMeaning: {meaning_str}, "
               f"\nTarget: {actual_target_commands_str}")
    save_dir = self.visualize_command(
        situation,
        command,
        target_demonstration,
        mission=mission,
        parent_save_dir=parent_save_dir)
    return save_dir

  def visualize_data_examples(self, visualize_per_split=0):
    """Visualize data examples. Sample examples from each split if not any."""
    if not self._examples_to_visualize and visualize_per_split == 0:
      logging.info("No examples to visualize.")
    save_dirs = []
    if not self._examples_to_visualize and visualize_per_split > 0:
      for split, data_pairs in self._data_pairs.items():
        num_to_sample = min(len(data_pairs), visualize_per_split)
        examples = random.sample(data_pairs, k=num_to_sample)
        for data_example in examples:
          save_dir = self.visualize_data_example(
              data_example, parent_save_dir=split)
          save_dirs.append(save_dir)
    else:
      for data_example in self._examples_to_visualize:
        save_dir = self.visualize_data_example(data_example)
        save_dirs.append(save_dir)
    return save_dirs

  def generate_distinct_objects(self, referred_size, referred_color,
                                referred_shape, actual_size, actual_color):
    """Similar to parent's function but generate more disntict for the general case."""
    objects = []
    obligatory_objects = []
    # E.g. distinct from 'circle' -> no other circles;
    # generate some random objects of each other shape.
    if not referred_size and not referred_color:
      all_shapes = self._object_vocabulary.object_shapes
      all_shapes.remove(referred_shape)
      for shape in all_shapes:
        shapes = []
        for _ in range(2):
          shapes.append((self._object_vocabulary.sample_size(),
                         self._object_vocabulary.sample_color(), shape))
        objects.append(shapes)
      return objects, obligatory_objects
    return super().generate_distinct_objects(referred_size, referred_color,
                                             referred_shape, actual_size,
                                             actual_color)

  def is_distinct(self, referred_size, referred_color, referred_shape,
                  actual_object, other_object):
    """Check whether other object is distinct from the referred actual object."""
    other_size, other_color, other_shape = other_object
    actual_size, actual_color, _ = actual_object
    # E.g. distinct from 'circle' -> no other circles.
    if not referred_size and not referred_color:
      if referred_shape == other_shape:
        return False
      return True
    # E.g. distinct from 'red circle' -> no other red circles of any size.
    if not referred_size:
      if referred_color == other_color and referred_shape == other_shape:
        return False
      return True
    if referred_size == "small":
      all_other_sizes = self.get_larger_sizes(actual_size)
    elif referred_size == "big":
      all_other_sizes = self.get_smaller_sizes(actual_size)
    # E.g. distinct from 'small circle' ->
    # no circles of size <= than target in any color.
    if not referred_color:
      if other_shape == referred_shape and other_size not in all_other_sizes:
        return False
      return True
    # E.g. distinct from 'small red circle' -> no red circles of size <= target.
    if (other_shape == referred_shape and other_color == actual_color and
        other_size not in all_other_sizes):
      return False
    return True

  def filter_distinct_objects(self, filter_fn, objects):
    """Filter distinct objects with filter function."""
    distinct_objects = []
    for obj in objects:
      if filter_fn(other_object=obj):
        distinct_objects.append(obj)
    return distinct_objects

  def remove_invalid_siutations(self, situations, target_predicate,
                                ref_predicate):
    """Filter out invalid situations."""
    # No constraint if no reference object.
    if not ref_predicate["location"]:
      return situations
    valid_situations = []
    for situation in situations:
      # Target and reference predicate can't be the same.
      if (target_predicate["size"] == ref_predicate["size"] and
          target_predicate["color"] == ref_predicate["color"] and
          target_predicate["noun"] == ref_predicate["noun"]):
        continue
      ref_position = self._vocabulary.translate_word(ref_predicate["location"])
      # Relative position should be within the grid.
      if ref_position != "next to":
        relative_position = self._world.get_relative_position(
            situation["target_position"], ref_position)
        if self._world.within_grid(relative_position):
          valid_situations.append(situation)
      # There should be > 1 available positions nearby.
      elif self._world.get_nearby_positions(situation["target_position"]):
        valid_situations.append(situation)
    return valid_situations

  def generate_distract_objects(self,
                                distinct_objects,
                                obligatory_objects,
                                target_object,
                                target_ref_object,
                                target_position,
                                ref_position,
                                ref_predicate,
                                max_distract_objects,
                                is_target_distinct_from_ref,
                                max_trials=3):
    """Generate and place distract objects based on target and its reference."""
    distract_positions = []
    num_distract_objects = random.sample(
        range(1, max_distract_objects + 1), k=1).pop()
    trail = 0
    # Sample positions for distract objects. Avoid target being distractor's
    # reference object when target is not distinct from its reference.
    # e.g. when target object is red square and reference object is square.
    for _ in range(num_distract_objects):
      if self._vocabulary.translate_word(
          ref_predicate["location"]) == "next to":
        # The distractor cannot be next to the reference object.
        position = self._world.sample_non_nearby_position(ref_position)
        while (not is_target_distinct_from_ref and
               position in self._world.get_nearby_positions(target_position)):
          position = self._world.sample_non_nearby_position(ref_position)
          trail += 1
          if trail > max_trials:
            return False
      else:
        position = self._world.sample_position()
        while (not is_target_distinct_from_ref and
               target_position == self._world.get_relative_position(
                   position, ref_predicate["location"])):
          position = self._world.sample_position()
          trail += 1
          if trail > max_trials:
            return False
      if not position:
        return False
      self._world.place_object(target_object, position=position)
      distract_positions.append(position)

    # Sample some reference objects for distractors.
    reference_objects = random.sample(
        distinct_objects, k=num_distract_objects - len(obligatory_objects))
    objects_to_place = obligatory_objects + reference_objects
    assert len(objects_to_place) == len(
        distract_positions) == num_distract_objects

    # Place distractors' reference objects.
    for ref_object, position in zip(objects_to_place, distract_positions):
      size, color, shape = ref_object
      ref_location = self._vocabulary.translate_word(ref_predicate["location"])
      p = random.uniform(0, 1)
      if ref_location == "next to":
        # With 1/2 probability we don't place any reference object.
        if p < 0.5 and ref_object not in obligatory_objects:
          continue
        # With 1/2 probability, we place a random reference object.
        other_position = self._world.sample_nearby_position(position)
      else:
        # With 1/3 probability we don't place any reference object.
        if p < 1 / 3 and ref_object not in obligatory_objects:
          continue
        # With 1/3 probability, we place a reference object that is the same
        # as target's reference but with a different relative position.
        if 1 / 3 <= p < 2 / 3 and ref_object not in obligatory_objects:
          other_position = self._world.sample_nearby_position(
              position, exclude_locations=[ref_location])
          size, color, shape = target_ref_object
        # With 1/3 probability, we place a random reference object.
        else:
          other_position = self._world.sample_nearby_position(position)
      if other_position:
        self._world.place_object(
            world_lib.Object(size=size, color=color, shape=shape),
            position=other_position)
    return True

  def place_target_reference_object(self, ref_predicate,
                                    is_distinct_from_target, target_position):
    """Place and return target's refenrece object and its position."""
    possible_ref_objects = self.generate_possible_targets(
        referred_size=self._vocabulary.translate_word(ref_predicate["size"]),
        referred_color=self._vocabulary.translate_word(ref_predicate["color"]),
        referred_shape=self._vocabulary.translate_word(ref_predicate["noun"]))
    possible_ref_objects = self.filter_distinct_objects(is_distinct_from_target,
                                                        possible_ref_objects)
    if not possible_ref_objects:
      return None, None
    ref_object = random.sample(possible_ref_objects, k=1).pop()
    ref_object = world_lib.Object(*ref_object)
    # Always have a valid position as invalid situations are already removed.
    ref_position = self._world.get_relative_position(
        target_position,
        self._vocabulary.translate_word(ref_predicate["location"]))
    self._world.place_object(ref_object, position=ref_position)
    self._world.reference_object = ref_object
    return ref_object, ref_position

  def initialize_world_from_spec(self,
                                 situation_spec,
                                 referred_size,
                                 referred_color,
                                 referred_shape,
                                 actual_size,
                                 sample_percentage=0.5,
                                 min_other_objects=0,
                                 max_objects=12,
                                 ref_predicate=None,
                                 max_distract_objects=1):
    """Intialize the world. Return True if initialization is valid."""
    self._world.clear_situation()
    self._world.place_agent_at(situation_spec["agent_position"])
    target_shape = situation_spec["target_shape"]
    target_color = situation_spec["target_color"]
    target_size = situation_spec["target_size"]
    target_object = world_lib.Object(
        size=target_size, color=target_color, shape=target_shape)
    target_position = situation_spec["target_position"]
    self._world.place_object(target_object, target_position, target=True)
    distinct_objects, obligatory_objects = self.generate_distinct_objects(
        referred_size=self._vocabulary.translate_word(referred_size),
        referred_color=self._vocabulary.translate_word(referred_color),
        referred_shape=self._vocabulary.translate_word(referred_shape),
        actual_size=actual_size,
        actual_color=target_color)

    if ref_predicate and ref_predicate["location"]:
      is_distinct_from_target = functools.partial(
          self.is_distinct,
          referred_size=self._vocabulary.translate_word(referred_size),
          referred_color=self._vocabulary.translate_word(referred_color),
          referred_shape=self._vocabulary.translate_word(referred_shape),
          actual_object=target_object)
      # Place the reference object, which should be distinct from the target.
      ref_object, ref_position = self.place_target_reference_object(
          ref_predicate, is_distinct_from_target, target_position)
      if ref_object is None:
        return False
      ref_size, ref_color, _ = ref_object
      is_target_distinct_from_ref = self.is_distinct(
          referred_size=self._vocabulary.translate_word(ref_predicate["size"]),
          referred_color=self._vocabulary.translate_word(
              ref_predicate["color"]),
          referred_shape=self._vocabulary.translate_word(ref_predicate["noun"]),
          actual_object=ref_object,
          other_object=target_object)

      # Place distract objects and their reference objects if any. The distract
      # objects should be distinct from the target and reference object.
      distinct_ref_objects, obligatory_ref_objects = self.generate_distinct_objects(
          referred_size=self._vocabulary.translate_word(ref_predicate["size"]),
          referred_color=self._vocabulary.translate_word(
              ref_predicate["color"]),
          referred_shape=self._vocabulary.translate_word(ref_predicate["noun"]),
          actual_size=ref_size,
          actual_color=ref_color)
      distinct_ref_objects = itertools.chain(*distinct_ref_objects)
      distinct_ref_objects = self.filter_distinct_objects(
          is_distinct_from_target, distinct_ref_objects)
      filtered_obligatory_ref_objects = self.filter_distinct_objects(
          is_distinct_from_target, obligatory_ref_objects)
      # Return if there is any obligatory that is not distinct from target.
      if len(filtered_obligatory_ref_objects) < len(obligatory_ref_objects):
        return False
      # Return if there is no valid distract objects.
      if not distinct_ref_objects + obligatory_ref_objects:
        return False

      valid_distractors = self.generate_distract_objects(
          distinct_ref_objects,
          obligatory_ref_objects,
          target_object,
          ref_object,
          target_position,
          ref_position,
          ref_predicate,
          max_distract_objects=max_distract_objects,
          is_target_distinct_from_ref=is_target_distinct_from_ref)
      if not valid_distractors:
        return False

      # Remove objects that are not distinct from reference object.
      is_distinct_from_ref = functools.partial(
          self.is_distinct,
          referred_size=self._vocabulary.translate_word(ref_predicate["size"]),
          referred_color=self._vocabulary.translate_word(
              ref_predicate["color"]),
          referred_shape=self._vocabulary.translate_word(ref_predicate["noun"]),
          actual_object=ref_object)
      distinct_objects = [
          self.filter_distinct_objects(is_distinct_from_ref, obj_list)
          for obj_list in distinct_objects
      ]
      distinct_objects = [obj_list for obj_list in distinct_objects if obj_list]
      obligatory_objects = self.filter_distinct_objects(is_distinct_from_ref,
                                                        obligatory_objects)

    # Place other random objects.
    if len(distinct_objects) > 1:
      num_to_sample = int(len(distinct_objects) * sample_percentage)
    else:
      num_to_sample = len(distinct_objects)
    num_to_sample = min(max(min_other_objects, num_to_sample), max_objects)
    objects_to_place = obligatory_objects
    sampled_objects = random.sample(distinct_objects, k=num_to_sample)
    for sampled_objs in sampled_objects:
      objects_to_place.extend(sampled_objs)
    for size, color, shape in objects_to_place:
      other_position = self._world.sample_position()
      self._world.place_object(
          world_lib.Object(size=size, color=color, shape=shape),
          position=other_position)
    return True

  def generate_data(self,
                    situation_specifications,
                    max_examples=None,
                    other_objects_sample_percentage=0.5,
                    split_type="uniform",
                    visualize_per_template=0,
                    visualize_per_split=0,
                    train_percentage=0.8,
                    min_other_objects=0,
                    max_objects=12,
                    cut_off_target_length=25,
                    max_distract_objects=1,
                    max_examples_per_target=None):
    """Generate data pairs given situations and assign them to corresponding splits."""
    example_count = 0
    dropped_examples = 0
    dropped_situations = 0
    for t_num, template_derivations in self._grammar.all_derivations.items():
      visualized_per_template = 0
      visualized_per_split = {split: 0 for split in self._possible_splits}
      for derivation_num, derivation in enumerate(template_derivations):
        arguments = []
        derivation.meaning(arguments)
        assert len(arguments) == 1, "Only one target object supported."
        adverb = ""
        for word in derivation.words():
          if word in self._vocabulary.get_adverbs():
            adverb = word
        _, target_predicate, ref_predicate = arguments.pop().to_predicate(
            return_ref_predicate=True)
        possible_target_objects = self.generate_possible_targets(
            referred_size=self._vocabulary.translate_word(
                target_predicate["size"]),
            referred_color=self._vocabulary.translate_word(
                target_predicate["color"]),
            referred_shape=self._vocabulary.translate_word(
                target_predicate["noun"]))
        for target_size, target_color, target_shape in possible_target_objects:
          relevant_situations = situation_specifications[target_shape][
              target_color][target_size]
          relevant_situations = self.remove_invalid_siutations(
              relevant_situations, target_predicate, ref_predicate)
          num_relevant_situations = len(relevant_situations)
          if num_relevant_situations == 0:
            continue
          if max_examples_per_target:
            num_relevant_situations = min(max_examples_per_target,
                                          num_relevant_situations)
            relevant_situations = random.sample(
                relevant_situations, k=num_relevant_situations)
          idx_to_visualize = random.sample(
              list(range(num_relevant_situations)), k=1).pop()

          if split_type == "uniform":
            idx_for_train = random.sample(
                list(range(num_relevant_situations)),
                k=int(num_relevant_situations * train_percentage))
            idx_for_train = set(idx_for_train)
          for i, relevant_situation in enumerate(relevant_situations):
            visualize = False
            if (example_count + 1) % 10000 == 0:
              logging.info(
                  "%02d/%02d templates, %05d/%05d derivations, %08d examples",
                  t_num, len(self._grammar.all_derivations), derivation_num,
                  len(template_derivations), example_count + 1)
            valid_init = self.initialize_world_from_spec(
                relevant_situation,
                referred_size=target_predicate["size"],
                referred_color=target_predicate["color"],
                referred_shape=target_predicate["noun"],
                actual_size=target_size,
                sample_percentage=other_objects_sample_percentage,
                min_other_objects=min_other_objects,
                max_objects=max_objects,
                ref_predicate=ref_predicate,
                max_distract_objects=max_distract_objects)
            if not valid_init:
              dropped_situations += 1
              if i == idx_to_visualize:
                idx_to_visualize += 1
              continue
            situation = self._world.get_current_situation()
            assert situation.direction_to_target == relevant_situation[
                "direction_to_target"]
            assert situation.distance_to_target == relevant_situation[
                "distance_to_target"]
            target_commands, _, target_action = self.demonstrate_command(
                derivation, initial_situation=situation)

            if i == idx_to_visualize:
              visualize = True
            if visualized_per_template >= visualize_per_template:
              visualize = False
            if adverb and visualized_per_template <= visualize_per_template:
              visualize = True
            if split_type == "uniform":
              if i in idx_for_train:
                splits = ["train"]
              else:
                splits = ["test"]
            elif split_type == "generalization":
              splits = self.assign_splits(
                  target_size, target_color, target_shape, target_action,
                  situation.direction_to_target, target_predicate,
                  self._vocabulary.translate_word(adverb))
              if not splits:
                splits = ["train"]
              elif len(splits) > 1:
                dropped_examples += 1
                self._world.clear_situation()
                continue
            elif split_type == "target_lengths":
              if len(target_commands) > cut_off_target_length:
                splits = ["test"]
              else:
                splits = ["train"]
            elif split_type == "spatial_relation":
              splits = self.assign_relation_splits(target_color, target_shape,
                                                   ref_predicate,
                                                   self._world.reference_object)
              if not splits:
                splits = ["train"]
              elif len(splits) > 1:
                dropped_examples += 1
                self._world.clear_situation()
                continue
            else:
              raise ValueError("Unknown split_type in .get_data_pairs().")
            if visualized_per_split[splits[0]] <= visualize_per_split:
              visualized_per_split[splits[0]] += 1
              visualize = True
            self.fill_example(
                command=derivation.words(),
                derivation=derivation,
                situation=situation,
                target_commands=target_commands,
                verb_in_command=target_action,
                location_prep=ref_predicate["location"],
                target_predicate=target_predicate,
                visualize=visualize,
                adverb=adverb,
                splits=splits)
            for split in splits:
              self._template_identifiers[split].append(t_num)
            example_count += 1
            if visualize:
              visualized_per_template += 1
            self._world.clear_situation()
          if max_examples and example_count >= max_examples:
            return example_count, dropped_examples, dropped_situations
    return example_count, dropped_examples, dropped_situations

  def get_data_pairs(self,
                     max_examples=None,
                     num_resampling=1,
                     other_objects_sample_percentage=0.5,
                     split_type="uniform",
                     visualize_per_template=0,
                     visualize_per_split=0,
                     train_percentage=0.8,
                     min_other_objects=0,
                     max_objects=12,
                     k_shot_generalization=0,
                     make_dev_set=False,
                     cut_off_target_length=25,
                     max_distract_objects=1,
                     max_examples_per_target=None):
    """Generate a set of situations and generate all possible commands."""
    if k_shot_generalization > 0 and split_type == "uniform":
      logging.info(
          "WARNING: k_shot_generalization set to %d but for split_type"
          "uniform this is not used.", k_shot_generalization)

    # Save current situation of the world for later restoration.
    current_situation = self._world.get_current_situation()
    current_mission = self._world.mission
    self.reset_dataset()

    # Generate all situations and commands.
    situation_specifications = self.generate_situations(
        num_resampling=num_resampling)
    self.generate_all_commands()
    example_count, dropped_examples, dropped_situations = self.generate_data(
        situation_specifications,
        max_examples=max_examples,
        other_objects_sample_percentage=other_objects_sample_percentage,
        split_type=split_type,
        visualize_per_template=visualize_per_template,
        visualize_per_split=visualize_per_split,
        train_percentage=train_percentage,
        min_other_objects=min_other_objects,
        max_objects=max_objects,
        cut_off_target_length=cut_off_target_length,
        max_distract_objects=max_distract_objects,
        max_examples_per_target=max_examples_per_target)
    logging.info("Dropped %d examples due to "
                 "belonging to multiple splits.", dropped_examples)
    logging.info(
        "Dropped %d situations due to "
        "invalid reference object initialization.", dropped_situations)

    if split_type in ["generalization", "spatial_relation"]:
      self.make_test_set(
          percentage=(1 - self._percentage_train), type_set="test")
    logging.info("Discarding equivalent examples, may take a while...")
    equivalent_examples = self.discard_equivalent_examples()
    logging.info(
        "Discarded %d examples from the test set that"
        "were already in the training set.", equivalent_examples)
    logging.info("Number of examples: %d", example_count + 1)

    if make_dev_set:
      self.make_test_set(percentage=self._percentage_dev, type_set="dev")

    if k_shot_generalization > 0:
      if split_type == "generalization":
        self.move_k_examples_to_train(k_shot_generalization, split="adverb_1")

    # restore situation
    self.initialize_world(current_situation, mission=current_mission)

  def assign_relation_splits(self, target_color, target_shape, referred_ref,
                             ref_object):
    """Assign the data example to corresponding splits."""
    ref_color = ref_object.color
    ref_shape = ref_object.shape
    splits = []
    # Experiment 1: visual generalization
    if ((target_color == "red" and target_shape == "square") or
        (ref_color == "red" and ref_shape == "square")):
      splits.append("visual")
    # Experiment 2: novel target and reference combination
    if (target_color == "green" and target_shape == "square" and
        ref_color == "blue" and ref_shape
        == "circle") or (target_color == "blue" and target_shape == "circle" and
                         ref_color == "green" and ref_shape == "square"):
      splits.append("relation")
    # Experiment 3: novel target
    if target_color == "yellow" and target_shape == "square":
      splits.append("referent")
    # Experimentt 4: novel relative position
    if self._vocabulary.translate_word(referred_ref["location"]) == "north of":
      splits.append("relative_position_1")
    # Experiment 5: novel relatiove postiion composition
    if self._vocabulary.translate_word(
        referred_ref["location"]) == "south west of":
      splits.append("relative_position_2")
    return splits
