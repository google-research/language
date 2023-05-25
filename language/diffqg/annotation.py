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
"""Reads gold and predicted annotations and pairs them based on the text."""
import dataclasses
import json

from typing import Optional, Dict, Iterator, Any


@dataclasses.dataclass
class Annotation:
  """A single DiffQG annotation, either gold or model-generated."""

  base_text: str
  target_text: str
  answer_span: str
  question: Optional[str]
  is_edited: Optional[bool] = None

  def __init__(self, json_dic: Dict[str, Any]) -> None:
    self.base_text = json_dic["base"]
    self.target_text = json_dic["target"]
    self.question = json_dic["q"] if json_dic["q"] else None
    self.answer_span = json_dic["a"]
    self.is_edited = json_dic["is_edited"] if "is_edited" in json_dic else None

  def key_annotation(self) -> str:
    return f"{self.base_text}::{self.target_text}::{self.answer_span}"


@dataclasses.dataclass
class PairedAnnotation:
  """A gold annotation paired with a predicted annotation."""

  answer_span: str
  base_text: str
  target_text: str
  is_edited: bool
  gold_question: Optional[str]
  pred_question: Optional[str]

  def __init__(self, gold_anno: Annotation, pred_anno: Annotation) -> None:
    if gold_anno.answer_span != pred_anno.answer_span:
      raise ValueError(
          "Invalid PairedAnnotation, mismatched answers: "
          f"'{gold_anno.answer_span}' vs. "
          f"'{pred_anno.answer_span}'."
      )
    if gold_anno.base_text != pred_anno.base_text:
      raise ValueError(
          "Invalid PairedAnnotation, mismatched base texts: "
          f"'{gold_anno.base_text}' vs. "
          f"'{pred_anno.base_text}'."
      )
    if gold_anno.target_text != pred_anno.target_text:
      raise ValueError(
          "Invalid PairedAnnotation, mismatched target texts: "
          f"'{gold_anno.target_text}' vs. "
          f"'{pred_anno.target_text}'."
      )
    if pred_anno.is_edited is not None:
      raise ValueError(
          "Possible mismatch of gold/pred annotations in making a "
          "pair, pred_anno has no valid value for 'is_edited'."
      )
    self.answer_span = gold_anno.answer_span
    self.base_text = gold_anno.base_text
    self.target_text = gold_anno.target_text
    self.gold_question = gold_anno.question
    self.pred_question = pred_anno.question
    self.is_edited = gold_anno.is_edited

  def __str__(self):
    return json.dumps(dataclasses.asdict(self))

  def key_annotation(self) -> str:
    return f"{self.base_text}::{self.target_text}::{self.answer_span}"


def _read_annotations(diffqg_fi: str) -> Dict[str, Annotation]:
  """Reads annotations from a file in either the gold or predicted format.

  Args:
    diffqg_fi: Path to the input file.

  Returns:
    Dictionary mapping a unique string identifier of the annotation to an
      Annotation object.
  """
  with open(diffqg_fi, "rt") as fr:
    json_dics = [json.loads(s) for s in fr.readlines()]
  print(f"Read {len(json_dics)} jsons from {diffqg_fi}")
  overall_dic = {}

  for json_dic in json_dics:
    anno = Annotation(json_dic)
    key = anno.key_annotation()
    if key in overall_dic:
      print(f"Possible repeated annotation for {key}")
      if anno.question != overall_dic[key].question:
        print("Error! Different questions.")
        print(f"--\n{anno.question}\n{overall_dic[key].question}\n--")
    overall_dic[key] = anno

  return overall_dic


def make_paired_annotations(
    gold_generation_fi: str, pred_generation_fi: str
) -> Iterator[PairedAnnotation]:
  """Reads annotations from files and match them based on the text.

  Args:
    gold_generation_fi: Path to the gold examples. This file should be provided.
    pred_generation_fi: Path to the model predicted examples. Format of this
      file can be found in the README.

  Yields:
    Each predicted + gold annotation pair.
  """
  gold_annos = _read_annotations(gold_generation_fi)
  pred_annos = _read_annotations(pred_generation_fi)
  for key, gold_anno in gold_annos.items():
    if key not in pred_annos:
      print(f"Gold annotation missing from predictions: {key}")
      continue
    pred_anno = pred_annos[key]
    yield PairedAnnotation(gold_anno, pred_anno)
