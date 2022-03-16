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
"""Convert COGS logical forms into variable-free forms."""
import re


def cogs_lf_to_funcall(lf):
  """Converts the given COGS logical form into the variable-free form.

  - Nouns (entities and unaries) become values:
      Jack --> Jack
      cat ( x _ 1 ) --> cat
      * cat ( x _ 1 ) --> * cat
  - Verbs become functions, and their roles become argument names:
      eat . agent ( x _ 2 , Jack ) --> eat ( agent = Jack )
  - The variables representing nouns resolve to their values:
      cat ( x _ 1 ) AND eat . agent ( x _ 2 , x _ 1 ) --> eat ( agent = cat )

  This converter constructs a graph where variables are nodes and binaries
  are edges. After identifying the root, it then performs depth-first traversal
  to construct the output.

  Args:
    lf: Logical form string.

  Returns:
    The converted logical form.
  """
  if "LAMBDA" in lf or "(" not in lf:
    raise ValueError(f"Cannot parse a primitive: {lf}")

  # Parse the terms in the logical form
  # Example: toss . agent ( x _ 1 , John ) --> [toss, agent], [x _ 1, John]
  terms = []
  for raw_term in re.split(" ; | AND ", lf):
    match = re.match(r"(.*) \( (.*) \)", raw_term)
    if not match:
      raise ValueError(f"Malformed term: {raw_term}")
    labels = match.group(1).split(" . ")
    args = match.group(2).split(" , ")
    if len(args) not in (1, 2):
      raise ValueError(f"Invalid number of args: {args}")
    terms.append((labels, args))

  # `nodes` maps variables to node name (e.g., "x _ 3" -> "* cat").
  nodes = {}
  for labels, args in terms:
    if args[0] in nodes:
      # The variable has already been seen; check for conflicts.
      if nodes[args[0]] not in (labels[0], "* " + labels[0]):
        raise ValueError(
            f"Conflicting node name: {nodes[args[0]]} vs. {labels[0]}")
    else:
      nodes[args[0]] = labels[0]

  # `children` maps variables to a list of (edge name, target node).
  children = {}
  # Potential root nodes; any node being a child will be removed.
  root_candidates = list(nodes)
  for labels, args in terms:
    if len(args) == 2:
      if args[0] not in children:
        children[args[0]] = []
      children[args[0]].append((" . ".join(labels[1:]), args[1]))
      if args[1] in root_candidates:
        root_candidates.remove(args[1])
  if len(root_candidates) != 1:
    raise ValueError(f"Multiple roots: {root_candidates}")
  root = root_candidates[0]

  # Depth-first traverse the graph to construct the funcall
  def dfs(node):
    if node not in nodes:
      # Named entity such as "John"
      if node.startswith("x _"):
        raise ValueError(f"Unbound variable {node}")
      if node in children:
        raise ValueError(f"Named entity {node} has children {children[node]}")
      return [node]
    else:
      # A noun like "cat" or a verb like "jump"
      if node not in children:
        return [nodes[node]]
      funcall_args = []
      for edge_label, edge_target in children[node]:
        funcall_args.append([edge_label, "="] + dfs(edge_target))
      funcall = [nodes[node], "("]
      for i, funcall_arg in enumerate(funcall_args):
        if i != 0:
          funcall.append(",")
        funcall.extend(funcall_arg)
      funcall.append(")")
      return funcall

  return " ".join(dfs(root))
