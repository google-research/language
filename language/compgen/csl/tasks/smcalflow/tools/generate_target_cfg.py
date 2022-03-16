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
"""Generates target CFG.

The grammar is a bit loose in certain cases, e.g. might conflate
datetimes/dates/times and datetime/date/time constraints, so could be improved.
"""

from absl import app
from absl import flags

from language.compgen.csl.targets import target_grammar

from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input txt file.")

flags.DEFINE_string("output", "", "Output txt file.")

# Below is a collection of 3-tuples specifying the target CFG.
# We will replace a subspan nested between parens with a nonterminal
# (the 3rd item in the tuple), if the preceding tokens match the 2nd item
# in the tuple, and, if specified, the parent nonterminal matches the
# 1st item in the tuple.
PREFIX_SET = (
    (None, ":name # (", "name#"),
    (None, ":name (", "name"),
    (None, ":number # (", "number#"),
    (None, ":number (", "number"),
    (None, ":holiday # (", "holiday#"),
    (None, ":holiday (", "holiday"),
    (None, ":day # (", "day#"),
    (None, ":day (", "day"),
    (None, ":month # (", "month#"),
    (None, ":month (", "month"),
    (None, ":year # (", "year#"),
    (None, ":year (", "year"),
    (None, ":dow # (", "dow#"),
    (None, ":dow (", "dow"),
    (None, ":day1 # (", "dow#"),
    (None, ":day2 # (", "dow#"),
    (None, ":week # (", "week#"),
    (None, ":week (", "week"),
    (None, ":attendees (", "attendees"),
    (None, ":duration (", "duration"),
    (None, ":start (", "datetime"),
    (None, ":end (", "datetime"),
    (None, ":subject (", "subject"),
    (None, ":location (", "location"),
    (None, ":date (", "date"),
    (None, ":startDate (", "date"),
    (None, ":endDate (", "date"),
    (None, ":date1 (", "date"),
    (None, ":date2 (", "date"),
    (None, ":dateRange (", "date"),
    (None, ":time (", "time"),
    (None, ":time1 (", "time"),
    (None, ":time2 (", "time"),
    (None, ":period (", "period"),
    (None, ":minutes (", "minutes"),
    (None, ":minutes # (", "minutes#"),
    (None, ":hours (", "hours"),
    (None, ":hours # (", "hours#"),
    (None, ":range (", "range"),
    (None, ":timeRange (", "timeRange"),
    (None, ":dateTime (", "dateTime"),
    (None, ":dateTime1 (", "dateTime"),
    (None, ":dateTime2 (", "dateTime"),
    (None, ":recipient (", "recipient"),
    (None, ":people (", "people"),
    (None, ":showAs (", "showAs"),
    (None, "Yield :output (", "output"),
    (None, "CreateCommitEventWrapper :event (", "eventwrapper"),
    (None, "CreatePreflightEventWrapper :constraint (", "eventconstraint"),
    (None, "FindEventWrapperWithDefaults :constraint (", "eventconstraint"),
    (None, "FindLastEvent :constraint (", "eventconstraint"),
    (None, "FindNumNextEvent :constraint (", "eventconstraint"),
    ("attendees", "andConstraint (", "attendees"),
    ("attendees", "andConstraint ( ##ATTENDEES ) (", "attendees"),
    ("subject", "# (", "string"),
    ("location", "# (", "locationkeyphrase"),
    ("datetime", "?< (", "datetime"),
    ("datetime", "?<= (", "datetime"),
    ("datetime", "?= (", "datetime"),
    ("datetime", "?>= (", "datetime"),
    ("datetime", "?> (", "datetime"),
    ("date", "?< (", "date"),
    ("date", "?<= (", "date"),
    ("date", "?= (", "date"),
    ("date", "?>= (", "date"),
    ("date", "?> (", "date"),
    ("time", "?< (", "time"),
    ("time", "?<= (", "time"),
    ("time", "?= (", "time"),
    ("time", "?>= (", "time"),
    ("time", "?> (", "time"),
    ("duration", "?< (", "duration"),
    ("duration", "?<= (", "duration"),
    ("duration", "?= (", "duration"),
    ("duration", "?>= (", "duration"),
    ("duration", "?> (", "duration"),
    (None, "adjustByPeriodDuration (", "datetime"),
    (None, "adjustByPeriodDuration ( ##DATEIME ) (", "periodduration"),
    (None, "DateTimeConstraint :constraint (", "time"),
    (None, "nextDayOfMonth (", "date"),
    (None, "nextDayOfMonth ( ##DATE ) # (", "number"),
    (None, "nextDayOfWeek (", "date"),
    (None, "nextDayOfWeek ( ##DATE ) # (", "dow#"),
    (None, "nextMonthDay (", "date"),
    (None, "nextMonthDay ( ##DATE ) # (", "month#"),
    (None, "nextMonthDay ( ##DATE ) # ( ##MONTH# ) # (", "number"),
    (None, "adjustByPeriod (", "date"),
    (None, "adjustByPeriod ( ##DATE ) (", "period"),
    (None, "toMonths # (", "number"),
    (None, "toWeeks # (", "number"),
    (None, "toYears # (", "number"),
    (None, "toDays # (", "number"),
    (None, "toHours # (", "number"),
    (None, "toMinutes # (", "number"),
    (None, "addDurations (", "duration"),
    (None, "addDurations ( ##DURATION ) (", "duration"),
)


def read_txt(filename):
  """Read file to list of lines."""
  lines = []
  with gfile.GFile(filename, "r") as txt_file:
    for line in txt_file:
      line = line.decode().rstrip()
      lines.append(line)
  print("Loaded %s lines from %s." % (len(lines), filename))
  return lines


def parse_to_nested_list(target):
  """Parse target to nested lists based on parens."""
  tokens = target.split(" ")
  buffer_stack = [[]]
  for token in tokens:
    if token == ")":
      buffer = buffer_stack.pop()
      if not buffer_stack:
        raise ValueError("Empty buffer stack.\ntarget: `%s`\nbuffer: `%s`" %
                         (target, buffer))
      buffer_stack[-1].append(buffer)
    buffer_stack[-1].append(token)
    if token == "(":
      buffer_stack.append([])
  if len(buffer_stack) != 1:
    raise ValueError("Bad buffer_stack %s for target %s" %
                     (buffer_stack, target))
  return buffer_stack


def print_nested_list(nested_lists, indent=0):
  if isinstance(nested_lists, list):
    for element in nested_lists:
      print_nested_list(element, indent + 2)
  else:
    padding = " " * indent
    print("%s%s" % (padding, nested_lists))


class PrefixChecker(object):
  """Check prefix for matches."""

  def __init__(self):
    # Map of integer to map of prefix tuple to nonterminal symbol.
    # Integer -> (Tuple -> String)
    self.len_to_prefix_map = {}

    for parent_nt, prefix_string, nt in PREFIX_SET:
      lhs_nt = nt.upper()
      prefix = tuple(prefix_string.split(" "))
      prefix_len = len(prefix)
      if prefix_len not in self.len_to_prefix_map:
        self.len_to_prefix_map[prefix_len] = {}
      prefix_map = self.len_to_prefix_map[prefix_len]
      prefix_map[(parent_nt, prefix)] = lhs_nt

  def get_nt(self, parent_nt, rhs_symbols):
    """Return NT if prefix matches."""
    for prefix_len, prefix_map in self.len_to_prefix_map.items():
      if len(rhs_symbols) < prefix_len:
        continue
      current_prefix = tuple(rhs_symbols[-prefix_len:])
      if (None, current_prefix) in prefix_map:
        # Matched with a prefix for any NT.
        return prefix_map[(None, current_prefix)]
      elif (parent_nt.lower(), current_prefix) in prefix_map:
        # Matched with a prefix.
        return prefix_map[(parent_nt.lower(), current_prefix)]
      elif "CreateCommitEventWrapper" in current_prefix:
        print("Prefix `%s` not in map" % (current_prefix,))
    return None


def get_rhs_and_extract_rules(parent_nt, prefix_checker, nested_lists, ruleset):
  """Extract rules from nested_lists."""

  rhs_symbols = []
  for current_symbol in nested_lists:
    if isinstance(current_symbol, list):
      sub_nt = prefix_checker.get_nt(parent_nt, rhs_symbols)
      if sub_nt:
        # Replace with nonterminal symbol.
        rhs_nt = "%s%s" % (target_grammar.NON_TERMINAL_PREFIX, sub_nt)
        rhs_symbols.append(rhs_nt)
        # Recurse and add sub-rule.
        sub_rhs = get_rhs_and_extract_rules(sub_nt, prefix_checker,
                                            current_symbol, ruleset)
        ruleset.add((sub_nt, " ".join(sub_rhs)))
      else:
        # Flatten.
        sub_rhs = get_rhs_and_extract_rules(parent_nt, prefix_checker,
                                            current_symbol, ruleset)
        rhs_symbols.extend(sub_rhs)
    else:
      rhs_symbols.append(current_symbol)

  # Return RHS.
  return rhs_symbols


def main(unused_argv):
  targets = read_txt(FLAGS.input)
  prefix_checker = PrefixChecker()
  ruleset = set()

  for target in targets:
    nested_lists = parse_to_nested_list(target)
    rhs = get_rhs_and_extract_rules(target_grammar.ROOT_SYMBOL, prefix_checker,
                                    nested_lists, ruleset)
    ruleset.add((target_grammar.ROOT_SYMBOL, " ".join(rhs)))

  target_rules = []
  for lhs, rhs in sorted(list(ruleset)):
    rule = target_grammar.TargetCfgRule(lhs, rhs)
    target_rules.append(rule)

  target_grammar.rules_to_txt_file(target_rules, FLAGS.output)


if __name__ == "__main__":
  app.run(main)
