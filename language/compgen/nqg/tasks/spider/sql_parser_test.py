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
"""Tests for sql_parser."""

from language.compgen.nqg.tasks.spider import sql_parser

import tensorflow as tf

SQL_1 = ("select avg(t1.age) , t3.dorm_name from student as t1 join lives_in as"
         " t2 on t1.stuid = t2.stuid join dorm as t3 on t3.dormid = t2.dormid "
         "group by t3.dorm_name")
EXP_PARSE_1 = """ROOT => ##SELECT group by ##GROUP_EXPR
-SELECT => select ##SELECT_EXPR from ##FROM_EXPR
--SELECT_EXPR => ##SELECT_ATOM , ##SELECT_ATOM
---SELECT_ATOM => avg ( ##SELECT_ATOM )
----SELECT_ATOM => ##T . ##T
-----T => t1
-----T => age
---SELECT_ATOM => ##T . ##T
----T => t3
----T => dorm_name
--FROM_EXPR => ##FROM_EXPR join ##JOIN
---FROM_EXPR => ##FROM_EXPR join ##JOIN
----FROM_EXPR => ##T as ##T
-----T => student
-----T => t1
----JOIN => ##T on ##JOIN_ATOM
-----T => ##T as ##T
------T => lives_in
------T => t2
-----JOIN_ATOM => ##T = ##T
------T => ##T . ##T
-------T => t1
-------T => stuid
------T => ##T . ##T
-------T => t2
-------T => stuid
---JOIN => ##T on ##JOIN_ATOM
----T => ##T as ##T
-----T => dorm
-----T => t3
----JOIN_ATOM => ##T = ##T
-----T => ##T . ##T
------T => t3
------T => dormid
-----T => ##T . ##T
------T => t2
------T => dormid
-GROUP_EXPR => ##T . ##T
--T => t3
--T => dorm_name"""

SQL_2 = ("select distinct driverid , stop from pitstops where duration > "
         "(select min(duration) from pitstops where raceid = 841)")
EXP_PARSE_2 = """ROOT => ##SELECT where ##EXPR
-SELECT => select distinct ##SELECT_EXPR from ##FROM_EXPR
--SELECT_EXPR => ##SELECT_ATOM , ##SELECT_ATOM
---SELECT_ATOM => driverid
---SELECT_ATOM => stop
--FROM_EXPR => pitstops
-EXPR => ##SELECT_ATOM > ##SELECT_ATOM
--SELECT_ATOM => duration
--SELECT_ATOM => ( ##ROOT )
---ROOT => ##SELECT where ##EXPR
----SELECT => select ##SELECT_EXPR from ##FROM_EXPR
-----SELECT_EXPR => min ( ##SELECT_ATOM )
------SELECT_ATOM => duration
-----FROM_EXPR => pitstops
----EXPR => ##SELECT_ATOM = ##SELECT_ATOM
-----SELECT_ATOM => raceid
-----SELECT_ATOM => 841"""


class SqlParserTest(tf.test.TestCase):

  def test_parse_simple(self):
    parse = sql_parser.parse_sql("select foo from bar")
    self.assertIsNotNone(parse)

  def test_parse_1(self):
    parse = sql_parser.parse_sql(SQL_1)
    parse_str = sql_parser.parse_to_str(parse)
    self.assertEqual(parse_str, EXP_PARSE_1)

  def test_parse_2(self):
    parse = sql_parser.parse_sql(SQL_2)
    parse_str = sql_parser.parse_to_str(parse)
    self.assertEqual(parse_str, EXP_PARSE_2)

  def test_expand_unit_rules_cycle(self):
    # Test for cycles in expanding unit rules.
    rule_1 = sql_parser.Rule("X", "##Y")
    rule_2 = sql_parser.Rule("Y", "##X")
    with self.assertRaises(ValueError):
      _ = sql_parser.expand_unit_rules([rule_1, rule_2])


if __name__ == "__main__":
  tf.test.main()
