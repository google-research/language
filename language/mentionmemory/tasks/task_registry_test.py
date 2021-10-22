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
"""Tests for task registry."""

from absl.testing import absltest

from language.mentionmemory.tasks import base_task
from language.mentionmemory.tasks import task_registry


@task_registry.register_task('decorated_task')
class DecoratedTask(base_task.BaseTask):
  pass


class UnDecoratedTask(base_task.BaseTask):
  pass


class InvalidTask(object):
  pass


class TaskRegistryTest(absltest.TestCase):

  def test_decorated_task(self):
    """Simple test to verify that decorated tasks have been registered."""
    task_name = task_registry.get_registered_task('decorated_task')
    self.assertIsNotNone(task_name)
    self.assertEqual(task_name.__name__, 'DecoratedTask')

  def test_undecorated_task(self):
    """Simple test to verify that we can register tasks at runtime."""
    # Register the task.
    task_registry.register_task('undecorated_task')(UnDecoratedTask)

    # Retrieve it.
    task_name = task_registry.get_registered_task('undecorated_task')
    self.assertIsNotNone(task_name)
    self.assertEqual(task_name.__name__, 'UnDecoratedTask')

    # Verify that we can still access previously registerd decorated layers.
    task_name = task_registry.get_registered_task('decorated_task')
    self.assertIsNotNone(task_name)
    self.assertEqual(task_name.__name__, 'DecoratedTask')

  def test_invalid_task(self):
    """Verify we get an exception when trying to register invalid task."""
    with self.assertRaises(TypeError):
      task_registry.register_task('invalid_task')(InvalidTask)  # pytype: disable=wrong-arg-types

  def test_multiple_task_registrations(self):
    """Verify that re-using an already registered name raises an exception."""
    with self.assertRaises(ValueError):
      task_registry.register_task('decorated_task')(UnDecoratedTask)


if __name__ == '__main__':
  absltest.main()
