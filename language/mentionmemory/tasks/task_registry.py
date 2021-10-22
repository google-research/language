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
"""Contains registry and registration function for tasks."""



from language.mentionmemory.tasks import base_task

_TASK_REGISTRY = {}

_BaseTaskVar = TypeVar('_BaseTaskVar', bound='base_task.BaseTask')


def register_task(
    name):
  """Register task.

  Task should implement BaseTask abstraction. Used as decorator, for example:

  @register_task('my_task')
  class MyTask(BaseTask):

  Args:
    name: name of registered task.

  Returns:
    Mapping from BaseTask to BaseTask.
  """

  def _wrap(cls):
    """Decorator inner wrapper needed to support `name` argument."""
    if not issubclass(cls, base_task.BaseTask):
      raise TypeError('Invalid task. Task %s does not subclass BaseTask.' %
                      cls.__name__)

    if name in _TASK_REGISTRY:
      raise ValueError(
          'Task name %s has already been registered with class %s' %
          (name, _TASK_REGISTRY[name].__name__))

    _TASK_REGISTRY[name] = cls

    return cls

  return _wrap


def get_registered_task(name):
  """Takes in task name and returns corresponding task from registry."""
  return _TASK_REGISTRY[name]
