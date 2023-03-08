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
"""Install projects from the Language Team."""
import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="language",
    version="0.0.1.dev",
    packages=find_packages(),
    description="Google AI Language.",
    long_description=read("README.md"),
    author="Google Inc.",
    url="https://github.com/google-research/language",
    license="Apache 2.0",
    extras_require={
        "consistent-zero-shot-nmt": [
            "tensorflow-probability==0.6.0",
            "tensor2tensor==1.11.0",
        ],
    },
)
