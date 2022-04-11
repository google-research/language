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
#!/bin/bash

# Install requirements.txt
pip install -r requirements.txt

# Install huggingface transformers from github so that we have access to example
# scripts.
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .

# Download Roberta checkpoint.
cd ../
mkdir roberta
gsutil cp -R gs://gresearch/ASQA/ckpts/roberta-squad roberta/
