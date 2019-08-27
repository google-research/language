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
#!/usr/bin/env bash

set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

OUTPUT_DIR="${1:-iwslt17-official}"
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"
mkdir -p $OUTPUT_DIR_DATA

if [ ! -f "${OUTPUT_DIR_DATA}/DeEnItNlRo-DeEnItNl.tgz" ]; then
  echo "Downloading IWSLT17. This may take a while..."
  curl "https://wit3.fbk.eu/archive/2017-01-trnmted//texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.tgz" \
    -H "Connection: keep-alive" \
    -H "Upgrade-Insecure-Requests: 1" \
    -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.84 Safari/537.36" \
    -H "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8" \
    -H "Referer: https://wit3.fbk.eu/download.php?release=2017-01-trnmted&type=texts&slang=DeEnItNlRo&tlang=DeEnItNlRo" \
    -H "Accept-Encoding: gzip, deflate, br" \
    -H "Accept-Language: en-US,en;q=0.9,ru;q=0.8" \
    -H "Cookie: PHPSESSID=mre4mfh9h6llic3qjmli0lnc72" \
    -o "${OUTPUT_DIR_DATA}/DeEnItNlRo-DeEnItNl.tgz" \
    --compressed
fi

# Extract everything
if [ ! -d "${OUTPUT_DIR_DATA}/original" ]; then
  echo "Extracting all files..."
  mkdir -p "${OUTPUT_DIR_DATA}/original"
  tar -xvzf "${OUTPUT_DIR_DATA}/DeEnItNlRo-DeEnItNl.tgz" -C "${OUTPUT_DIR_DATA}/original" --strip=1
fi

echo "All done."
