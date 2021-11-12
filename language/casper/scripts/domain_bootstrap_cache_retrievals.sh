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
# Cache the retrievals for domain bootstrapping setups.
# There are 2 setups: seen-bootstrap and unseen-bootstrap.
# For both setups, the retrieval index has O_train + N_support at test time.
# For seen-bootstrap, that is also the index during training.
# But for unseen-bootstrap, the index has only O_train during training.
set -e
set -u

[[ -z "${DATA_DIR}" ]] && echo "Error: DATA_DIR must be set." && exit 1

# Filter the data by domains
for domain in {alarm,calling,event,messaging,music}; do
  pattern='"domain": "'"${domain}"'"'
  grep "${pattern}" "${DATA_DIR}/raw/train.no_exemplars.jsonl" \
    > "${DATA_DIR}/raw/train.only-${domain}.no_exemplars.jsonl"
  grep -v "${pattern}" "${DATA_DIR}/raw/train.no_exemplars.jsonl" \
    > "${DATA_DIR}/raw/train.except-${domain}.no_exemplars.jsonl"
  grep "${pattern}" "${DATA_DIR}/raw/dev.no_exemplars.jsonl" \
    > "${DATA_DIR}/raw/dev.only-${domain}.no_exemplars.jsonl"
  grep -v "${pattern}" "${DATA_DIR}/raw/dev.no_exemplars.jsonl" \
    > "${DATA_DIR}/raw/dev.except-${domain}.no_exemplars.jsonl"
done

# N_support = 100 random examples from the bootstrapped domain
# Hard-coded here for reproducibility.
SUPPORT_ALARM='\(1760\|399\|13197\|2301\|1467\|13041\|10627\|7104\|1229\|8053\|4382\|11457\|1812\|8970\|5956\|7097\|4307\|7090\|9235\|4682\|1573\|8039\|724\|14795\|8272\|13642\|5997\|5988\|2090\|2045\|11251\|253\|2073\|9174\|6607\|11418\|11443\|12908\|14345\|9959\|2696\|6339\|10164\|2377\|408\|1991\|13142\|4305\|13941\|14350\|12136\|1188\|7825\|13302\|10011\|4362\|1576\|4200\|10541\|14849\|1216\|8394\|6395\|1560\|9993\|4557\|1471\|2366\|623\|7256\|1295\|15598\|14259\|14809\|12696\|1637\|13371\|4646\|8155\|12367\|9108\|6482\|10108\|10056\|12238\|4549\|6330\|7472\|3675\|14666\|6051\|20\|2030\|7304\|7989\|10098\|11360\|4056\|14080\|7392\)'
SUPPORT_CALLING='\(5789\|11539\|230\|2133\|8417\|13453\|3966\|13071\|4399\|15340\|9845\|13351\|2933\|1261\|990\|8625\|12506\|2862\|11406\|1958\|9599\|13657\|7764\|8366\|7734\|2564\|9957\|4558\|15246\|12370\|3874\|12719\|14956\|10842\|14485\|2547\|11627\|7982\|8678\|8175\|2268\|12816\|177\|11591\|14535\|10246\|5404\|4035\|5219\|2446\|723\|3464\|10306\|9286\|9718\|9139\|7348\|4467\|9830\|1468\|6536\|14630\|6516\|15611\|4158\|5595\|13501\|6173\|5832\|13808\|13141\|15031\|3577\|1433\|124\|11671\|2599\|3653\|1660\|11309\|5760\|11105\|15432\|11329\|1362\|12998\|275\|6779\|12951\|12233\|7578\|13939\|12699\|1262\|8884\|14522\|7201\|4164\|12702\|13136\)'
SUPPORT_EVENT='\(468\|13930\|14237\|14035\|14001\|13154\|3651\|2470\|12993\|7159\|9475\|6548\|11466\|12918\|14018\|13786\|5865\|842\|12328\|14580\|638\|4916\|4144\|4119\|13347\|1985\|11771\|11712\|8362\|4238\|9654\|5075\|7067\|682\|3083\|7475\|13000\|15308\|11968\|2857\|13508\|299\|12707\|6928\|3761\|6872\|12536\|3328\|928\|5408\|1524\|10921\|9575\|14741\|7080\|4970\|3833\|8647\|9486\|10148\|14523\|4979\|6960\|6361\|12488\|11836\|10338\|12481\|12218\|15497\|1680\|8001\|3768\|2949\|418\|11788\|4011\|6433\|11894\|3952\|6096\|14274\|587\|8561\|14633\|8354\|10178\|6013\|14460\|6688\|5615\|12910\|1527\|7682\|11124\|3313\|6164\|2896\|12760\|12726\)'
SUPPORT_MESSAGING='\(10417\|4990\|8056\|11544\|1018\|5130\|3530\|9636\|2749\|8843\|904\|11955\|5757\|8694\|13788\|11412\|12906\|15240\|12785\|3273\|8046\|4862\|10658\|12400\|8321\|14075\|9685\|7011\|71\|9498\|12522\|3147\|5863\|8461\|7945\|14541\|13210\|11091\|7059\|1989\|5150\|5787\|2661\|9644\|12430\|1443\|10147\|12426\|12538\|8644\|6446\|12774\|13874\|15060\|187\|14619\|11165\|11847\|9176\|3739\|8848\|5492\|10102\|8255\|4442\|13618\|10576\|2761\|1521\|7636\|15223\|5345\|3568\|1146\|11285\|9542\|292\|12495\|10690\|11637\|707\|13832\|5721\|8281\|2561\|10445\|3217\|12959\|1588\|14661\|4394\|3950\|7399\|9018\|12315\|13217\|15154\|14550\|11947\|184\)'
SUPPORT_MUSIC='\(5776\|5534\|11643\|12140\|11755\|10855\|13687\|1556\|4926\|6519\|96\|6075\|4291\|3677\|2937\|13795\|5359\|3686\|8679\|8513\|158\|14257\|1061\|7393\|10173\|8809\|12157\|469\|3385\|14914\|8702\|7793\|12286\|1185\|11006\|4088\|4630\|14490\|9708\|4166\|13785\|10187\|471\|9754\|727\|605\|9629\|6575\|11528\|8703\|14788\|2311\|4796\|1630\|14072\|8962\|7809\|15387\|4186\|2579\|1798\|779\|559\|1273\|14802\|14756\|14868\|10521\|11588\|4061\|7545\|12934\|217\|331\|12509\|2821\|9923\|13610\|7888\|7360\|228\|4021\|13641\|4521\|2158\|11080\|4374\|5868\|12722\|560\|2666\|6756\|11797\|1924\|7040\|11459\|9052\|7162\|4319\|11260\)'

grep "\"en-train-${SUPPORT_ALARM}\"" "${DATA_DIR}/raw/train.no_exemplars.jsonl" \
  > "${DATA_DIR}/raw/train.100ex-alarm.no_exemplars.jsonl"
grep "\"en-train-${SUPPORT_CALLING}\"" "${DATA_DIR}/raw/train.no_exemplars.jsonl" \
  > "${DATA_DIR}/raw/train.100ex-calling.no_exemplars.jsonl"
grep "\"en-train-${SUPPORT_EVENT}\"" "${DATA_DIR}/raw/train.no_exemplars.jsonl" \
  > "${DATA_DIR}/raw/train.100ex-event.no_exemplars.jsonl"
grep "\"en-train-${SUPPORT_MESSAGING}\"" "${DATA_DIR}/raw/train.no_exemplars.jsonl" \
  > "${DATA_DIR}/raw/train.100ex-messaging.no_exemplars.jsonl"
grep "\"en-train-${SUPPORT_MUSIC}\"" "${DATA_DIR}/raw/train.no_exemplars.jsonl" \
  > "${DATA_DIR}/raw/train.100ex-music.no_exemplars.jsonl"

# Cache the retrievals
for domain in {alarm,calling,event,messaging,music}; do
  # Retrieval index = O_train + N_support
  index_files="${DATA_DIR}/raw/train.except-${domain}.no_exemplars.jsonl"
  index_files+=",${DATA_DIR}/raw/train.100ex-${domain}.no_exemplars.jsonl"

  example_files="${DATA_DIR}/raw/train.except-${domain}.no_exemplars.jsonl"
  example_files+=",${DATA_DIR}/raw/train.100ex-${domain}.no_exemplars.jsonl"
  example_files+=",${DATA_DIR}/raw/dev.except-${domain}.no_exemplars.jsonl"
  example_files+=",${DATA_DIR}/raw/dev.only-${domain}.no_exemplars.jsonl"

  output_files="${DATA_DIR}/raw/train.except-${domain}.use-large.100-shot.jsonl"
  output_files+=",${DATA_DIR}/raw/train.100ex-${domain}.use-large.100-shot.jsonl"
  output_files+=",${DATA_DIR}/raw/dev.except-${domain}.use-large.100-shot.jsonl"
  output_files+=",${DATA_DIR}/raw/dev.only-${domain}.use-large.100-shot.jsonl"

  python -m language.casper.retrieve.cache_query_retrievals \
    --alsologtostderr \
    --retriever=use --embedder_size=large --neighbor_filter=simple \
    --index_files="${index_files}" \
    --example_files="${example_files}" \
    --output_files="${output_files}"

  # Retrieval index = O_train only
  index_files="${DATA_DIR}/raw/train.except-${domain}.no_exemplars.jsonl"
  example_files="${DATA_DIR}/raw/train.except-${domain}.no_exemplars.jsonl"
  output_files="${DATA_DIR}/raw/train.except-${domain}.use-large.0-shot.jsonl"

  python -m language.casper.retrieve.cache_query_retrievals \
    --alsologtostderr \
    --retriever=use --embedder_size=large --neighbor_filter=simple \
    --index_files="${index_files}" \
    --example_files="${example_files}" \
    --output_files="${output_files}"
done
