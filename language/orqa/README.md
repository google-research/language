# Open Retrieval Question Answering (ORQA)
This directory contains code for the paper for ORQA:

> [Latent Retrieval for Weakly Supervised Open Domain Question Answering](https://arxiv.org/abs/1906.00300)

> Kenton Lee, Ming-Wei Chang, Kristina Toutanova

> In ACL 2019

The main code is in the Google AI Language repository:

```bash
git clone https://github.com/google-research/language
cd language
```

## Requirements
We require Python 3.7, TensorFlow 2.1.0, and ScaNN 1.0:

```bash
conda create --name orqa python=3.7
source activate orqa
pip install -r language/orqa/requirements.txt
```

Run a unit test to make ensure basic dependencies were installed correctly:

```bash
python -m language.orqa.utils.scann_utils_test
```

## Getting the data

### WebQuestions and CuratedTrec
Download the data from DrQA:

```bash
git clone https://github.com/facebookresearch/DrQA.git
cd DrQA
export DRQA_PATH=$(pwd)
sh download.sh
```

### Natural Questions (Open)
Install [gsutil](https://cloud.google.com/storage/docs/gsutil_install) and
download the data from the Natural Questions cloud bucket:

```bash
mkdir original_nq
gsutil -m cp -R gs://natural_questions/v1.0 original_nq
cd original_nq
export ORIG_NQ_PATH=$(pwd)
```

Run preprocessing code for stripping away everything except question-answer
pairs with short answers containing at most five tokens:

```bash
python -m language.orqa.preprocessing.convert_to_nq_open \
  --logtostderr \
  --input_pattern=$ORIG_NQ_PATH/v1.0/train/nq-*.jsonl.gz \
  --output_path=$ORIG_NQ_PATH/open/NaturalQuestions-train.txt
python -m language.orqa.preprocessing.convert_to_nq_open \
  --logtostderr \
  --input_pattern=$ORIG_NQ_PATH/v1.0/dev/nq-*.jsonl.gz \
  --output_path=$ORIG_NQ_PATH/open/NaturalQuestions-dev.txt
```

### Resplitting the data
None of the datasets have publically available train/dev/test splits, so we
create our own:

```bash
export RESPLIT_PATH=<PATH_TO_FINAL_RESPLIT_DATA>
python -m language.orqa.preprocessing.create_data_splits \
  --logtostderr \
  --nq_train_path=$ORIG_NQ_PATH/open/NaturalQuestions-train.txt \
  --nq_dev_path=$ORIG_NQ_PATH/open/NaturalQuestions-dev.txt \
  --wb_train_path=$DRQA_PATH/data/datasets/WebQuestions-train.txt \
  --wb_test_path=$DRQA_PATH/data/datasets/WebQuestions-test.txt \
  --ct_train_path=$DRQA_PATH/data/datasets/CuratedTrec-train.txt \
  --ct_test_path=$DRQA_PATH/data/datasets/CuratedTrec-test.txt \
  --output_dir=$RESPLIT_PATH
```

Expect to find the following number of examples in each split:

|| Train | Dev | Test |
|---|---|---|---|
|Natural Questions (open)| 79168 | 8757 | 3610
|WebQuestions | 3417 | 361 | 2032
|CuratedTrec | 1353 | 133 | 694

Each line in data is a JSON dictionary with the following format:

```
{ "question": "what type of fuel goes in a zippo", "answer": ["lighter fluid", "butane"] }
```

The result of this resplitting for Natural Questions and WebQuestions can be
found at `gs://orqa-data/resplit`.

## Evaluation
Format your predictions as a jsonlines file, where each line is a JSON
dictionary with the following format:

```
{ "question": "what type of fuel goes in a zippo", "prediction": "butane" }
```

Run the evaluation script with paths to the references and predictions as
arguments:

```bash
python -m language.orqa.evaluation.evaluate_predictions \
  --references_path=<PATH_TO_REFERENCES_FILE> \
  --predictions_path=<PATH_TO_PREDICTIONS_FILE>
```

CuratedTrec references are formatted as regular expression, and
`--is_regex=true` should be passed in as an argument in that case.

## Modeling

### Preprocessing Wikipedia

Download Wikipedia and use WikiExtractor to remove everything but raw text:

```bash
wget https://archive.org/download/enwiki-20181220/enwiki-20181220-pages-articles.xml.bz2
INPUT_PATH=$(pwd)/enwiki-20181220-pages-articles.xml.bz2
OUTPUT_PATH=$(pwd)/enwiki-20181220
python -m wikiextractor.WikiExtractor \
  -o $OUTPUT_PATH \
  --json \
  --filter_disambig_pages \
  --quiet \
  --processes 12 \
  $INPUT_PATH
```

Convert those raw texts into blocks of text, which is used both for pre-training
and as a database for retrieval:

```bash
python -m language.preprocessing.preprocess_wiki_extractor \
 --input_pattern=<PATH_TO_WIKI_EXTRACTED_DIR> \
 --output_pattern=<PATH_TO_DATA_BASE_DIR>
```

* `blocks.tfr`: A TFRecords file where each entry is a string representing a block of text from Wikipedia.
* `titles.tfr` : A TFRecords file where the i'th entry is the title of the page to which the i'th block belongs.
* `examples.tfr` : A TFRecords file where the i'th entry is a tf.train.Example with the pre-tokenized title, block, and sentence breaks of the i'th block.

The result of running `preprocess_wiki_extractor` on the December 20th 2018
version of Wikipedia is available at `gs://orqa-data/enwiki-20181220`.

### Inverse Cloze Task (ICT) pre-training:

We recommend using TPUs for ICT pre-training due to the effectiveness of large
batch sizes (we used 4096 in the paper).

##### Training on TPU

```bash
MODEL_DIR=gs://<YOUR_BUCKET>/<ICT_MODEL_DIR>
TFHUB_CACHE_DIR=gs://<YOUR_BUCKET>/<TFHUB_CACHE_DIR>
TFHUB_CACHE_DIR=$TFHUB_CACHE_DIR \
TPU_NAME=<NAME_OF_TPU>
python -m language.orqa.experiments.ict_experiment \
  --model_dir=$MODEL_DIR \
  --bert_hub_module_path=https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1 \
  --examples_path=gs://orqa-data/enwiki-20181220/examples.tfr \
  --save_checkpoints_steps=1000 \
  --batch_size=4096 \
  --num_train_steps=100000 \
  --tpu_name=$TPU_NAME \
  --use_tpu=True
```

##### Continuous evaluation on GPU

```bash
MODEL_DIR=gs://<YOUR_BUCKET>/<ICT_MODEL_DIR>
TF_CONFIG='{"cluster": {"chief": ["host:port"]}, "task": {"type": "evaluator", "index": 0}}' \
python -m language.orqa.experiments.ict_experiment \
  --model_dir=$MODEL_DIR \
  --bert_hub_module_path=https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1 \
  --examples_path=gs://orqa-data/enwiki-20181220/examples.tfr \
  --batch_size=32 \
  --num_eval_steps=1 \
```

### Compute the dense vector index over Wikipedia:

Computing the dense vector index can be done via TPU, GPU, or embarrassingly
parallel CPU computation. The following command is for TPU indexing.

```bash
MODULE_PATH=gs://<YOUR_BUCKET>/<ICT_MODEL_DIR>/export/tf_hub/<TIMESTAMP>/ict
TPU_NAME=<NAME_OF_TPU>
python -m language.orqa.predict.encode_blocks \
  --retriever_module_path=$MODULE_PATH \
  --examples_path=gs://orqa-data/enwiki-20181220/examples.tfr \
  --tpu_name=$TPU_NAME \
  --use_tpu=True
```

The result of the pre-trained ICT model along with the dense vector index that
would be found at the `encoded` directory is available at `gs://orqa-data/ict`.

### Open Retrieval Question Answering (ORQA) fine-tuning:

Due to the complexities of dealing with regular expressions on the fly in the
middle of a neural network model (required for CuratedTrec), we only release
support for WebQuestions and Natural Questions.

#### Compiling custom ops
ORQA requires a couple of custom ops used for string and token manipulation:

```bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared language/orqa/ops/orqa_ops.cc -o language/orqa/ops/orqa_ops.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```

Run the unit test to make sure those ops were compiled properly:

```bash
python -m language.orqa.ops.orqa_ops_test
```

#### Fine-tuning

For fine-tuning, we recommend a machine with a 12GB ram GPU and 64GB ram CPU.

The following example uses the WebQuestions dataset:

##### Training on GPU

```bash
MODEL_DIR=<LOCAL_OR_GS_MODEL_DIR>
TF_CONFIG='{"cluster": {"chief": ["host:port"]}, "task": {"type": "chief", "index": 0}}' \
python -m language.orqa.experiments.orqa_experiment \
  --retriever_module_path=gs://orqa-data/ict \
  --block_records_path=gs://orqa-data/enwiki-20181220/blocks.tfr \
  --data_root=gs://orqa-data/resplit \
  --model_dir=$MODEL_DIR \
  --dataset_name=WebQuestions \
  --num_train_steps=$(( 3417 * 20 )) \
  --save_checkpoints_steps=1000
```

##### Continuous dev evaluation GPU

```bash
MODEL_DIR=<LOCAL_OR_GS_MODEL_DIR>
TF_CONFIG='{"cluster": {"chief": ["host:port"]}, "task": {"type": "evaluator", "index": 0}}' \
python -m language.orqa.experiments.orqa_experiment \
  --retriever_module_path=gs://orqa-data/ict \
  --block_records_path=gs://orqa-data/enwiki-20181220/blocks.tfr \
  --data_root=gs://orqa-data/resplit \
  --model_dir=$MODEL_DIR \
  --dataset_name=WebQuestions
```

##### Final test evaluation on GPU

```bash
MODEL_DIR=<LOCAL_OR_GS_MODEL_DIR>
python -m language.orqa.experiments.orqa_eval \
  --retriever_module_path=gs://orqa-data/ict \
  --block_records_path=gs://orqa-data/enwiki-20181220/blocks.tfr \
  --dataset_path=gs://orqa-data/resplit/WebQuestions.resplit.test.jsonl \
  --model_dir=$MODEL_DIR
```


#### Running the demo:

Trained WebQuestions and Natural Questions models are available at
`gs://orqa-data/orqa_nq_model` and `gs://orqa-data/orqa_wq_model` respectively.
Note that these models are about 1 point below published numbers due to training
variance and slight implementation differences due to open-sourcing constraints.
To try the web demo with the Natural Questions model, run the following and
point your browser to `<IP_ADDRESS>:8080`.

```bash
python -m language.orqa.predict.orqa_demo \
  --model_dir=gs://orqa-data/orqa_nq_model \
  --port=8080
```
