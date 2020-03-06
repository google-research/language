# Open Retrieval Question Answering (ORQA)
This directory contains code for the paper for ORQA:

> [Latent Retrieval for Weakly Supervised Open Domain Question Answering](https://arxiv.org/abs/1906.00300)

> Kenton Lee, Ming-Wei Chang, Kristina Toutanova

> In ACL 2019

## Requirements
We assume Python 3 and TensorFlow 1.14 or above for all code in this project.

## Getting the data
Download the Google AI Language repository for preprocessing code.

```bash
git clone https://github.com/google-research/language
cd language
```

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

export RESPLIT_PATH=<PATH_TO_FINAL_RESPLIT_DATA>
```bash
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
{
  "question": "what type of fuel goes in a zippo",
  "answer": ["lighter fluid", "butane"]
}
```

## Evaluation
Format your predictions as a jsonlines file, where each line is a JSON
dictionary with the following format:

```
{
  "question": "what type of fuel goes in a zippo",
  "prediction": "butane"
}
```

Run the evaluation script with paths to the references and predictions as
arguments:

```bash
python -m language.orqa.evaluation.evaluate_predictions \
  --references_path=<PATH_TO_REFERENCES_FILE>
  --predictions_path=<PATH_TO_PREDICTIONS_FILE>
```

CuratedTrec references are formatted as regular expression, and
`--is_regex=true` should be passed in as an argument in that case.

## Modeling
Coming soon!
