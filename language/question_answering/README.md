# Getting Started
**Create a directory for downloaded data:**

```
export DATA_DIR=data
mkdir -p $DATA_DIR
```

**Download the Natural Questions data:**

```
gsutil -m cp -r gs://natural_questions $DATA_DIR
export NQ_DATA_DIR=$DATA_DIR/natural_questions/v1.0
```

**Preprocess data for the long answer model:**

```
python -m language.question_answering.preprocessing.create_nq_long_examples \
  --input_pattern=$NQ_DATA_DIR/dev/nq-dev-*.jsonl.gz \
  --output_dir=$NQ_DATA_DIR/dev
python -m language.question_answering.preprocessing.create_nq_long_examples \
  --input_pattern=$NQ_DATA_DIR/train/nq-train-*.jsonl.gz \
  --output_dir=$NQ_DATA_DIR/train
```

**Preprocess data for the short answer pipeline model:**

```
python -m language.question_answering.preprocessing.create_nq_short_pipeline_examples \
  --input_pattern=$NQ_DATA_DIR/dev/nq-dev-*.jsonl.gz \
  --output_dir=$NQ_DATA_DIR/dev
python -m language.question_answering.preprocessing.create_nq_short_pipeline_examples \
  --input_pattern=$NQ_DATA_DIR/train/nq-train-*.jsonl.gz \
  --output_dir=$NQ_DATA_DIR/train
```

**Download pre-trained word embeddings:**

```
curl https://nlp.stanford.edu/data/glove.840B.300d.zip > $DATA_DIR/glove.840B.300d.zip
unzip $DATA_DIR/glove.840B.300d.zip -d $DATA_DIR
```

**Create a directory for the models:**

```
export MODELS_DIR=models
mkdir -p $MODELS_DIR
```

**Train the long answer model.**

```
python -m language.question_answering.experiments.nq_long_experiment \
  --embeddings_path=$DATA_DIR/glove.840B.300d.txt \
  --nq_long_train_pattern=$NQ_DATA_DIR/train/nq-train-*.long.tfr \
  --nq_long_eval_pattern=$NQ_DATA_DIR/dev/nq-dev-*.long.tfr \
  --num_eval_steps=100 \
  --batch_size=4 \
  --model_dir=$MODELS_DIR/nq_long
```

**Train the short answer pipeline model.**

```
python -m language.question_answering.experiments.nq_short_pipeline_experiment \
  --embeddings_path=$DATA_DIR/glove.840B.300d.txt \
  --nq_short_pipeline_train_pattern=$NQ_DATA_DIR/train/nq-train-*.short_pipeline.tfr \
  --nq_short_pipeline_eval_pattern=$NQ_DATA_DIR/dev/nq-dev-*.short_pipeline.tfr \
  --num_eval_steps=10 \
  --model_dir=$MODELS_DIR/nq_short_pipeline
```
