# TEK Representations
This code is currently not runnable outside Google due to two proprietary dependencies -- (a) an entity linker and (b) a key-value store. We are currently working on open-source alternatives.

The rest of the code assumes all data and models reside in the paths contained in `$data_dir` and `$model_dir`.

## Pre-training
This is how the data for pre-training is created *if an implementation of a key-value is available*.

```
export pretraining_files=$data_dir/pretraining/roberta/msl512_mbg128/wiki.tfrecord
export background_corpus=$data_dir/background/linked_wikipedia.store
python -m language.tek_representations.preprocess.create_pretraining_data \
--input_file=$background_corpus \
--output_file=$pretraining_files \
--background_corpus_file=$background_corpus \
--vocab_file=$model_dir/pretrained/ \
--max_seq_length=512 \
--max_background_len=128 \
```

This is how you run the pre-training:

```
python -m language.tek_representations.run_pretraining \
--do_train \
--do_eval \
--bert_config_file=$model_dir/pretrained/base/roberta/bert_config.json \
--init_checkpoint=$model_dir/pretrained/base/roberta/bert_model.ckpt \
--input_file= $pretraining_files\
--output_dir= $model_dir/pretrained/base/roberta/msl512_mbg128 \
--train_batch_size=512 \
--eval_batch_size=512 \
--learning_rate=5e-05 \
--num_train_steps=200000 \
--save_checkpoints_steps=20000 \
```

## QA Preprocessing

```
export vocab_file=$model_dir/pretrained/
export corpus_file=$data_dir/background/linked_wikipedia.store

# MRQA in-domain
export mrqa_preprocessed=$data_dir/mrqa/webref_preprocessed
export mrqa_annotated=$data_dir/mrqa/webref_annotated

# TEK preprocessing
for dataset in HotpotQA SQuAD TriviaQA-web NewsQA SearchQA NaturalQuestions;do \
mkdir $mrqa_preprocessed/$dataset; \
mkdir $mrqa_preprocessed/$dataset/type.ngram-msl.512-mbg.128; \
python -m language.tek_representations.preprocess.prepare_mrqa_data --input_data_dir=$mrqa_annotated/$dataset --output_data_dir=$mrqa_preprocessed/$dataset/type.ngram-msl.512-mbg.128 --vocab_file=$vocab_file --datasets=$dataset --split=dev --background_type=ngram  --corpus_file=$corpus_file --is_training=False &\
done

for dataset in HotpotQA SQuAD TriviaQA-web NewsQA SearchQA NaturalQuestions;do \
python -m language.tek_representations.preprocess.prepare_mrqa_data --input_data_dir=$mrqa_annotated/$dataset --output_data_dir=$mrqa_preprocessed/$dataset/type.ngram-msl.512-mbg.128 --vocab_file=$vocab_file --datasets=$dataset --split=train --background_type=ngram  --corpus_file=$corpus_file --is_training=True &\
done
```

For preprocessing TriviaQA, additionally set `--include_unknowns=0.02` and change the paths.

Finally, count the features to set the number of steps.

```
python -m  language.tek_representations.preprocess.count_features \
--output_file=$data_dir/mrqa/webref_preprocessed/counts.txt \
--preprocessed_dir=$data_dir/mrqa/webref_preprocessed/
```

## Fine-tuning



The relevant data required for the project is assumed to be in `$model_dir`. All
paths used in the following commands are assumed to be relative to this path.
Train MRQA

```
python -m language.tek_representations.run_mrqa \
--bert_config_file=$model_dir/pretrained/base/roberta//bert_config.json  \
--datasets=NewsQA0SQuAD0HotpotQA0NaturalQuestionsShort0TriviaQA-web0SearchQA \
--do_train \
--do_predict \
--learning_rate=1e-5  \
--num_train_epochs=3 \
--prefix=type.ngram-msl.512-mbg.128 \
--eval_features_file=$data_dir/mrqa/webref_preprocessed/*/type.ngram-msl.512-mbg.128/dev.features* \
--eval_tf_filename=$data_dir/mrqa/webref_preprocessed/*/type.ngram-msl.512-mbg.128/dev.tfrecord*  \
--init_checkpoint=$model_dir/pretrained/base/msl512_mbg128 \
--metrics_file=$model_dir/finetuned/mrqa_base/seed-2.lr-1e-5.epochs-3.cp-msl512_mbg128.ds-all.preprocess-type.ngram-msl.512-mbg.128/_metrics.txt \
--num_train_file=$data_dir/mrqa/webref_preprocessed/counts.txt* \
--output_dir=$model_dir/finetuned/mrqa_base/seed-2.lr-1e-5.epochs-3.cp-msl512_mbg128.ds-all.preprocess-type.ngram-msl.512-mbg.128 \
--output_prediction_file=$model_dir/finetuned/mrqa_base/seed-2.lr-1e-5.epochs-3.cp-msl512_mbg128.ds-all.preprocess-type.ngram-msl.512-mbg.128/_predictions.json \
--predict_file=$data_dir/mrqa/raw_with_ids/dev/*/ \
--train_precomputed_file=$data_dir/mrqa/webref_preprocessed/*/type.ngram-msl.512-mbg.128/train.tfrecord* \
--vocab_file=$model_dir/pretrained/ \
```

Train TriviaQA

```
python -m language.tek_representations.run_mrqa \
--bert_config_file=$model_dir/pretrained/base/roberta//bert_config.json \
--datasets=ShardedTriviaQAWikiTfIdf \
--do_predict \
--do_train \
--triviaqa_eval \
--learning_rate=2e-5 \
--num_train_epochs=5 \
--prefix=type.ngram-msl.512-mbg.128 \
--vocab_file=$model_dir/pretrained/ \
--eval_features_file=$data_dir/triviaqa/webref_preprocessed/ShardedTriviaQAWikiTfIdf/type.ngram-msl.512-mbg.128/dev.features* \
--eval_tf_filename=$data_dir/triviaqa/webref_preprocessed/ShardedTriviaQAWikiTfIdf/type.ngram-msl.512-mbg.128/dev.tfrecord* \
--init_checkpoint=$model_dir/pretrained/base/msl512_mbg128 \
--metrics_file=$model_dir/finetuned/tqa_base/seed-2.lr-2e-5.epochs-5.cp-msl512_mbg128.ds-all.preprocess-type.ngram-msl.512-mbg.128/_metrics.txt \
--num_train_file=$data_dir/triviaqa/webref_preprocessed/counts.txt* \
--output_dir=$model_dir/finetuned/tqa_base/seed-2.lr-2e-5.epochs-5.cp-msl512_mbg128.ds-all.preprocess-type.ngram-msl.512-mbg.128 \
--output_prediction_file=$model_dir/finetuned/tqa_base/seed-2.lr-2e-5.epochs-5.cp-msl512_mbg128.ds-all.preprocess-type.ngram-msl.512-mbg.128/_predictions.json \
--predict_file=$data_dir/triviaqa/raw_with_ids//dev/ShardedTriviaQAWikiTfIdf*\
--train_precomputed_file=$data_dir/triviaqa/webref_preprocessed/ShardedTriviaQAWikiTfIdf/type.ngram-msl.512-mbg.128/train.tfrecord*\
```

Evaluate MRQA out-of-domain by setting `do_train` to `false` and changing the
path of evaluation files to those in the out-of-domain split.

```
python -m language.tek_representations.run_mrqa \
--bert_config_file=$model_dir/pretrained/base/roberta//bert_config.json \
--corpus_file=$data_dir/background/linked_wikipedia.sst \
--datasets=NewsQA0SQuAD0HotpotQA0NaturalQuestionsShort0TriviaQA-web0SearchQA \
--do_predict \
--nodo_train \
--prefix=type.ngram-msl.512-mbg.128 \
--eval_features_file=$data_dir/mrqa_ood/webref_preprocessed/*/type.ngram-msl.512-mbg.128/dev.features* \
--eval_tf_filename=$data_dir/mrqa_ood/webref_preprocessed/*/type.ngram-msl.512-mbg.128/dev.tfrecord* \
--init_checkpoint=$model_dir/pretrained/base/msl512_mbg128 \
--metrics_file=$model_dir/finetuned//mrqa_base/seed-3.lr-1e-5.epochs-3.cp-msl512_mbg128.ds-all.preprocess-type.ngram-msl.512-mbg.128/BioASQ0DROP0DuoRCParaphraseRC0RACE0RelationExtraction0TextbookQA_metrics.txt \
--num_train_file=$data_dir/mrqa/webref_preprocessed/counts.txt* \
--output_dir=$model_dir/finetuned//mrqa_base/seed-3.lr-1e-5.epochs-3.cp-msl512_mbg128.ds-all.preprocess-type.ngram-msl.512-mbg.128 \
--output_prediction_file=$model_dir/finetuned//mrqa_base/seed-3.lr-1e-5.epochs-3.cp-msl512_mbg128.ds-all.preprocess-type.ngram-msl.512-mbg.128/BioASQ0DROP0DuoRCParaphraseRC0RACE0RelationExtraction0TextbookQA_predictions.json \
--predict_file=$data_dir/mrqa_ood/raw_with_ids/dev/*/ \
--train_precomputed_file=$data_dir/mrqa/webref_preprocessed/*/type.ngram-msl.512-mbg.128/train.tfrecord* \
--vocab_file=$model_dir/pretrained/ \
```
