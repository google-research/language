# Embedding Query Language (EmQL)

This is the codebase for NeurIPS 2020 paper "[Faithful Embeddings for Knowledge Base Queries](https://arxiv.org/abs/2004.03658)".

## Preprocess
Please download the original data and run the preprocess scripts. You should have the following directory structure after preprocessing:

```
emql
├── datasets
  ├── MetaQA
    ├── 1hop
    ├── 2hop
    ├── 3hop
  ├── WebQSP
  └── Query2Box
      ├── FB15k
      ├── FB15k-237
      └── NELL995
```

#### MetaQA
1. Download [MetaQA](https://github.com/yuyuz/MetaQA) dataset into
    ```
    ./datasets/raw/MetaQA
    ```

2. Run preprocessing script
    ```
    $ python3 preprocess/metaqa_preprocess.py --meta_dir=./datasets/raw/metaqa/ --output_dir=./dataset/MetaQA
    ```

3. Copy kb.txt to MetaQA 1hop sub-directories (same for 2hop/ and 3hop/)
    ```
    $ cp ./datasets/raw/MetaQA/kb.txt ./datasets/MetaQA/1hop/
    ```

#### WebQuestionsSP
1. Download [WebQuestionsSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763) dataset into
    ```
    ./datasets/WebQSP
    ```

#### Query2Box
1. Download [Query2Box](https://github.com/hyren/query2box/tree/master/data) dataset into
    ```
    ./datasets/Query2Box
    ```

2. Run preprocessing script
    ```
    $ python3 preprocess/query2box_preprocess.py --query2box_dir=./datasets/query2box/ --output_dir=./datasets/query2box
    ```

#### Download BERT vocab
    ```
    $ wget -O preprocess/bert_vocab.txt https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
    ```

##  Experiments

#### Compositional Query Language (Query2Box)
First pretrain the KB embeddings using the task name "mixture", i.e. jointly training 4 sub-tasks: set encoding (and decoding), intersection, union, and relational following.

```
python3 main.py --mode=train --name=mixture --model_name=FB15k-mixture \
--batch_size=8 --epochs=50 --train_entity_emb=True --train_relation_emb=True \
--root_dir=./datasets/Query2Box/FB15k/ --kb_file=kb.txt --checkpoint_dir=/tmp/
```

Then run evaluation on Query2Box tasks. To evaluate on all tasks, you may want to run it from a bash file.

```
for task in '1c' '2c' '3c' '2i' '3i' 'ic' 'ci' '2u' 'uc'
do
    python3 main.py --mode=pred --name=query2box_$task --model_name=query2box-eval \
    --batch_size=10 --root_dir=./datasets/Query2Box/FB15k/ --kb_file=kb.txt \
    --checkpoint_dir=/tmp/ --load_model_dir=/tmp/FB15k-mixture/ \
    --intermediate_top_k=1000 --cm_width=5000 --num_eval=-1 --use_cm_sketch=False
done

```

A few hyper-parameters to note here:
1. intermediate_top_k: number of facts to retrieve at each step. You can improve recall with a large k, but will decrease the efficiency.
2. cm_width: cm_width should be at least 2 * intermediate_top_k to avoid collisions. You may also change cm_depth accordingly.
3. use_cm_sketch: decide if count-min sketches are applied at intermediate following steps. If True, fact retrieval results will be filtered by their subject entities using thee count-min sketch computed from the previous step.
4. kb_file: one may train on kb.txt and evaluate on kb_test.txt (one experiment presented in the paper).
5. You can also set embedding dimensions to custom values (see flags).


#### KBQA

First pretrain the KB embeddings.

```
python3 main.py --mode=train --name=mixture --model_name=metaqa-mixture \
--batch_size=24 --hidden_size=64 --num_online_eval=100 --eval_time=300 --epochs=50 \
--train_entity_emb=True --train_relation_emb=True \
--root_dir=./datasets/WikiMovies/ --checkpoint_dir=/tmp/
```

Then finetune on MetaQA datasets.
```
python3 main.py --mode=train --name=metaqa3 --model_name=MetaQA-3hop \
--batch_size=64 --hidden_size=64 --cm_width=200 --intermediate_top_k=20 \
--root_dir=./datasets/ --checkpoint_dir=/tmp/ --load_model_dir=/tmp/metaqa-mixture
```
You may want to increase the intermediate_top_k to improve the recall at each intermediate steps. From our experiments, we observe that having a large intermediate_top_k usually leads to better performance. Please remember increase the cm_width to avoid collisions with large intermediate_top_k.
