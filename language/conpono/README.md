# Conpono: Pretraining with Contrastive Sentence Objectives Improves Discourse Performance of Language Models

Dan Iter (Stanford - Work done at Google), Kelvin Guu (Google), Larry Lansing (Google), Dan Jurafsky (Stanford)

This work will appear at ACL 2020 (https://arxiv.org/abs/2005.10389)

## Abstract

Conpono stands for "Contrastive Position and Ordering with Negatives Objective".
It is an inter-sentence objective that is trained at the same time as a masked language model.
The resulting model will have similar properties to BERT-Base but with a superior discourse-level representation
in the pooled representation output.
We currently have a paper under review with evaluations on DiscoEval, RTE, COPA and ReCoRD.

## Software

Running most modeling code depends on having BERT installed (https://github.com/google-research/bert)

`cpc/` - The main codebase for the Conpono model, including the various alternative
architectures, isolated and uni-encoders. `cpc/preproc` contains scripts for converting
text data into tfrecord for easing feeding into our model.

`evals/` - contains evaluation code for coherence, classification, multiple-choice
RACE, ReCord, SQuAD, and others. Does not contain evaluation code for DiscoEval.

`binary_order/` - contains code for training and evaluating a binary classifier
for sentence ordering.

`create_pretrain_data/` - beam jobs for creating Wikipedia and BooksCorpus
datasets and scripts with run details. These are not the final scripts for our
Conpono Model. See `cpc/preproc` for those scripts.

`reconstruct/` - contains code for a predecessor to Conpono that reorders 5
sentences into its original order.


## Model weights

For window size k=2
https://storage.googleapis.com/conpono/conpono_k2/graph.pbtxt
https://storage.googleapis.com/conpono/conpono_k2/model.ckpt.data-00000-of-00001
https://storage.googleapis.com/conpono/conpono_k2/model.ckpt.index
https://storage.googleapis.com/conpono/conpono_k2/model.ckpt.meta

for window size k=4
https://storage.googleapis.com/conpono/conpono_k4/graph.pbtxt
https://storage.googleapis.com/conpono/conpono_k4/model.ckpt.data-00000-of-00001
https://storage.googleapis.com/conpono/conpono_k4/model.ckpt.index
https://storage.googleapis.com/conpono/conpono_k4/model.ckpt.meta
