# Model Extraction of BERT-based APIs

This folder contains the original codebase used to conduct the experiments in the ICLR 2020 paper *[Thieves on Sesame Street! Model Extraction of BERT-based APIs](https://arxiv.org/abs/1910.12366)*. The OpenReview discussion for this paper can be found [here](https://openreview.net/forum?id=Byl5NREFDr).

## Setup

Please follow the setup in [google-research/language](https://github.com/google-research/language). This codebase requires [google-research/bert](https://github.com/google-research/bert) for all its experiments.

## Experiments on SST2, MNLI

Please find more details in [`steal_bert_classifier/README.md`](steal_bert_classifier/README.md). The codebase can be trivially modified for any classification task using BERT expecting a single sentence input or a pair of sentences as input.

## Experiments on SQuAD 1.1, SQuAD 2.0, BoolQ

Please find more details in [`steal_bert_qa/README.md`](steal_bert_qa/README.md).

## Citation

If you find this paper or codebase useful, please cite us.

```
@inproceedings{krishna2020thieves,
  title={Thieves on Sesame Street! Model Extraction of BERT-based APIs},
  author={Krishna, Kalpesh and Tomar, Gaurav Singh and Parikh, Ankur P and Papernot, Nicolas and Iyyer, Mohit},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```


