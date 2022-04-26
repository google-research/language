# MultiBERTs

MultiBERTs is a collection of checkpoints and a statistical library to
support robust research on BERT. In particular, we provide 25 reproductions of BERT pre-training which allow one to distinguish findings that apply to a specific _artifact_ (i.e., a particular instance of the model) from those that apply to the more general _procedure_ (which includes the model architecture, training data, and loss function).

Concretely, the release includes three components:

* A set of 25 BERT-Base models (English, uncased), trained with the same hyper-parameters but different random seeds.
* For the first five models, 28 checkpoints captured during the course of pre-training (140 checkpoints total).
* A statistical library (`multibootstrap.py`) and notebook examples to demonstrate its use.

We describe the release in detail and present example analyses in the [MultiBERTs paper](https://arxiv.org/abs/2106.16163), published at ICLR 2022. All the data and checkpoints mentioned on this page and in the paper are available on [our Cloud Bucket](https://console.cloud.google.com/storage/browser/multiberts/public).



## Models

We release 25 English BERT-Base models (12 Transformer layers, hidden embeddings size 768, 12 attention heads, 110M parameters). They are directly compatible with the original [BERT repository](https://github.com/google-research/bert), and should be used as described on the repository's [README](https://github.com/google-research/bert#readme). We replicated the training configuration of the original [BERT-base](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip), with a few differences highlighted in the [MultiBERTs paper](https://arxiv.org/abs/2106.16163).

The models should be used with the [BERT config file](https://storage.googleapis.com/multiberts/public/bert_config.json)
and [WordPiece vocab file](https://storage.googleapis.com/multiberts/public/vocab.txt).
You may acess the models through the [Cloud Browser](https://console.cloud.google.com/storage/browser/multiberts/public/models)
or with the following links:

| | | | | |
|:---:|:---:|:---:|:---:|:---:|
| [0] | [1] | [2] | [3] | [4] |
| [5] | [6] | [7] | [8] | [9] |
| [10] | [11] | [12] | [13] | [14] |
| [15] | [16] | [17] | [18] | [19] |
| [20] | [21] | [22] | [23] | [24] |


[0]: https://storage.googleapis.com/multiberts/public/models/seed_0.zip
[1]: https://storage.googleapis.com/multiberts/public/models/seed_1.zip
[2]: https://storage.googleapis.com/multiberts/public/models/seed_2.zip
[3]: https://storage.googleapis.com/multiberts/public/models/seed_3.zip
[4]: https://storage.googleapis.com/multiberts/public/models/seed_4.zip
[5]: https://storage.googleapis.com/multiberts/public/models/seed_5.zip
[6]: https://storage.googleapis.com/multiberts/public/models/seed_6.zip
[7]: https://storage.googleapis.com/multiberts/public/models/seed_7.zip
[8]: https://storage.googleapis.com/multiberts/public/models/seed_8.zip
[9]: https://storage.googleapis.com/multiberts/public/models/seed_9.zip
[10]: https://storage.googleapis.com/multiberts/public/models/seed_10.zip
[11]: https://storage.googleapis.com/multiberts/public/models/seed_11.zip
[12]: https://storage.googleapis.com/multiberts/public/models/seed_12.zip
[13]: https://storage.googleapis.com/multiberts/public/models/seed_13.zip
[14]: https://storage.googleapis.com/multiberts/public/models/seed_14.zip
[15]: https://storage.googleapis.com/multiberts/public/models/seed_15.zip
[16]: https://storage.googleapis.com/multiberts/public/models/seed_16.zip
[17]: https://storage.googleapis.com/multiberts/public/models/seed_17.zip
[18]: https://storage.googleapis.com/multiberts/public/models/seed_18.zip
[19]: https://storage.googleapis.com/multiberts/public/models/seed_19.zip
[20]: https://storage.googleapis.com/multiberts/public/models/seed_20.zip
[21]: https://storage.googleapis.com/multiberts/public/models/seed_21.zip
[22]: https://storage.googleapis.com/multiberts/public/models/seed_22.zip
[23]: https://storage.googleapis.com/multiberts/public/models/seed_23.zip
[24]: https://storage.googleapis.com/multiberts/public/models/seed_24.zip

You may download all the models as follows:

```
for ckpt in {0..24} ; do
  wget "https://storage.googleapis.com/multiberts/public/models/seed_${ckpt}.zip"
  unzip "seed_${ckpt}.zip"
done

```



## Intermediate Checkpoints

We release checkpoints captured during the course of pre-training for the first five models. The aim is to support researchers interested in learning dynamics. We saved a checkpoint every 20,000 training steps up to 200,000 steps, then every 100,000 steps up to 2 million (1 training step = 256 sequences of 512 tokens).


| | | | | |
|:---:|:---:|:---:|:---:|:---:|
| [0](https://storage.googleapis.com/multiberts/public/intermediates/seed_0.zip) | [1](https://storage.googleapis.com/multiberts/public/intermediates/seed_1.zip) | [2](https://storage.googleapis.com/multiberts/public/intermediates/seed_2.zip) | [3](https://storage.googleapis.com/multiberts/public/intermediates/seed_3.zip) | [4](https://storage.googleapis.com/multiberts/public/intermediates/seed_4.zip) |

You may download all the models as follows:

```
for ckpt in {0..4} ; do
  wget "https://storage.googleapis.com/multiberts/public/intermediates/seed_${ckpt}.zip"
  unzip "seed_${ckpt}.zip"
done

```

Note: the archives are large (>10 GB). You may download the checkpoints selectively by using the
[Cloud Storage browser](https://console.cloud.google.com/storage/browser/multiberts/public/intermediates) instead.


## Statistical Library

[`multibootstrap.py`](https://github.com/google-research/language/blob/master/language/multiberts/multibootstrap.py) is our implementation of the Multi-Bootstrap, a non-parametric procedure to help researchers estimate significance and report confidence intervals when working with multiple pretraining seeds.

Additional details are provided in the [MultiBERTs paper](https://arxiv.org/pdf/2106.16163). The following notebooks also demonstrate example usage, and will reproduce the results from the paper:

- [`coref.ipynb`](https://github.com/google-research/language/blob/master/language/multiberts/coref.ipynb) - Winogender coreference example from Section 4 and Appendix D of the paper; includes both paired and unpaired examples.
- [`2m_vs_1m.ipynb`](https://github.com/google-research/language/blob/master/language/multiberts/2m_vs_1m.ipynb.ipynb) - Paired analysis from Appendix E.1 of the paper, comparing 2M vs 1M steps of pretraining.
- [`multi_vs_original.ipynb`](https://github.com/google-research/language/blob/master/language/multiberts/multi_vs_original.ipynb.ipynb) - Unpaired analysis from Appendix E.2 of the paper, comparing MultiBERTs to the original BERT release.


## How to cite

```
@inproceedings{sellam2022multiberts,
  title={The Multi{BERT}s: {BERT} Reproductions for Robustness Analysis},
  author={Thibault Sellam and Steve Yadlowsky and Ian Tenney and Jason Wei and Naomi Saphra and Alexander D'Amour and Tal Linzen and Jasmijn Bastings and Iulia Raluca Turc and Jacob Eisenstein and Dipanjan Das and Ellie Pavlick},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=K0E_F0gFDgA}
}
```

## Contact information

If you have a technical question regarding the dataset, code, or publication, please send us an email (see paper).


## Disclaimer

This is not an official Google product.

