# Experiment data

The tables below list the training and evaluation data for each experiment in the paper.

* The **finetune_on** column specifies the subdirectories in `${DATA_DIR}/t5/` to get training data from. The model should be fine-tuned on the `train.*` files from such subdirectories, with a uniform mixture (i.e., an equal chance of picking each directory).

* The **predict_on** column specifies the evaluation data. The model should perform inference on the `dev.*` files from the specified subdirectory (or the `test.*` files for test set evaluation in the standard setup).

## Table 1 / Table 7: Standard setup

| experiment | finetune_on | predict_on |
| --- | --- | --- |
| T5 | `noaug` | `noaug` |
| CASPER | `samp5x20`, `samp5x20.anon` | `top5` |
| CASPER_orig | `samp5x20` | `top5` |
| CASPER_anon | `samp5x20.anon` | `top5.anon` |

## Table 3: Ablation

| experiment | finetune_on | predict_on |
| --- | --- | --- |
| USE-large, sampled-5 | `samp5x20` | `top5` |
| none (baseline T5) | `noaug` | `noaug` |
| BERT-base, sampled-5 | `samp5x20.bert-base` | `top5.bert-base` |
| BERT-large, sampled-5  | `samp5x20.bert-large` | `top5.bert-large` |
| oracle, sampled-5 | `samp5x20.oracle` | `top5.oracle` |
| USE-large, top-1 | `top1` | `top1` |
| USE-large, top-5 | `top5` | `top5` |
| USE-large, sampled-1 | `samp1x20` | `top1` |
| USE-large, sampled-3 | `samp3x20` | `top3` |
| USE-large, sampled-10 | `samp10x20` | `top10` |

## Table 4 / Table 10: Domain bootstrapping

The variable `${domain}` is one of the following: `alarm`, `calling`, `event`, `messaging`, `music`.

| experiment | finetune_on | predict_on |
| --- | --- | --- |
| T5, standard, eval on N_dev | `noaug` | `only-${domain}.noaug` |
| T5, standard, eval on O_dev | `noaug` | `except-${domain}.noaug` |
| T5, seen-boot., eval on N_dev | `except-${domain}.noaug`, `100ex-${domain}.noaug` | `only-${domain}.noaug` |
| T5, seen-boot., eval on O_dev | `except-${domain}.noaug`, `100ex-${domain}.noaug` | `except-${domain}.noaug` |
| T5, unseen-boot., eval on N_dev | `except-${domain}.noaug` | `only-${domain}.noaug` |
| T5, unseen-boot., eval on O_dev | `except-${domain}.noaug` | `except-${domain}.noaug` |
| CASPER, standard, eval on N_dev | `samp5x20`, `samp5x20.anon` | `only-${domain}.100-shot.top5` |
| CASPER, standard, eval on O_dev | `samp5x20`, `samp5x20.anon` | `except-${domain}.100-shot.top5` |
| CASPER, seen-boot., eval on N_dev | `except-${domain}.100-shot.samp5x20`, `except-${domain}.100-shot.samp5x20.anon`, `100ex-${domain}.100-shot.samp5x20`, `100ex-${domain}.100-shot.samp5x20.anon` | `only-${domain}.100-shot.top5` |
| CASPER, seen-boot., eval on O_dev | `except-${domain}.100-shot.samp5x20`, `except-${domain}.100-shot.samp5x20.anon`, `100ex-${domain}.100-shot.samp5x20`, `100ex-${domain}.100-shot.samp5x20.anon` | `except-${domain}.100-shot.top5` |
| CASPER, unseen-boot., eval on N_dev | `except-${domain}.0-shot.samp5x20`, `except-${domain}.0-shot.samp5x20.anon` | `only-${domain}.100-shot.top5` |
| CASPER, unseen-boot., eval on O_dev | `except-${domain}.0-shot.samp5x20`, `except-${domain}.0-shot.samp5x20.anon` | `except-${domain}.100-shot.top5` |
| CASPER_orig, standard, eval on N_dev | `samp5x20` | `only-${domain}.100-shot.top5` |
| CASPER_orig, standard, eval on O_dev | `samp5x20` | `except-${domain}.100-shot.top5` |
| CASPER_orig, seen-boot., eval on N_dev | `except-${domain}.100-shot.samp5x20`, `100ex-${domain}.100-shot.samp5x20` | `only-${domain}.100-shot.top5` |
| CASPER_orig, seen-boot., eval on O_dev | `except-${domain}.100-shot.samp5x20`, `100ex-${domain}.100-shot.samp5x20` | `except-${domain}.100-shot.top5` |
| CASPER_orig, unseen-boot., eval on N_dev | `except-${domain}.0-shot.samp5x20` | `only-${domain}.100-shot.top5` |
| CASPER_orig, unseen-boot., eval on O_dev | `except-${domain}.0-shot.samp5x20` | `except-${domain}.100-shot.top5` |
| CASPER_anon, standard, eval on N_dev | `samp5x20.anon` | `only-${domain}.100-shot.top5.anon` |
| CASPER_anon, standard, eval on O_dev | `samp5x20.anon` | `except-${domain}.100-shot.top5.anon` |
| CASPER_anon, seen-boot., eval on N_dev | `except-${domain}.100-shot.samp5x20.anon`, `100ex-${domain}.100-shot.samp5x20.anon` | `only-${domain}.100-shot.top5.anon` |
| CASPER_anon, seen-boot., eval on O_dev | `except-${domain}.100-shot.samp5x20.anon`, `100ex-${domain}.100-shot.samp5x20.anon` | `except-${domain}.100-shot.top5.anon` |
| CASPER_anon, unseen-boot., eval on N_dev | `except-${domain}.0-shot.samp5x20.anon` | `only-${domain}.100-shot.top5.anon` |
| CASPER_anon, unseen-boot., eval on O_dev | `except-${domain}.0-shot.samp5x20.anon` | `except-${domain}.100-shot.top5.anon` |

## Figure 4: Fast update

Fine-tuning should be done on top of the best checkpoint of "T5, unseen-boot."

| experiment | finetune_on | predict_on |
| --- | --- | --- |
| T5: O_train → N_sup, eval on N_dev | `100ex-${domain}.noaug` | `only-${domain}.noaug` |
| T5: O_train → N_sup, eval on O_dev | `100ex-${domain}.noaug` | `except-${domain}.noaug` |
| T5: O_train → O_train + N_sup, eval on N_dev | `except-${domain}.noaug`, `100ex-${domain}.noaug` | `only-${domain}.noaug` |
| T5: O_train → O_train + N_sup, eval on O_dev | `except-${domain}.noaug`, `100ex-${domain}.noaug` | `except-${domain}.noaug` |

## Table 5 / Table 8: Parse guiding

| experiment | finetune_on | predict_on |
| --- | --- | --- |
| T5, standard eval | `noaug` | `noaug` |
| CASPER, standard eval | `samp5x20`, `samp5x20.anon` | `top5` |
| CASPER, oracle eval | `samp5x20`, `samp5x20.anon` | `ora5xT` |
| CASPER + oracle train + no tag at test time, standard eval | `samp5x20`, `samp5x20.anon`, `ora5x20.plat`, `ora5x20.anon.plat` | `top5` |
| CASPER + oracle train + no tag at test time, oracle eval | `samp5x20`, `samp5x20.anon`, `ora5x20.plat`, `ora5x20.anon.plat` | `ora5xT` |
| CASPER + oracle train + add tag at test time, standard eval | `samp5x20`, `samp5x20.anon`, `ora5x20.plat`, `ora5x20.anon.plat` | `top5.plat` |
| CASPER + oracle train + add tag at test time, oracle eval | `samp5x20`, `samp5x20.anon`, `ora5x20.plat`, `ora5x20.anon.plat` | `ora5xT.plat` |

## Section 5.2: Use oracle training data but without guiding tag during training

| experiment | finetune_on | predict_on |
| --- | --- | --- |
| CASPER + oracle train without guiding tag + no tag at test time, oracle eval | `samp5x20`, `samp5x20.anon`, `ora5x20`, `ora5x20.anon` | `ora5xT` |

## Figure 5: Adversarial exemplars

| experiment | finetune_on | predict_on |
| --- | --- | --- |
| CASPER_orig + oracle train + add tag at test time, adversarial eval | `samp5x20`, `ora5x20.plat` | `adv5xT.plat` |

## Table 6 / Table 9: Schema refactoring

All models are trained on the synthesized pre-refactor data ("label-merge").

| experiment | finetune_on | predict_on |
| --- | --- | --- |
| T5, eval on pre-refactor data | `noaug.label-merge` | `noaug.label-merge` |
| T5, eval on post-refactor data | `noaug.label-merge` | `noaug` |
| CASPER, eval on pre-refactor data | `samp5.label-merge`, `samp5.label-merge.anon` | `top5.label-merge` |
| CASPER, eval on post-refactor data | `samp5.label-merge`, `samp5.label-merge.anon` | `top5` |
| CASPER_orig, eval on pre-refactor data | `samp5.label-merge` | `top5.label-merge` |
| CASPER_orig, eval on post-refactor data | `samp5.label-merge` | `top5` |
| CASPER + tag, eval on pre-refactor data | `samp5.label-merge`, `ora5.label-merge.plat` | `top5.label-merge` |
| CASPER + tag, eval on post-refactor data | `samp5.label-merge`, `ora5.label-merge.plat` | `top5.label-split.plat` |
| CASPER_orig + tag, eval on pre-refactor data | `samp5.label-merge`, `samp5.label-merge.anon`, `ora5.label-merge.plat`, `ora5.label-merge.plat.anon` | `top5.label-merge` |
| CASPER_orig + tag, eval on post-refactor data | `samp5.label-merge`, `ora5.label-merge.plat` | `top5.label-split.plat` |
