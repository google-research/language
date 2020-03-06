MetaQA experiments with NQL for the paper Scalable Neural Methods
  for Reasoning With a Symbolic Knowledge Base (ICLR 2020)

To reproduce experiment results in Table 3:

0. cd [this directory]

1. Download MetaQA datasets from

    https://github.com/yuyuz/MetaQA

2. Preprocess data

    python preprocess_data.py

3. Run tensorflow experiments

    MetaQA-2hop:
      python metaqa.py --rootdir /home/haitiansun/metaqa --num_hops=2 \
      --train_file=qa_van2_train.exam --dev_file=qa_van2_dev.exam \
      --test_file=qa_van2_test.exam --mask_seeds=False

    MetaQA-3hop:
      python metaqa.py --rootdir /home/haitiansun/metaqa --num_hops=3 \
      --train_file=qa_van3_train.exam --dev_file=qa_van3_dev.exam \
      --test_file=qa_van3_test.exam --mask_seeds=False

    Note: you may change --mask_seeds=True for results with "ReifKB + mask"
