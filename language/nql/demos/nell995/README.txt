Nell995 experiments with NQL for the paper Scalable Neural Methods
  for Reasoning With a Symbolic Knowledge Base (ICLR 2020)

To reproduce experiment results in Table 4:

0. cd [this directory]

1. Download MetaQA datasets from

    git clone https://github.com/shehzaadzd/MINERVA.git

2. Preprocess data

    python preprocess_data.py

3. Run tensorflow experiments

    python nell995.py --rootdir=nell995/ --task=concept_athletehomestadium \
    --num_hops=5 --epochs=50

    Note: you may change --task to run experiments on other queries.

    Available tasks:
        concept_agentbelongstoorganization
        concept_athletehomestadium
        concept_athleteplaysforteam
        concept_athleteplaysinleague
        concept_athleteplayssport
        concept_organizationheadquarteredincity
        concept_organizationhiredperson
        concept_personborninlocation
        concept_personleadsorganization
        concept_teamplaysinleague
        concept_teamplayssport
        concept_worksfor
