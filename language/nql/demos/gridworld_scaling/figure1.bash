#!/bin/bash

# where experimental data will be stored
DATA_DIR=${HOME}/new-results
DATA_STEM=${DATA_DIR}/nql
SOURCE_DIR=`pwd`

# generate data

bash ${SOURCE_DIR}/gendata_figure1.bash ${SOURCE_DIR} ${DATA_STEM} ${DATA_DIR}

# generate plots from data

python ${SOURCE_DIR}/plot_figure1.py ${DATA_STEM} ${DATA_DIR}

