##############################################################################
# generate and save data used fror figure1
##############################################################################

# parse command line args

SOURCE_DIR=$1
DATA_STEM=$2
DATA_DIR=$3

# binary used to collect each line in the graphs
EVAL="python ${SOURCE_DIR}/scaling_eval.py"

# make sure environment is correct

if [ ! -e ${DATA_DIR} ]; then
  echo output directory $DATA_DIR does not exist
  exit 1
fi

##############################################################################
# generate data for left-hand-side plot: time vs #entities for 3 algorithms

# for left-hand-side plot, sweep through these params and store results in tsv's
N_SWEEP=50,100,200,400,600,1000

# default NQL strategy - reified mixing
${EVAL} --vary_n ${N_SWEEP} --table_tag ${DATA_STEM}_reified_kb_

# 'late mixing' baseline
${EVAL} --vary_n ${N_SWEEP} --variant mix  --table_tag ${DATA_STEM}_late_mix_

# 'naive' baseline, which is only implemented w/o minibatches
${EVAL} --vary_n ${N_SWEEP} --variant sum  --minibatch_size 1 --table_tag ${DATA_STEM}_naive_

##############################################################################
# generate data for middle plot: time vs #rels for three algorithms

R_SWEEP=0,10,20,50,100,200,500,1000

# default NQL strategy - reified mixing
${EVAL} --vary_extra_rels ${R_SWEEP} --table_tag ${DATA_STEM}_reified_kb_

# 'late mixing' baseline
${EVAL} --vary_extra_rels ${R_SWEEP} --variant mix  --table_tag ${DATA_STEM}_late_mix_

# 'naive' baseline, which is only implemented w/o minibatches
${EVAL} --vary_extra_rels ${R_SWEEP} --variant sum  --minibatch_size 1 --table_tag ${DATA_STEM}_naive_

