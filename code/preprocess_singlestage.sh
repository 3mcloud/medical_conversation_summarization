# Generating data for single stage training
DATAFILES=("../data/train.jsonl" "../data/dev.jsonl" "../data/test.jsonl") # change this to match the path to data files
EXP="experiment_name"  # change this to a name of your choice, .source|.target|.meta|.hypo|.log will be saved here

for DATAFILE in ${DATAFILES[@]};
do
    echo $DATAFILE
    python preprocessing/generate_data.py \
        --mode "plain" \
        --file $DATAFILE \
        --exp $EXP
done
