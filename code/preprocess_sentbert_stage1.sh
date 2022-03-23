# Generating data for stage 1 multistage-Sentbert training
TRAINING=("../data/snippets_train.jsonl" "../data/snippets_dev.jsonl" "../data/snippets_test.jsonl") # change this to match the path to snippet dataset
DATAFILES=("../data/train.jsonl" "../data/dev.jsonl" "../data/test.jsonl") # change this to match the path to summarization data files
EXP="experiment_name"  # change this to a name of your choice, .source|.target|.meta|.hypo|.log will be saved here

# generate files used in model training from snippet dataset
for DATAFILE in ${TRAINING[@]};
do
    echo $DATAFILE
    python preprocessing/generate_data.py \
        --mode "plain" \
        --file $DATAFILE \
        --exp $EXP
done

# generate files used in inference from summarization dataset
for DATAFILE in ${DATAFILES[@]};
do
    echo $DATAFILE
    python preprocessing/generate_data.py \
        --mode "multistage" \
        --file $DATAFILE \
        --exp $EXP
done
