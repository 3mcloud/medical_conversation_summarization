# Generate data for stage 1 training in multistage Chunking method
DATAFILES=("../data/train.jsonl" "../data/dev.jsonl" "../data/test.jsonl") # change this to match the path to data files
EXP="experiment_name"  # change this to a name of your choice, .source|.target|.meta|.hypo|.log will be saved here
HLEN=128  # header length (# of words)
BLEN=384  # body length
BO=0.3333 # overlap percentage

for DATAFILE in ${DATAFILES[@]};
do
    echo ${DATAFILE}
    python preprocessing/generate_data.py \
        --mode "chunk" \
        --file ${DATAFILE} \
        --header_len ${HLEN} \
        --body_len ${BLEN} \
        --body_overlap ${BO} \
        --exp ${EXP}
done
