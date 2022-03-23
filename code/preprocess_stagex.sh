EXP="experiment_name"
CKPT=checkpoint1.pt  # name of best model checkpoint as determined by inference.sh
SUF=_stagex
MLEN=512

# generate pseudo summaries or single sentence summaries using stage 1 model
python evaluation/inference.py  \
  --bin_path "../../experiments/${EXP}/bin" \
  --data_folder "../experiments/${EXP}" \
  --best_checkpoint ${CKPT} \
  --train_suffix ${SUF} \
  --dev_suffix ${SUF} \
  --test_suffix ${SUF} \
  --max_len ${MLEN} \
  --batch_size 8 \
  --no_rouge


echo "...INFERENCE DONE, AGGREGATING DATA FOR STAGE 2 NOW..."

# aggregate pseudo summaries (or single sentence summaries) generated from stage1 of multistage training
# see aggregate_between_stage() function in preprocessing/generate_data.py for detailed usage
python preprocessing/generate_data.py \
  --exp ${EXP} \
  --mode aggregate
