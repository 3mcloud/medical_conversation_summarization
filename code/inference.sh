# example: model inference and selection of best checkpoint after training
EXP=experiment_name
SUF=""
REF=${SUF}  # see README for usage
MLEN=512

python evaluation/inference.py  \
  --bin_path "../../experiments/${EXP}/bin${SUF}" \
  --data_folder "../experiments/${EXP}" \
  --dev_suffix ${SUF} \
  --test_suffix ${SUF} \
  --ref_suffix ${REF} \
  --max_len ${MLEN} \
  --batch_size 8 \
