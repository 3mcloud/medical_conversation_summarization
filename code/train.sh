TOTAL_NUM_UPDATES=50000
WARMUP_UPDATES=200
LR=2.5e-5
ULR=1e-3
DLR=5e-7
BSZ=1
MAX_TOKENS=1024
UPDATE_FREQ=8
PATIENCE=3
BART_PATH=bart.large/model.pt  # can be set to other model checkpoints
EXP="experiment_name"
SUF=""
ARCH=bart_large
BINPATH=../experiments/${EXP}/bin${SUF}
LOG=${EXP}${SUF}_${ARCH}.log  # naming template for the log

echo "Experiment = " ${EXP}
echo "Bart checkpoint = " ${BART_PATH}
echo "Log file =" ${LOG}

CUDA_VISIBLE_DEVICES=0

rm checkpoints/checkpoint*.pt

python train/train.py ${BINPATH} \
    --bpe gpt2 \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer \
    --reset-dataloader \
    --reset-meters \
    --arch ${ARCH} \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.001 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES \
    --update-freq $UPDATE_FREQ \
    --patience $PATIENCE \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --required-batch-size-multiple 1 \
    --batch-size ${BSZ} \
    --memory-efficient-fp16 \
    --truncate-source \
    --keep-last-epochs 8  | tee ../experiments/${EXP}/${LOG}