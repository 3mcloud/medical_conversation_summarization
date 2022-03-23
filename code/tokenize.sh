# BPE preprocessing
EXP=experiment_name
SUF=''
SETS=("train" "dev")

echo $EXP
for SPLIT in ${SETS[@]};
do
     echo $SPLIT
     for LANG in source target
     do
         python preprocessing/multiprocessing_bpe_encoder.py \
         --encoder-json "encoding/encoder.json" \
         --vocab-bpe "encoding/vocab.bpe" \
         --inputs "../experiments/${EXP}/${SPLIT}${SUF}.$LANG" \
         --outputs "../experiments/${EXP}/${SPLIT}${SUF}.bpe.$LANG" \
         --workers 60 \
         #--keep-empty;
     done
 done

 # Binarization
fairseq-preprocess \
    --source-lang "source" \
    --target-lang "target" \
    --trainpref "../experiments/${EXP}/train${SUF}.bpe" \
    --validpref "../experiments/${EXP}/dev${SUF}.bpe" \
    --destdir "../experiments/${EXP}/bin${SUF}/" \
    --workers 60 \
    --srcdict "encoding/dict.txt" \
    --tgtdict "encoding/dict.txt"

