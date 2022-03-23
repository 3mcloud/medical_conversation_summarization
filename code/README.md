# Fine-tuning BART

The overall procedure of fine-tuning BART follows: 
1. preparing the data into lines of text format
1. bpe tokenization and binarization
1. model training
1. model selection and inference
1. evaluation


## Download Pretrained BART Model Checkpoint

```bash
bash ./download.sh
```
The script downloads pretrained `bart.large` model from fairseq; for other pretrained BART models, visit [this page](https://github.com/pytorch/fairseq/blob/main/examples/bart/README.md).


## Single Stage Fine-tuning

**Step 1.** Prepare your dataset into `.jsonl` format, example template provided in `../data/dummy.jsonl`. Note that two meta columns are provided in `dummy.jsonl`: (i) `cid` is the identifier for each conversation (ii) `sid` is the unique identifier for each reference summary. Current template support multiple reference summaries per conversation training, as long as both meta columns are provided.

**Step 2.** Generate `.source|.target|.meta` files, those will be saved in a folder in `../experiments`, with the name specified by `${EXP}` in the bash script. 
```bash 
bash ./preprocess_singlestage.sh
```

**Step 3.** Tokenization and binarization. This will create additional `.bpe.source|.bpe.target` files and a `bin/` subfolder in `../experiments/${EXP}`
```bash
bash ./tokenize.sh
```

**Step 4.** Train the model, all hyperparameters are set according to the paper, and can be tuned if needed
```bash
bash ./train.sh
```

**Step 5.** Select best model checkpoint by ROUGE score and generate `.hypo` files
```bash
bash ./inference.sh
```

**NOTE**
1. Make sure the env variables `EXP` and `SUF` are set consistently across the bash scripts. You can also create your own bash script to chain all scripts together as below
```bash
# don't forget to remove definition statement of those three env variables from all scripts
export DATAFILES=("<path_to_train_jsonl>" "<path_to_dev_jsonl>" "<path_to_test_jsonl>")
export EXP=exp_name
export SUF=''
bash ./preprocess_singlestage.sh
bash ./tokenize.sh
bash ./train.sh
bash ./inference.sh
```


## Muti-stage Fine-tuning - Chunking
Prepare the dataset in `.jsonl` format as specified in Step 1 in single stage fine-tuning.

Run the following scripts in order
```bash
bash ./preprocess_chunk_stage1.sh  # set ${DATAFILES} to point to your dataset

bash ./tokenize.sh  # set SUF="_stage1"

bash ./train.sh  # set SUF="_stage1"

bash ./inference.sh # set SUF="_stage1"

bash ./preprocess_stagex.sh # set CKPT=<best_model_checkpoint_from_previous_step> and set SUF="_stagex"

bash ./tokenize.sh  # set SUF="_stage2"

bash ./train.sh # set SUF="_stage2"

bash ./inference.sh # set SUF="_stage2"
```

**NOTE**

1. Make sure `EXP` and `SUF` are set correctly at each step. 
1. Best model checkpoint, after running `./inference.sh` for the first time, will be printed on screen as well as saved in a separate log file named `best_model_checkpoint${SUF}.log`; make sure to update it accordingly when running `./preprocess_stagex.sh`


## Evaluation

The procedures above end with model generated hypothesis summaries for all conversations, for detailed mean-of-mean, meant-of-best ROUGE score calculation and other types of evaluations, use `evaluation/evaluation_scripts.ipynb`.

Note that concept-based evaluation requires installation of `quickUMLS` library (follow the instructions [here](https://github.com/Georgetown-IR-Lab/QuickUMLS)).


## Additional Tutorials

### BPE tokenization
BPE encoder files (vocabulary, dictionary, etc.) are provided in `<this_repo>/code/encoding/`, the following bash script shows how to download them afresh:
```bash
cd encoding

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
```

### Mean-of-mean ROUGE calculation during inference:
The ROUGE calculation in `evaluation/inference.py` is a simple average across all sample points; if you want to use mean-of-mean ROUGE score for selecting best model checkpoint (note that this only applies to **single stage fine-tuning** and **stage 2 of multistage fine-tuning**), follow these steps:
1. During preprocessing, in addition to regular `.source|.target|.meta` files, generate `.target|.meta` files that contains **all** reference summaries; it is recommended to use different suffix for these files, e.g. `dev_all.target`
2. Run `evaluation/inference.py` as follows
```bash
EXP=<experiment_name>
SUF=<suffix_of_the_data_files>
REF='_all'
MLEN=512

python evaluation/inference.py  \
  --bin_path "../../experiments/${EXP}/bin${SUF}" \
  --data_folder "../experiments/${EXP}" \
  --dev_suffix ${SUF} \
  --test_suffix ${SUF} \
  --ref_suffix ${REF} \
  --max_len ${MLEN} \
  --batch_size 8 \
  --rouge_against_all #<-- this line added
```