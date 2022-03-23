## Generate specific datasets
You can write your own script of generating data files for BART fine-tuning, the required format is detailed as below:
1. Name template must follow `train|dev|test${SUF}`, e.g. `train_stage1`
2. For train/dev/test, make sure to generate all three files: `.source, .target, .meta`
3. Each line in `.source` must be the complete text for one input
4. Each line in `.target` must be one complete reference summary
5. At least two identifiers: `physician` and `id`, must be saved in `.meta`; the former maps to each conversation, and the latter maps to each reference summary

### Generate fine-tuning data from ASR dataset
1. Download and unzip the ASR dataset by using the download script from [scribe_resources](https://github.mmm.com/OneNLU-scribing-research-ML/scribe_resources/tree/main/datasets) repo
```bash
python scribe_resources/datasets/download.py oak asr -d <path_to_this_repo>/data/

# unzip the file
```

2. Run sample codes in `generate_specific_data.ipynb`, make sure to change the path variables in the last cell to point to the location of the json file folder in the ASR dataset

3. Follow instructions in `code/` for fine-tuning experiments