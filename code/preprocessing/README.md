## Generate specific datasets
You can write your own script of generating data files for BART fine-tuning, the required format is detailed as below:
1. Name template must follow `train|dev|test${SUF}`, e.g. `train_stage1`
2. For train/dev/test, make sure to generate all three files: `.source, .target, .meta`
3. Each line in `.source` must be the complete text for one input
4. Each line in `.target` must be one complete reference summary
5. At least two identifiers: `cid` and `sid`, must be saved in `.meta`; the former maps to each conversation, and the latter maps to each reference summary