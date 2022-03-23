# Leveraging BART on medical conversation summarization (3M MModal version)
This repo implements algorithm in the paper: [Leveraging Pretrained Models for Automatic Summarization of Doctor-Patient Conversations](https://aclanthology.org/2021.findings-emnlp.313/)


## Environment Setup
It's recommended to create your own virtual environment before setup.
```bash
pip install -r code/requirements.txt
```


## Repo Contents
```bash
code/                 # source codes, see README in this folder for instructions on running the experiments
data/                 # should host original data files (usually .jsonl format), "dummy.jsonl" included as an example
experiments/          # each experiment creates a subfolder here. "dummy/" folder shows example files that can be present
```

## Running Experiments
See **README** in `code/` folder for details


## Citation
Please cite the paper if using this repo:
```bibtex
@inproceedings{zhang-etal-2021-leveraging-pretrained,
    title = "Leveraging Pretrained Models for Automatic Summarization of Doctor-Patient Conversations",
    author = "Zhang, Longxiang  and
      Negrinho, Renato  and
      Ghosh, Arindam  and
      Jagannathan, Vasudevan  and
      Hassanzadeh, Hamid Reza  and
      Schaaf, Thomas  and
      Gormley, Matthew R.",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.313",
    doi = "10.18653/v1/2021.findings-emnlp.313",
    pages = "3693--3712",
}
```

