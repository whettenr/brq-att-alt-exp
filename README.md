# brq-att-alt-exp
repository for experimenting with alternative forms of attention and best-rq


# instructions

- create env with speechbrain (see below)
- set up yaml file 
    - for examples of pretraining see haparams folder
    - for examples of finetuning see the finetune
- there are two files that are needed for running the model in speechbrain 
    - mask.py (used in pretraining *see train.py*)
    - quantiser.py (used in yaml file)


```bash
conda create --name NAME_OF_ENV python=3.11
conda activate NAME_OF_ENV

pip install -r requirements.txt
pip install -e .
```
