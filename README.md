# CFA: Compositional Feature Alignment

Code for the paper _Enhancing Compositinal Generalization via Compositional Feature Alignment_

CREDITS: Our code is heavily based on https://github.com/mlfoundations/wise-ft and https://github.com/locuslab/FLYP. We thank the authors for open sourcing their code.

## Setting up conda env
```bash
conda create -n cfa python=3.11 anaconda
conda activate cfa
pip install -r requirements.txt
```

## Datasets
We conducted our experiments on [DomainBed](https://github.com/facebookresearch/DomainBed) and [Wilds](https://wilds.stanford.edu/datasets/). Put the datasets under ``./cfa/data/``.


## Run CFA
Our CFA is a two stage finetuning method. To run CFA, fill out all the parameters in the lauching files.
### Stage-1
To run Stage-1 of CFA and Linear Probing, fill out the configs in ``linear_probe_exps.py``.
```bash
python linear_probes_exps.py  # Run Linear Probing Codes
```

### Stage-2
To run Stage-2 of CFA, Finetuning and LP-FT, fill out the configs in ``launch_exp.py``. 
```bash
python launch_exp.py  # Run Finetuning Codes
```

### WiSE
To perform model interpolation, fill out the configs in ``interpolate.py``.
```bash
python interpolate.py  # Run WiSE
```
