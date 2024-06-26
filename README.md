# TCRcube: ML model for pMHC-TCR predictions

## Setup

Set up TCRcube by running
```bash
git clone https://github.com/aqlaboratory/TCRcube.git
cd TCRcube
pip install -e .
```
This would clone the repository, install dependencies and set up TCRcube as a packge.

## Pre-computing ESM2 / AF2 representations

* ESM2 representations
    * Need to be pre-computed using [extract.py](https://github.com/facebookresearch/esm/blob/main/scripts/extract.py) script from ESM2 repository
    * Fasta file for [extract.py](https://github.com/facebookresearch/esm/blob/main/scripts/extract.py) can be prepared from CSV file with train/test data set using [prepare_fasta_ESM2-full.py](precompute_repres/prepare_fasta_ESM2-full.py) and [prepare_fasta_ESM2-isol.py](precompute_repres/prepare_fasta_ESM2-isol.py) scripts
    * `esm2_t36_3B_UR50D` version of ESM2 was used in the paper
* AF2 representations
    * Extracted from AF2 structure predictions using [TCRdock](https://github.com/phbradley/TCRdock) pipeline
    * Modify TCRdock scripts to output numpy arrays with repres 
        * Use [TCRdock_predict_utils.patch](precompute_repres/TCRdock_predict_utils.patch) and [TCRdock_alphafold_model.py.patch](precompute_repres/TCRdock_alphafold_model.py.patch) to for the modifications
        * In [TCRdock](https://github.com/phbradley/TCRdock) repo: `patch -p1 predict_utils.py < TCRdock_predict_utils.patch && patch -p1 alphafold/model/model.py < TCRdock_alphafold_model.py.patch`
    * Follow the [README](https://github.com/phbradley/TCRdock/blob/main/README.md) in TCRdock repo to do AF2 predictions (and get the evoformer and structure module repres)

## Inference on test set using pre-trained models

* 

## Training

