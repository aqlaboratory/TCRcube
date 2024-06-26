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

* Use [predict.py](predict.py) script for inference, see `python predict.py -h` for options
* You will need test set CSV files and for the respective models also pre-computed ESM2 / AF2 representations.
* See e.g. [test_panpeptide.csv](data/test_panpeptide.csv) for example input CSV file with the test set data
* Outputs AUROC and AUPR and CSV file with predictions for individual data points
* Example for amino acid identity model: 
```bash
python predict.py -g 0 -t AAidpos data/test_panpeptide.csv checkpoints/panpeptide_AAidpos.pt --outfile AAidpos_panpeptide.csv
```
* Example for AF2-evo model: 
```bash
python predict.py -g 0 -t AF2evo data/test_panpeptide.csv checkpoints/panpeptide_AF2-evo.pt 
        --pos_dir AF2repres/panpeptide_pos 
        --neg_dir AF2repres/panpeptide_neg 
        --outfile AF2-evo_panpeptide.csv
```
* Example for ESM2-isol model: 
```bash
python predict.py -g 0 -t ESM2isol data/test_panpeptide.csv checkpoints/panpeptide_ESM2-isol.pt 
        --cdr3a_esm2isol_dir ESM2repres/cdr3a/ 
        --cdr3b_esm2isol_dir ESM2repres/cdr3b 
        --pep_esm2isol_dir ESM2repres/pep 
        --mhc_esm2isol_dir ESM2repres/mhc
```

## Training

* Use [train.py](predict.py) script for inference, see `python train.py -h` for options
* You will need train and test set CSV files and for the respective models also pre-computed ESM2 / AF2 representations.
* Example for training ESM2-full version of the model: 
```bash
python -u train.py -g 0 data/train_panpeptide.csv data/test_panpeptide.csv -t ESM2full 
        --pos_dir_train ESM2repres/full_train_pos 
        --neg_dir_train ESM2repres/full_train_neg/ 
        --pos_dir_test ESM2repres/full_test_pos/ 
        --neg_dir_test ESM2repres/full_train_neg/ 
        --out_checkpoint panpeptide_ESM2-full.pt  > train_panpeptide_ESM2-full.out
```

