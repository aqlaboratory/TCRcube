"""
Prepare fasta files for extraction of ESM2 representations of full pMHC-TCR complexes
Expects CSV file with train/test with following columns:
targetid; target_chainseq
"""

import pandas as pd
from sys import argv,stderr

if len(argv)<3:
    stderr("Usage: prepare_fasta_ESM2-full.py dataset.csv output.fasta")
    exit(1)

df=pd.read_csv(argv[1],sep=';')

with open (argv[2],'w') as f:
    for i,row in df.iterrows():
        seq=row['target_chainseq'].replace("/",50*'G') # polyGly linkers
        f.write(f">{row['targetid']}\n{seq}\n")