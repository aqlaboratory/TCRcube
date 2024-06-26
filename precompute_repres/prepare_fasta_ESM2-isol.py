"""
Prepare fasta files for extraction of ESM2 representations of isolated pMHC-TCR fragments
Expects CSV file with train/test with following columns:
cdr3a; cdr3b; peptide; mhc_alpha; target_chainseq
"""
import pandas as pd
from sys import argv,stderr

if len(argv)<6:
    stderr("Usage: prepare_fasta_ESM2-isol.py dataset.csv cdr3a.fasta cdr3b.fasta pep.fasta mhc.fasta")
    exit(1)

df=pd.read_csv(argv[1],sep=';')

cdr3a=df.drop_duplicates(subset='cdr3a')['cdr3a'].tolist()
cdr3b=df.drop_duplicates(subset='cdr3b')['cdr3b'].tolist()
eps=df.drop_duplicates(subset='peptide')['peptide'].tolist()
mhccl=df.drop_duplicates(subset='mhc_alpha')
mhc=[]
mhcn=[]
for i,m in mhccl.iterrows():
    seq=m['target_chainseq'].split("/")[0]
    mhc.append(seq)
    print (m['mhc_alpha'])
    mhcn.append(m['mhc_alpha'].replace("*","").replace(":",""))

with open (argv[2],"w") as f:
    for x in cdr3a:
        f.write(">%s\n%s\n" % (x,x))

with open (argv[3],"w") as f:
    for x in cdr3b:
        f.write(">%s\n%s\n" % (x,x))

with open (argv[4],"w") as f:
    for x in eps:
        f.write(">%s\n%s\n" % (x,x))

with open (argv[5],"w") as f:
    for x,n in zip(mhc,mhcn):
        f.write(">%s\n%s\n" % (n,x))
        