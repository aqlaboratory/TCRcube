"""
PyTorch datasets for different types of input representations
"""
import pandas as pd
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ESM2FullDataset(Dataset):
    """
    Dataset of ESM2 embeddings extracted for the whole pMHC-TCR complex
    """
    def __init__(self, csv_file, esm2_dir):
        self.df=pd.read_csv(csv_file, sep=';') # pandas data frame with sequences and binding label (0/1)
        self.esm2_dir=esm2_dir # directory with the representations of pMHC-TCR complexes

    def __getitem__(self, index):
        data = self.df.iloc[index] # select row in pandas data frame
        # load pre-computed ESM2 embeddings
        # naming convention: targetid.pt
        repres=torch.load(os.path.join(self.esm2_dir, '%s.pt' % (data['targetid'])))['representations'][33] 
        # now let's extract CDR3a, CDR3b, peptide and MHC
        fullseq=data['target_chainseq'].replace("/",50*"G") # add polyGly like during ESM2 inference
        cdr3bstart=fullseq.find(data['cdr3b'])
        cdr3bstop=cdr3bstart+len(data['cdr3b'])
        cdr3astart=fullseq.find(data['cdr3a'])
        cdr3astop=cdr3astart+len(data['cdr3a'])
        pepstart=fullseq.find(data['peptide'])
        pepstop=pepstart+len(data['peptide'])
        CDR3a=repres[cdr3astart:cdr3astop] # even though the sequence is padded, it's in the end and thus this is valid
        CDR3b=repres[cdr3bstart:cdr3bstop]
        peptide=repres[pepstart:pepstop]
        mhcseq=data['target_chainseq'].split("/")[0]
        mhc=repres[:len(mhcseq)] # full MHC portion is at the begining of the repres
        l=[len(data['cdr3a']),len(data['cdr3b']),len(data['peptide']),len(mhcseq)]
        binds = torch.tensor(data['binds'], dtype=torch.float32)
        lens = torch.tensor(l, dtype=torch.int)
        return (CDR3a,CDR3b,peptide,mhc,binds,lens) # returns five embedding tensors and binary y value (binds /does not bind = 1/0)
    def __len__(self):
        return len(self.df)
    
class ESM2IsolDataset(Dataset):
    """
    Dataset of ESM2 embeddings extracted for the peptide, MHC and TCR peptide fragments in isolation
    """
    def __init__(self, csv_file, cdr3adir, cdr3bdir, pepdir, mhcdir):
        self.df=pd.read_csv(csv_file, sep=';') # pandas data frame with sequences and binding label (0/1)
        self.cdr3adir=cdr3adir
        self.cdr3bdir=cdr3bdir
        self.pepdir=pepdir
        self.mhcdir=mhcdir

    def __getitem__(self, index):
        data = self.df.iloc[index] # select row in pandas data frame
        # load pre-computed ESM2 embeddings
        # naming convention: sequence.pt (MHC_allele.pt for MHC)
        CDR3a = torch.load(os.path.join(self.cdr3adir, '%s.pt' % (data['cdr3_sequence_alpha'])))['representations'][33]
        CDR3b = torch.load(os.path.join(self.cdr3bdir, '%s.pt' % (data['cdr3_sequence_beta'])))['representations'][33]
        peptide = torch.load(os.path.join(self.pepdir, '%s.pt' % (data['epitope'])))['representations'][33]
        mhc = torch.load(os.path.join(self.mhcdir, '%s.pt' % (data['mhc_alpha'].replace("*","").replace(":",""))))['representations'][33]
        mhcseq=data['target_chainseq'].split("/")[0]
        l=[len(data['cdr3a']),len(data['cdr3b']),len(data['peptide']),len(mhcseq)]
        binds = torch.tensor(data['binds'], dtype=torch.float32)#.reshape(1,) # 0/1
        lens = torch.tensor(l, dtype=torch.int)
        return (CDR3a,CDR3b,peptide,mhc,binds,lens) # returns five embedding tensors and binary y value (binds /does not bind = 1/0)
    def __len__(self):
        return len(self.df)
    
class AF2EvoformerDataset(Dataset):
    """
    Dataset of AF2 evoformer embeddings of the whole pMHC-TCR complex
    """
    def __init__(self, csv_file, af2_pos_dir, af2_neg_dir):
        self.df=pd.read_csv(csv_file, sep=';') # pandas data frame with sequences and binding status (0/1)
        self.pos_dir=af2_pos_dir
        self.neg_dir=af2_neg_dir

    def __getitem__(self, index):
        data = self.df.iloc[index] # select row in pandas data frame
        # load pre-computed AF2 embeddings
        if data['binds']==1: # positive data points have different prefix then negative
            fname=os.path.join(self.pos_dir,f"pos_{data['targetid']}_model_1_model_2_ptm_single.npy")
        else:
            fname=os.path.join(self.neg_dir,f"neg_{data['targetid']}_model_1_model_2_ptm_single.npy")
        a=np.load(fname)
        repres=torch.from_numpy(a) # representations of the full-length pMHC-TCR
        # now let's extract CDR3a, CDR3b, peptide and MHC
        fullseq=data['target_chainseq'].replace("/","")
        cdr3bstart=fullseq.find(data['cdr3b'])
        cdr3bstop=cdr3bstart+len(data['cdr3b'])
        cdr3astart=fullseq.find(data['cdr3a'])
        cdr3astop=cdr3astart+len(data['cdr3a'])
        pepstart=fullseq.find(data['peptide'])
        pepstop=pepstart+len(data['peptide'])
        CDR3a=repres[cdr3astart:cdr3astop] # even though the sequence is padded in AF2, it's in the end and thus this is valid
        CDR3b=repres[cdr3bstart:cdr3bstop]
        peptide=repres[pepstart:pepstop]
        mhcseq=data['target_chainseq'].split("/")[0]
        mhc=repres[:len(mhcseq)] # full MHC portion
        l=[len(data['cdr3a']),len(data['cdr3b']),len(data['peptide']),len(mhcseq)]
        binds = torch.tensor(data['binds'], dtype=torch.float32)
        lens = torch.tensor(l, dtype=torch.int)
        return (CDR3a,CDR3b,peptide,mhc,binds,lens) # returns five embedding tensors and binary y value (binds /does not bind = 1/0)
    def __len__(self):
        return len(self.df)

class AF2StrModDataset(Dataset):
    """
    Dataset of AF2 structure module embeddings of the whole pMHC-TCR complex
    """
    def __init__(self, csv_file, af2_pos_dir, af2_neg_dir):
        self.df=pd.read_csv(csv_file, sep=';') # pandas data frame with sequences and binding status (0/1)
        self.pos_dir=af2_pos_dir
        self.neg_dir=af2_neg_dir

    def __getitem__(self, index):
        data = self.df.iloc[index] # select row in pandas data frame
        # load pre-computed AF2 embeddings
        if data['binds']==1: # positive data points have different prefix then negative
            fname=os.path.join(self.pos_dir,f"pos_{data['targetid']}_model_1_model_2_ptm_structure-single.npy")
        else:
            fname=os.path.join(self.neg_dir,f"neg_{data['targetid']}_model_1_model_2_ptm_structure-single.npy")
        a=np.load(fname)
        repres=torch.from_numpy(a) # representations of the full-length pMHC-TCR
        # now let's extract CDR3a, CDR3b, peptide and MHC
        fullseq=data['target_chainseq'].replace("/","")
        cdr3bstart=fullseq.find(data['cdr3b'])
        cdr3bstop=cdr3bstart+len(data['cdr3b'])
        cdr3astart=fullseq.find(data['cdr3a'])
        cdr3astop=cdr3astart+len(data['cdr3a'])
        pepstart=fullseq.find(data['peptide'])
        pepstop=pepstart+len(data['peptide'])
        CDR3a=repres[cdr3astart:cdr3astop] # even though the sequence is padded in AF2, it's in the end and thus this is valid
        CDR3b=repres[cdr3bstart:cdr3bstop]
        peptide=repres[pepstart:pepstop]
        mhcseq=data['target_chainseq'].split("/")[0]
        mhc=repres[:len(mhcseq)] # full MHC portion
        l=[len(data['cdr3a']),len(data['cdr3b']),len(data['peptide']),len(mhcseq)]
        binds = torch.tensor(data['binds'], dtype=torch.float32)
        lens = torch.tensor(l, dtype=torch.int)
        return (CDR3a,CDR3b,peptide,mhc,binds,lens) # returns five embedding tensors and binary y value (binds /does not bind = 1/0)
    def __len__(self):
        return len(self.df)
    
class AAidDataset(Dataset):
    """
    Dataset of amino acid identities
    """
    def __init__(self, csv):
        self.df=pd.read_csv(csv,sep=';')
        self.aaorder=list('ARNDCQEGHILKMFPSTWYVX')

    def __getitem__(self, index):
        data = self.df.iloc[index]
        a=torch.zeros((len(data['cdr3a'])))
        for i,aa in enumerate(list(data['cdr3a'])):
            pos=self.aaorder.index(aa) # at which position is the given amino acid encoded ?
            a[i]=pos

        b=torch.zeros((len(data['cdr3b'])))
        for i,aa in enumerate(list(data['cdr3b'])):
            pos=self.aaorder.index(aa) # at which position is the given amino acid encoded ?
            b[i]=pos 

        p=torch.zeros((len(data['peptide'])))
        for i,aa in enumerate(list(data['peptide'])):
            pos=self.aaorder.index(aa) # at which position is the given amino acid encoded ?
            p[i]=pos

        mhcseq=data['target_chainseq'].split("/")[0]
        m=torch.zeros((len(mhcseq)))
        for i,aa in enumerate(list(mhcseq)):
            pos=self.aaorder.index(aa) # at which position is the given amino acid encoded ?
            m[i]=pos

        binds = torch.tensor(data['binds'], dtype=torch.float32)
        l=[len(data['cdr3a']),len(data['cdr3b']),len(data['peptide']),len(mhcseq)]
        lens = torch.tensor(l, dtype=torch.int)
        return (a.unsqueeze(-1),b.unsqueeze(-1),p.unsqueeze(-1),m.unsqueeze(-1),binds,lens) # returns two embedding tensors and binary y value (binds/does not bind = 1/0)
    def __len__(self):
        return len(self.df)