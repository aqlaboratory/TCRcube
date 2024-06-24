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
    def __init__(self, csv_file, esm2_pos_dir, esm2_neg_dir):
        self.df=pd.read_csv(csv_file, sep=';') # pandas data frame with sequences and binding label (0/1)
        self.esm2_pos_dir=esm2_pos_dir # directory with the representations of pMHC-TCR complexes
        self.esm2_neg_dir=esm2_neg_dir # directory with the representations of pMHC-TCR complexes

    def __getitem__(self, index):
        data = self.df.iloc[index] # select row in pandas data frame
        # load pre-computed ESM2 embeddings
        # naming convention: targetid.pt
        if data['binds']==1: # positive data points have different prefix then negative
            repres=torch.load(os.path.join(self.esm2_pos_dir, '%s.pt' % (data['targetid'])))['representations'][36] 
        else:
            repres=torch.load(os.path.join(self.esm2_neg_dir, '%s.pt' % (data['targetid'])))['representations'][36] 
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
        CDR3a = torch.load(os.path.join(self.cdr3adir, '%s.pt' % (data['cdr3a'])))['representations'][36]
        CDR3b = torch.load(os.path.join(self.cdr3bdir, '%s.pt' % (data['cdr3b'])))['representations'][36]
        peptide = torch.load(os.path.join(self.pepdir, '%s.pt' % (data['peptide'])))['representations'][36]
        mhc = torch.load(os.path.join(self.mhcdir, '%s.pt' % (data['mhc_alpha'].replace("*","").replace(":",""))))['representations'][36]
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
    def __init__(self, csv_file, af2_pos_dir, af2_neg_dir,pos_prefix,neg_prefix):
        self.df=pd.read_csv(csv_file, sep=';') # pandas data frame with sequences and binding status (0/1)
        self.pos_dir=af2_pos_dir
        self.neg_dir=af2_neg_dir
        self.pos_prefix=pos_prefix
        self.neg_prefix=neg_prefix

    def __getitem__(self, index):
        data = self.df.iloc[index] # select row in pandas data frame
        # load pre-computed AF2 embeddings
        if data['binds']==1: # positive data points have different prefix then negative
            fname=os.path.join(self.pos_dir,f"{self.pos_prefix}_{data['targetid']}_model_1_model_2_ptm_single.npy")
        else:
            fname=os.path.join(self.neg_dir,f"{self.neg_prefix}_{data['targetid']}_model_1_model_2_ptm_single.npy")
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
    def __init__(self, csv_file, af2_pos_dir, af2_neg_dir,pos_prefix,neg_prefix):
        self.df=pd.read_csv(csv_file, sep=';') # pandas data frame with sequences and binding status (0/1)
        self.pos_dir=af2_pos_dir
        self.neg_dir=af2_neg_dir
        self.pos_prefix=pos_prefix
        self.neg_prefix=neg_prefix

    def __getitem__(self, index):
        data = self.df.iloc[index] # select row in pandas data frame
        # load pre-computed AF2 embeddings
        if data['binds']==1: # positive data points have different prefix then negative
            fname=os.path.join(self.pos_dir,f"{self.pos_prefix}_{data['targetid']}_model_1_model_2_ptm_structure-single.npy")
        else:
            fname=os.path.join(self.neg_dir,f"{self.neg_prefix}_{data['targetid']}_model_1_model_2_ptm_structure-single.npy")
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

# Padding functions  
def pad_tensor(tns, pad, dim):
        """
        Pads tensor.
    
        Arguments:
           tns   tensor to pad
           pad   size to pad to
                 int
           dim   dimension of tensor to pad
                 int
    
        Returns:
           padded tensor
        """
    
        pad_size = list(tns.shape)
        pad_size[dim] = pad - tns.size(dim)
    
        return torch.cat([tns, torch.zeros(*pad_size)], dim=dim)

def get_mask(tns, pad, dim):
        """
        Pads tensor.
    
        Arguments:
           tns   tensor to pad
           pad   size to pad to
                 int
           dim   dimension of tensor to pad
                 int
    
        Returns:
           padded tensor
        """
    
        pad_size = list(tns.shape)[:-1]
        pad_size[dim] = pad - tns.size(dim)
        tns_size = list(tns.shape)[:-1] # remove embedding dimension
        tns_size[dim] = tns.size(dim)
    
        # mask = 1 when there is relevant info, 0 where there is only padding zero
        return torch.cat([torch.ones(*tns_size), torch.zeros(*pad_size)], dim=dim)
    
class PadCollate:
        """
        collate_fn that pads according to the longest sequence
        in a batch or specified pad length.
        """
    
        def __init__(self, dim=0, d_data=2, max_lens=None, labeled=True,lens=True):
            self.dim = dim
            self.d_data = d_data
            self.labeled = labeled
            self.lens=lens
            if max_lens is None: self.max_lens = [None for i in range(d_data)]
            else: self.max_lens = max_lens
    
        def pad_collate(self, batch):
            xs = []
            masks=[]
            for i in range(self.d_data):
                if self.max_lens[i] is None:
                    max_len = max(map(lambda x: x[i].shape[self.dim], batch))
                else: max_len = self.max_lens[i]
                xs.append(torch.stack([pad_tensor(x[i], pad=max_len, dim=self.dim) for x in batch], dim=0))
                masks.append(torch.stack([get_mask(x[i], pad=max_len, dim=self.dim) for x in batch], dim=0))
            if self.labeled: xs.append(torch.stack([x[self.d_data] for x in batch], dim=0))
            if self.lens: xs.append(torch.stack([x[self.d_data+1] for x in batch], dim=0))
    
            return xs+masks
    
        def __call__(self, batch):
            return self.pad_collate(batch)