import torch
import torch.nn as nn
import torch.nn.functional as F


class TCRcube (nn.Module):
    def __init__(self,edim_a,edim_b,edim_p,edim_m,inner_dim):
        super().__init__()

        # projections of embeddings into a common inner dimension
        self.proja= nn.Linear(edim_a, inner_dim) 
        self.projb= nn.Linear(edim_b, inner_dim)
        self.projp= nn.Linear(edim_p, inner_dim)
        self.projm= nn.Linear(edim_m, inner_dim)

        # CDR3a output projection
        self.anorm = nn.LayerNorm(inner_dim)
        self.aout= nn.Linear(inner_dim, 1)

        # CDR3b output projection
        self.bnorm = nn.LayerNorm(inner_dim)
        self.bout= nn.Linear(inner_dim, 1)

    def forward(self, CDR3a, CDR3b, peptide,mhc,cdr3a_mask,cdr3b_mask,pep_mask,mhc_mask,lens):
        # masks are the same for all, construct them first
        cdr3a_mask=cdr3a_mask.unsqueeze(-1) # another dimension needed to be able to multiply with the embeddings
        pep_mask=pep_mask.unsqueeze(-1)
        mhc_mask=mhc_mask.unsqueeze(-1)
        maska=torch.einsum('bie,bje,bke->bijke',cdr3a_mask,pep_mask,mhc_mask) # generate 3D mask out of three 1D masks in a similar fashion as cube   
        cdr3b_mask=cdr3b_mask.unsqueeze(-1) 
        maskb=torch.einsum('bie,bje,bke->bijke',cdr3b_mask,pep_mask,mhc_mask) # generate 3D mask out of three 1D masks in a similar fashion as cube  
        
        
        ##### CDR3alpha - peptide interaction ####
        # project input embeddings into a common inner dimension 
        CDR3a_smallerdim=self.proja(CDR3a) # (batch_size, CDR3_length, embedding_dim) -> (batch_size, CDR3_length, inner_dim)
        pep_smallerdim=self.projp(peptide) # (batch_size, pep_length, embedding_dim) -> (batch_size, pep_length, inner_dim)
        mhc_smallerdim=self.projm(mhc)     # (batch_size, MHC_length, embedding_dim) -> (batch_size, MHC_length, inner_dim)

        acube=torch.einsum('bie,bje,bke->bijke',CDR3a_smallerdim,pep_smallerdim,mhc_smallerdim) # outer product -> (batch_size,CDR3_length, pep_length, MHC_length, inner_dim)


        aenterms=self.aout(F.relu(self.anorm(acube)))*maska # (batch_size,CDR3_length, pep_length, MHC_length, inner_dim) -> (batch_size,CDR3_length, pep_length, MHC_length, 1)
        aenterms=torch.squeeze(aenterms) # remove last dimension

        aen=torch.sum(aenterms,(-1,-2,-3))/lens[:,0]/lens[:,2]/lens[:,3] # sum and normalize by sequence lengths

        ##### CDR3beta - peptide interaction  ####

        # reduce dimensionality of AF2 embeddings
        CDR3b_smallerdim=self.projb(CDR3b) # (batch_size, CDR3_length, embedding_dim) -> (batch_size, CDR3_length, inner_dim)

        bcube=torch.einsum('bie,bje,bke->bijke',CDR3b_smallerdim,pep_smallerdim,mhc_smallerdim) # outer product -> (batch_size,CDR3_length, pep_length, MHC_length, inner_dim)

        benterms=self.bout(F.relu(self.bnorm(bcube)))*maskb # (batch_size,CDR3_length, pep_length, 128) -> (batch_size,CDR3_length, pep_length, 1)
        benterms=torch.squeeze(benterms) # remove last dimension

        ben=torch.sum(benterms,(-1,-2,-3))/lens[:,1]/lens[:,2]/lens[:,3] # sum and normalize by sequence lengths

        return aen + ben # sum the energy contributions of TCR alpha and TCR beta brenches
    

class PositionalEncoderSimpleMask(nn.Module):
    def __init__(self, model_dim, max_seq_len=200, concat=False):
        super(PositionalEncoderSimpleMask, self).__init__()
        self.pos_emb = nn.Embedding(max_seq_len, model_dim)
        self.concat = concat
   
    def forward(self,x):
        b, n, b,  device = *x.shape, x.device
        emb = self.pos_emb(torch.arange(n, device = device))#.unsqueeze(1)
        mask = x == 0 #This assumes 0 is the padding token
        x += emb
        x[mask] = 0
        return x

class TCRcube_AAidpos (nn.Module):
    def __init__(self,inner_dim,edim=21):
        super().__init__()
        self.embeddings = nn.Embedding(edim, edim, padding_idx=0)
        self.posenc = PositionalEncoderSimpleMask(edim)

        # projections of embeddings into a common inner dimension
        self.proja= nn.Linear(edim, inner_dim) 
        self.projb= nn.Linear(edim, inner_dim)
        self.projp= nn.Linear(edim, inner_dim)
        self.projm= nn.Linear(edim, inner_dim)

        # CDR3a output projection
        self.anorm = nn.LayerNorm(inner_dim)
        self.aout= nn.Linear(inner_dim, 1)

        # CDR3b output projection
        self.bnorm = nn.LayerNorm(inner_dim)
        self.bout= nn.Linear(inner_dim, 1)

    def forward(self, CDR3a, CDR3b, peptide,mhc,cdr3a_mask,cdr3b_mask,pep_mask,mhc_mask,lens):
        # learnable amino acid identity embeddings with positional encoding
        CDR3a=self.posenc(self.embeddings(CDR3a.squeeze(-1).long()))
        CDR3b=self.posenc(self.embeddings(CDR3b.squeeze(-1).long()))
        peptide=self.posenc(self.embeddings(peptide.squeeze(-1).long()))
        mhc=self.posenc(self.embeddings(mhc.squeeze(-1).long()))
        # masks are the same for all, construct them first
        cdr3a_mask=cdr3a_mask.unsqueeze(-1) # another dimension needed to be able to multiply with the embeddings
        pep_mask=pep_mask.unsqueeze(-1)
        mhc_mask=mhc_mask.unsqueeze(-1)
        maska=torch.einsum('bie,bje,bke->bijke',cdr3a_mask,pep_mask,mhc_mask) # generate 3D mask out of three 1D masks in a similar fashion as cube   
        cdr3b_mask=cdr3b_mask.unsqueeze(-1) 
        maskb=torch.einsum('bie,bje,bke->bijke',cdr3b_mask,pep_mask,mhc_mask) # generate 3D mask out of three 1D masks in a similar fashion as cube  
        
        
        ##### CDR3alpha - peptide interaction ####
        # project input embeddings into a common inner dimension 
        CDR3a_smallerdim=self.proja(CDR3a) # (batch_size, CDR3_length, embedding_dim) -> (batch_size, CDR3_length, inner_dim)
        pep_smallerdim=self.projp(peptide) # (batch_size, pep_length, embedding_dim) -> (batch_size, pep_length, inner_dim)
        mhc_smallerdim=self.projm(mhc)     # (batch_size, MHC_length, embedding_dim) -> (batch_size, MHC_length, inner_dim)

        acube=torch.einsum('bie,bje,bke->bijke',CDR3a_smallerdim,pep_smallerdim,mhc_smallerdim) # outer product -> (batch_size,CDR3_length, pep_length, MHC_length, inner_dim)


        aenterms=self.aout(F.relu(self.anorm(acube)))*maska # (batch_size,CDR3_length, pep_length, MHC_length, inner_dim) -> (batch_size,CDR3_length, pep_length, MHC_length, 1)
        aenterms=torch.squeeze(aenterms) # remove last dimension

        aen=torch.sum(aenterms,(-1,-2,-3))/lens[:,0]/lens[:,2]/lens[:,3] # sum and normalize by sequence lengths

        ##### CDR3beta - peptide interaction  ####

        # reduce dimensionality of AF2 embeddings
        CDR3b_smallerdim=self.projb(CDR3b) # (batch_size, CDR3_length, embedding_dim) -> (batch_size, CDR3_length, inner_dim)

        bcube=torch.einsum('bie,bje,bke->bijke',CDR3b_smallerdim,pep_smallerdim,mhc_smallerdim) # outer product -> (batch_size,CDR3_length, pep_length, MHC_length, inner_dim)

        benterms=self.bout(F.relu(self.bnorm(bcube)))*maskb # (batch_size,CDR3_length, pep_length, 128) -> (batch_size,CDR3_length, pep_length, 1)
        benterms=torch.squeeze(benterms) # remove last dimension

        ben=torch.sum(benterms,(-1,-2,-3))/lens[:,1]/lens[:,2]/lens[:,3] # sum and normalize by sequence lengths

        return aen + ben # sum the energy contributions of TCR alpha and TCR beta brenches


class TCRcube_AAid (nn.Module):
    def __init__(self,inner_dim,edim=21):
        super().__init__()
        self.embeddings = nn.Embedding(edim, edim, padding_idx=0)

        # projections of embeddings into a common inner dimension
        self.proja= nn.Linear(edim, inner_dim) 
        self.projb= nn.Linear(edim, inner_dim)
        self.projp= nn.Linear(edim, inner_dim)
        self.projm= nn.Linear(edim, inner_dim)

        # CDR3a output projection
        self.anorm = nn.LayerNorm(inner_dim)
        self.aout= nn.Linear(inner_dim, 1)

        # CDR3b output projection
        self.bnorm = nn.LayerNorm(inner_dim)
        self.bout= nn.Linear(inner_dim, 1)

    def forward(self, CDR3a, CDR3b, peptide,mhc,cdr3a_mask,cdr3b_mask,pep_mask,mhc_mask,lens):
        # learnable amino acid identity embeddings with positional encoding
        CDR3a=self.embeddings(CDR3a.squeeze(-1).long())
        CDR3b=self.embeddings(CDR3b.squeeze(-1).long())
        peptide=self.embeddings(peptide.squeeze(-1).long())
        mhc=self.embeddings(mhc.squeeze(-1).long())
        # masks are the same for all, construct them first
        cdr3a_mask=cdr3a_mask.unsqueeze(-1) # another dimension needed to be able to multiply with the embeddings
        pep_mask=pep_mask.unsqueeze(-1)
        mhc_mask=mhc_mask.unsqueeze(-1)
        maska=torch.einsum('bie,bje,bke->bijke',cdr3a_mask,pep_mask,mhc_mask) # generate 3D mask out of three 1D masks in a similar fashion as cube   
        cdr3b_mask=cdr3b_mask.unsqueeze(-1) 
        maskb=torch.einsum('bie,bje,bke->bijke',cdr3b_mask,pep_mask,mhc_mask) # generate 3D mask out of three 1D masks in a similar fashion as cube  
        
        
        ##### CDR3alpha - peptide interaction ####
        # project input embeddings into a common inner dimension 
        CDR3a_smallerdim=self.proja(CDR3a) # (batch_size, CDR3_length, embedding_dim) -> (batch_size, CDR3_length, inner_dim)
        pep_smallerdim=self.projp(peptide) # (batch_size, pep_length, embedding_dim) -> (batch_size, pep_length, inner_dim)
        mhc_smallerdim=self.projm(mhc)     # (batch_size, MHC_length, embedding_dim) -> (batch_size, MHC_length, inner_dim)

        acube=torch.einsum('bie,bje,bke->bijke',CDR3a_smallerdim,pep_smallerdim,mhc_smallerdim) # outer product -> (batch_size,CDR3_length, pep_length, MHC_length, inner_dim)


        aenterms=self.aout(F.relu(self.anorm(acube)))*maska # (batch_size,CDR3_length, pep_length, MHC_length, inner_dim) -> (batch_size,CDR3_length, pep_length, MHC_length, 1)
        aenterms=torch.squeeze(aenterms) # remove last dimension

        aen=torch.sum(aenterms,(-1,-2,-3))/lens[:,0]/lens[:,2]/lens[:,3] # sum and normalize by sequence lengths

        ##### CDR3beta - peptide interaction  ####

        # reduce dimensionality of AF2 embeddings
        CDR3b_smallerdim=self.projb(CDR3b) # (batch_size, CDR3_length, embedding_dim) -> (batch_size, CDR3_length, inner_dim)

        bcube=torch.einsum('bie,bje,bke->bijke',CDR3b_smallerdim,pep_smallerdim,mhc_smallerdim) # outer product -> (batch_size,CDR3_length, pep_length, MHC_length, inner_dim)

        benterms=self.bout(F.relu(self.bnorm(bcube)))*maskb # (batch_size,CDR3_length, pep_length, 128) -> (batch_size,CDR3_length, pep_length, 1)
        benterms=torch.squeeze(benterms) # remove last dimension

        ben=torch.sum(benterms,(-1,-2,-3))/lens[:,1]/lens[:,2]/lens[:,3] # sum and normalize by sequence lengths

        return aen + ben # sum the energy contributions of TCR alpha and TCR beta brenches


