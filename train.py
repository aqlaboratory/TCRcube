import torch
import sys
import os
import numpy as np
import pandas as pd
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from model.datasets import ESM2FullDataset, ESM2IsolDataset, AF2EvoformerDataset, AF2StrModDataset, AAidDataset, PadCollate
from model.models import TCRcube, TCRcube_AAid, TCRcube_AAidpos


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # inputs
    parser.add_argument("train_csv", help="CSV file with input sequences and binding information for the train set")
    parser.add_argument("test_csv", help="CSV file with input sequences and binding information for the test set")
    parser.add_argument("--restart",type=str,default=None,help="Restart checkpoint file to continue from")


    # paths to pre-computed repres
    parser.add_argument("--pos_dir_train", type=str, default=None, help="Directory with pre-computed full-lenght AF2/ESM2 repres for train positives")
    parser.add_argument("--neg_dir_train", type=str, default=None, help="Directory with pre-computed full-lenght AF2/ESM2 repres for train negatives")
    parser.add_argument("--pos_dir_test", type=str, default=None, help="Directory with pre-computed full-lenght AF2/ESM2 repres for test positives")
    parser.add_argument("--neg_dir_test", type=str, default=None, help="Directory with pre-computed full-lenght AF2/ESM2 repres for test negatives")

    parser.add_argument("--pos_prefix", type=str, default="pos", help="Prefix of the AF2 repres of the positive data points")
    parser.add_argument("--neg_prefix", type=str, default="neg", help="Prefix of the AF2 repres of the negative data points")
    parser.add_argument("--cdr3a_esm2isol_dir", type=str, default=None, help="Directory with pre-computed CDR3a ESM2 isolated repres")
    parser.add_argument("--cdr3b_esm2isol_dir", type=str, default=None, help="Directory with pre-computed CDR3b ESM2 isolated repres")
    parser.add_argument("--pep_esm2isol_dir", type=str, default=None, help="Directory with pre-computed peptide ESM2 isolated repres")
    parser.add_argument("--mhc_esm2isol_dir", type=str, default=None, help="Directory with pre-computed MHC ESM2 isolated repres")


    # general options
    parser.add_argument("-g", "--gpu", type=int, default=0, help="Number of gpu to use. Default = 0")
    parser.add_argument("--batchsize",type=int, default = 64, help="Batch size")
    parser.add_argument("-t", "--model_type", type=str, default="AAidpos", help="Type of the TCRcube model based on embeddings (AAidpos | AAid | AF2evo | AF2str | ESM2full | ESM2isol)")
    parser.add_argument("--inner_dim",type=int, default = 384, help="Inner dimension of the model")
    parser.add_argument("--lr",type=float, default = 1e-4, help="Learning rate")
    parser.add_argument("--epochs",type=int, default = 500, help="Number of training epochs")

    # Outputs
    parser.add_argument("--out_checkpoint", type=str, default=os.path.basename(__file__)+".pt", help="Checkpoint file where to save the pre-trained model") 
    parser.add_argument("--outfile",type=str, default = "predictions.csv", help="Output CSV file for predictions")
    options = parser.parse_args()

    # Try to calculate on GPU
    device = 'cuda:%d' % options.gpu if torch.cuda.is_available() else 'cpu'

    # Type-specific data sets and models
    match options.model_type:
        case "AAidpos":
            train_dataset = AAidDataset (options.train_csv)
            test_dataset = AAidDataset (options.test_csv)
            model = TCRcube_AAidpos(options.inner_dim)
        case "AAid":
            train_dataset = AAidDataset (options.train_csv)   
            test_dataset = AAidDataset (options.test_csv)
            model = TCRcube_AAid(options.inner_dim) 
        case "AF2evo":
            if (options.pos_dir_train) is None or (options.neg_dir_train) is None or (options.pos_dir_test) is None or (options.neg_dir_test) is None:
                sys.stderr.write ("You need to define directories with pre-computed AF2 repres (--pos_dir_train, --neg_dir_train, --pos_dir_test, --neg_dir_test)!\n")
                exit(1)
            train_dataset = AF2EvoformerDataset (options.train_csv,options.pos_dir_train,options.neg_dir_train,options.pos_prefix,options.neg_prefix)
            Adim = train_dataset[0][0].shape[1] #dimensions of AF2 embeddings
            test_dataset = AF2EvoformerDataset (options.test_csv,options.pos_dir_test,options.neg_dir_test,options.pos_prefix,options.neg_prefix)
            model = TCRcube(Adim, Adim, Adim, Adim, options.inner_dim)
        case "AF2str":
            if (options.pos_dir_train) is None or (options.neg_dir_train) is None or (options.pos_dir_test) is None or (options.neg_dir_test) is None:
                sys.stderr.write ("You need to define directories with pre-computed AF2 repres (--pos_dir_train, --neg_dir_train, --pos_dir_test, --neg_dir_test)!\n")
                exit(1)            
            train_dataset = AF2StrModDataset (options.train_csv,options.pos_dir_train,options.neg_dir_train,options.pos_prefix,options.neg_prefix) 
            Adim = train_dataset[0][0].shape[1] #dimensions of AF2 embeddings
            test_dataset = AF2StrModDataset (options.test_csv,options.pos_dir_test,options.neg_dir_test,options.pos_prefix,options.neg_prefix)
            model = TCRcube(Adim, Adim, Adim, Adim, options.inner_dim)            
        case "ESM2full":
            if (options.pos_dir_train) is None or (options.neg_dir_train) is None or (options.pos_dir_test) is None or (options.neg_dir_test) is None:
                sys.stderr.write ("You need to define directories with pre-computed ESM2 repres (--pos_dir_train, --neg_dir_train, --pos_dir_test, --neg_dir_test)!\n")
                exit(1)            
            train_dataset = ESM2FullDataset (options.train_csv,options.pos_dir_train,options.neg_dir_train)
            Edim = train_dataset[0][0].shape[1] #dimensions of ESM2 embeddings
            test_dataset = ESM2FullDataset (options.test_csv,options.pos_dir_test,options.neg_dir_test)
            model = TCRcube(Edim, Edim, Edim, Edim, options.inner_dim)
        case "ESM2isol":
            if (options.cdr3a_esm2isol_dir) is None or (options.cdr3b_esm2isol_dir) is None or (options.mhc_esm2isol_dir) is None or (options.pep_esm2isol_dir) is None:
                sys.stderr.write ("You need to define directories with pre-computed ESM2 repres (--cdr3a_esm2isol_dir, --cdr3b_esm2isol_dir,--pep_esm2isol_dir, --mhc_esm2isol_dir)!\n")
                exit(1)            
            train_dataset = ESM2IsolDataset (options.train_csv,options.cdr3a_esm2isol_dir,options.cdr3b_esm2isol_dir, options.pep_esm2isol_dir,options.mhc_esm2isol_dir) 
            Edim = train_dataset[0][0].shape[1] #dimensions of ESM2 embeddings
            test_dataset = ESM2IsolDataset (options.test_csv,options.cdr3a_esm2isol_dir,options.cdr3b_esm2isol_dir, options.pep_esm2isol_dir,options.mhc_esm2isol_dir) 
            model = TCRcube(Edim, Edim, Edim, Edim, options.inner_dim)
        case _:
            sys.stderr.write ("Wrong model type, please choose from: AAidpos | AAid | AF2evo | AF2str | ESM2full | ESM2isol\n")
            exit(1)
    
    model.train()
    model=model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=options.lr)

    # resume training from saved state 
    if options.restart is not None:
        checkpoint = torch.load(options.restart, map_location=torch.device(device)) # load the saved state dict

                # Restore state for model and optimizer
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start = checkpoint['epoch']
        losses = checkpoint['loss']
        test_losses = checkpoint['test_loss']
    else: # if there is no restart file, I start training from scratch
        start=0
        losses = []
        test_losses = []

    model.train()
    model=model.to(device)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=options.batchsize, collate_fn=PadCollate(d_data=4)) 
    test_dataloader = DataLoader(dataset=train_dataset, batch_size=options.batchsize, collate_fn=PadCollate(d_data=4)) 

    df_test = pd.read_csv(options.test_csv,sep=';')
    df_train = pd.read_csv(options.train_csv,sep=';')
    pos=df_train.loc[df_train['binds']==1]
    neg=df_train.loc[df_train['binds']==0]
    pw=torch.tensor([len(neg)/len(pos)]).to(device)

    # Defines a BCE with logits loss function
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)


    # Outer training loop
    print ("Epoch","Train loss","Test loss","AUROC","AUPR") # results table header
    for epoch in range(start,options.epochs):
        # Train inner loop: performs one train step and returns the corresponding loss
        mini_losses=[]
        for cdr3a,cdr3b,pep,mhc,y_batch,lens,cdr3a_mask,cdr3b_mask,pep_mask,mhc_mask in train_dataloader:  
            # Move the tensors from data loader to the device we are operating on    
            cdr3a = cdr3a.to(device)
            cdr3b = cdr3b.to(device)
            pep = pep.to(device)
            mhc = mhc.to(device)

            y_batch = y_batch.to(device) 
            lens = lens.to(device)   
            
            cdr3a_mask=cdr3a_mask.to(device)
            cdr3b_mask=cdr3b_mask.to(device)
            pep_mask=pep_mask.to(device)
            mhc_mask=mhc_mask.to(device)

        
            # Sets model to TRAIN mode
            model.train()
            # Makes predictions
            yhat = model(cdr3a,cdr3b,pep,mhc,cdr3a_mask,cdr3b_mask,pep_mask,mhc_mask,lens)
            # Computes loss
            loss = loss_fn(yhat, y_batch)
            # Computes gradients
            loss.backward()
            # Updates parameters and zeroes gradients
            optimizer.step()
            optimizer.zero_grad()
          
            mini_losses.append(loss.item())

        losses.append(np.mean(mini_losses))  

       # Test (validate) every epoch 
        mini_losses=[]
        ys=[]
        preds=[]
        energies=[]
        with torch.no_grad():
            for cdr3a,cdr3b,pep,mhc,y_val,lens,cdr3a_mask,cdr3b_mask,pep_mask,mhc_mask in test_dataloader:
                # Move the tensors to the working device
                cdr3a = cdr3a.to(device)
                cdr3b = cdr3b.to(device)
                pep = pep.to(device)
                mhc = mhc.to(device)

                y_val = y_val.to(device)
                lens = lens.to(device)

                cdr3a_mask=cdr3a_mask.to(device)
                cdr3b_mask=cdr3b_mask.to(device)
                pep_mask=pep_mask.to(device)
                mhc_mask=mhc_mask.to(device)
                   
                model.eval() # Swith model to evaluation mode
       
                yhat = model(cdr3a,cdr3b,pep,mhc,cdr3a_mask,cdr3b_mask,pep_mask,mhc_mask,lens) # get the prediction

                loss = loss_fn(yhat,y_val) # get loss for this batch
                mini_losses.append(loss.item())
                
                # Collect the predictions for evaluation
                ys.extend(torch.reshape(y_val, (-1,)).to('cpu').numpy())
                pred = torch.sigmoid(yhat)
                pred = pred.to('cpu')
                preds.extend(torch.reshape(pred, (-1,)).numpy())
                energies.extend(torch.reshape(yhat, (-1,)).to('cpu').numpy())
        auroc = roc_auc_score(ys, preds) # compute AUROC for this epoch 
        test_losses.append(np.mean(mini_losses))
        precision, recall, thresholds = precision_recall_curve(ys, preds) # PR curve
        # Use AUC function to calculate the area under the curve of precision recall curve
        aupr = auc(recall, precision) 
        # Validation results before first trainign
        print (epoch,losses[-1],test_losses[-1],auroc,aupr)  # print results for this epoch

        checkpoint = {'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': losses,
                     'test_loss': test_losses}        
        torch.save(checkpoint, options.out_checkpoint)

        # Save the current predictions to the output CSV file
        df_test['Probability']=preds 
        df_test['Energy']=energies
        df_test.to_csv(options.outfile,sep=';',index=False)
