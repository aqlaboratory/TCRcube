import torch
import sys
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from model.datasets import ESM2FullDataset, ESM2IsolDataset, AF2EvoformerDataset, AF2StrModDataset, AAidDataset, PadCollate
from model.models import TCRcube, TCRcube_AAid, TCRcube_AAidpos


def get_inner_dim(statedict):
    return len(statedict['proja.weight'])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # inputs
    parser.add_argument("csv", help="CSV file with input sequences and binding information for the test set")
    parser.add_argument("checkpoint", type=str, default=None, help="Checkpoint file with pre-trained model") 

    # paths to pre-computed repres
    parser.add_argument("--pos_dir", type=str, default=None, help="Directory with pre-computed full-lenght AF2/ESM2 repres for test positives")
    parser.add_argument("--neg_dir", type=str, default=None, help="Directory with pre-computed full-lenght AF2/ESM2 repres for test negatives")
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
    parser.add_argument("--outfile",type=str, default = "predictions.csv", help="Output CSV file for predictions")
    options = parser.parse_args()

    # Try to calculate on GPU
    device = 'cuda:%d' % options.gpu if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    checkpoint = torch.load(options.checkpoint, map_location=torch.device(device))

    inner_dim=get_inner_dim(checkpoint['model_state_dict'])

    # Type-specific data sets and models
    match options.model_type:
        case "AAidpos":
            dataset = AAidDataset (options.csv)
            model = TCRcube_AAidpos(inner_dim)
        case "AAid":
            dataset = AAidDataset (options.csv)   
            model = TCRcube_AAid(inner_dim) 
        case "AF2evo":
            if (options.pos_dir) is None or (options.neg_dir) is None:
                sys.stderr.write ("You need to define directories with pre-computed AF2 repres (--pos_dir and --neg_dir)!\n")
                exit(1)
            dataset = AF2EvoformerDataset (options.csv,options.pos_dir,options.neg_dir,options.pos_prefix,options.neg_prefix)
            Adim = dataset[0][0].shape[1] #dimensions of AF2 embeddings
            model = TCRcube(Adim, Adim, Adim, Adim, inner_dim)
        case "AF2str":
            if (options.pos_dir) is None or (options.neg_dir) is None:
                sys.stderr.write ("You need to define directories with pre-computed AF2 repres (--pos_dir and --neg_dir)!\n")
                exit(1)            
            dataset = AF2StrModDataset (options.csv,options.pos_dir,options.neg_dir,options.pos_prefix,options.neg_prefix) 
            Adim = dataset[0][0].shape[1] #dimensions of AF2 embeddings
            model = TCRcube(Adim, Adim, Adim, Adim, inner_dim)            
        case "ESM2full":
            if (options.pos_dir) is None or (options.neg_dir) is None:
                sys.stderr.write ("You need to define directories with pre-computed ESM2 repres (--pos_dir and --neg_dir)!\n")
                exit(1)            
            dataset = ESM2FullDataset (options.csv,options.pos_dir,options.neg_dir)
            Edim = dataset[0][0].shape[1] #dimensions of ESM2 embeddings
            model = TCRcube(Edim, Edim, Edim, Edim, inner_dim)
        case "ESM2isol":
            if (options.cdr3a_esm2isol_dir) is None or (options.cdr3b_esm2isol_dir) is None or (options.mhc_esm2isol_dir) is None or (options.pep_esm2isol_dir) is None:
                sys.stderr.write ("You need to define directories with pre-computed ESM2 repres (--cdr3a_esm2isol_dir, --cdr3a_esm2isol_dir,--pep_esm2isol_dir, --mhc_esm2isol_dir)!\n")
                exit(1)            
            dataset = ESM2IsolDataset (options.csv,options.cdr3a_esm2isol_dir,options.cdr3b_esm2isol_dir, options.pep_esm2isol_dir,options.mhc_esm2isol_dir) 
            Edim = dataset[0][0].shape[1] #dimensions of ESM2 embeddings
            model = TCRcube(Edim, Edim, Edim, Edim, inner_dim)
        case _:
            sys.stderr.write ("Wrong model type, please choose from: AAidpos | AAid | AF2evo | AF2str | ESM2full | ESM2isol\n")
    

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model=model.to(device)

    dataloader = DataLoader(dataset=dataset, batch_size=options.batchsize, collate_fn=PadCollate(d_data=4)) 

    df_test = pd.read_csv(options.csv,sep=';')

    ys=[]
    preds=[]
    energies=[]
    with torch.no_grad():
        for cdr3a,cdr3b,pep,mhc,y_val,lens,cdr3a_mask,cdr3b_mask,pep_mask,mhc_mask in dataloader:
                # transfer tensors to the device
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
                
                # get the prediction
                model.eval()
                yhat = model(cdr3a, cdr3b, pep, mhc, cdr3a_mask, cdr3b_mask, pep_mask, mhc_mask, lens)
                
                # AUROC
                ys.extend(torch.reshape(y_val, (-1,)).to('cpu').numpy())
                pred = torch.sigmoid(yhat)
                pred = pred.to('cpu')
                preds.extend(torch.reshape(pred, (-1,)).numpy())
                energies.extend(torch.reshape(yhat, (-1,)).to('cpu').numpy())
        auroc = roc_auc_score(ys, preds) # compute AUROC for this epoch 
        precision, recall, thresholds = precision_recall_curve(ys, preds)
        # Use AUC function to calculate the area under the curve of precision recall curve
        aupr = auc(recall, precision) 

        print ("AUROC: %.2f\nAUPR: %.2f" % (auroc, aupr))

        df_test['Probability']=preds #Kd values should have same order as energies, as there is no shuffeling
        df_test['Energy']=energies
        df_test.to_csv(options.outfile,sep=';',index=False)
