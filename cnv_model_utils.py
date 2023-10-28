from IPython.core.display import display, HTML
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd

import time
import torch
import torch.nn.functional as F

import os
import copy
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import seaborn as sea
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix



##===================================================================================##
#
# Set all seeds
#
##===================================================================================##

# kindly manually change seed here only !! seed=322 should be used by default
## seeds to be used from 314 -> 323
init_seed = 316

def get_seed():
    return init_seed

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




##===================================================================================##
#
# Dataset preprocessing
#
##===================================================================================##




def meth_data_preprocess(df_luad, df_lusu):
    ## drop that row from df_lusu
    df_lusu.drop(index=[13885], axis=0, inplace=True)
    
    df_lusu = df_lusu.T
    df_luad = df_luad.T
    
    new_header_luad = df_luad.iloc[0]
    new_header_lusu = df_lusu.iloc[0]
    df_luad = df_luad[1:]
    df_lusu = df_lusu[1:]
    df_luad.columns = new_header_luad
    df_lusu.columns = new_header_lusu
    
    label_luad = np.ones(len(df_luad), dtype=int) ## 1 : LUAD
    label_lusu = np.zeros(len(df_lusu), dtype=int) ## 0 : LUSU
    df_luad.insert(len(df_luad.columns), 'label', label_luad)
    df_lusu.insert(len(df_lusu.columns), 'label', label_lusu)

    ## fill nan with mean of the respective columns for luad and lusc
    start = time.time()
    for col in df_luad.columns[df_luad.isnull().any(axis=0)]:
        df_luad[col].fillna(df_luad[col].mean(),inplace=True)
    print('Time taken for LUAD fill NaN: ', time.time()-start)

    start = time.time()
    for col in df_lusu.columns[df_lusu.isnull().any(axis=0)]:
        df_lusu[col].fillna(df_lusu[col].mean(),inplace=True)
    print('Time taken for LUSC fill NaN: ', time.time()-start)
    
    df_final = df_luad.append(df_lusu)
    df_final.reset_index(drop=True, inplace=True)
    
    return df_final


def cnv_dataset_preprocess(df_luad, df_lusu):
    
    
    ## check for missing values, NaN, or Null 
    print("Missing values count in LUAD cohort = ", df_luad.isna().values.sum())
    print("Missing values count in LUSC cohort = ", df_lusu.isna().values.sum())
    
    ## make 'Gene Symbol' as the index
    df_luad.set_index('Gene Symbol', inplace=True)
    df_lusu.set_index('Gene Symbol', inplace=True)
    
    ## 1. save gene symbols as a list for future purpose
    ## 2. transpose the dataset
    ## 3. after transpose, remove index column (i.e., the instance names) from both the dataset
    columns = list(df_luad.index) ## we could have also used df_lusu, as both the datasets have same genes
    df_luad = df_luad.T
    df_lusu = df_lusu.T
    df_luad.reset_index(drop=True, inplace=True)
    df_lusu.reset_index(drop=True, inplace=True)
    
    ## 1. concat both datasets
    ## 2. insert 'label' column ('1' for LUAD and '0' for LUSU)
    ## 3. remove those columns (genes) that have same values across all the rows
    label_1 = np.ones(len(df_luad), dtype=int)
    label_0 = np.zeros(len(df_lusu), dtype=int)
    df_luad.insert(0, "label", label_1)
    df_lusu.insert(0, "label", label_0)
    df_final = pd.concat([df_luad, df_lusu])
    nunique = df_final.nunique()
    cols_to_drop = nunique[nunique == 1].index
    print('Dropping duplicate columns. Columns to drop = ', cols_to_drop)
    df_final = df_final.drop(cols_to_drop, axis=1)
    
    ## 1. save 'label' column in a list
    ## 2. remove this column from df_final
    ## 3. save df_final.columns in another list
    ## 4. finally, reset the index of the dataset and return df_final and the two saved lists
    labels = list(df_final['label'])
    df_final.drop(df_final.columns[0], axis=1, inplace=True)
    columns = list(df_final.columns)
    df_final.reset_index(drop=True, inplace=True)
    
    
    print('Done. Returning preprocessed dataset, gene names, and classes of the instances')
    ## return modified luad and lusu
    return df_final, columns, labels
    





##===================================================================================##
#
# Training encoder
#
##===================================================================================##

def train_encoder(
    num_epochs,
    model,
    optimizer,
    device, 
    train_loader,
    valid_loader=None,
    loss_fn=None,
    logging_interval=100,
    skip_epoch_stats=False,
    patience=5
):
    
    count_patience = 0
    init_loss = 1000
    mean_loss_enc = 0
    log_dict = {'train_loss_per_batch': [],'final_loss':0.}

    if loss_fn is None:
        loss_fn = F.mse_loss
        
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        mean_loss_batch = 0
        for batch_idx, (features, _) in enumerate(tqdm(train_loader)):
            features = features.to(device)
            features = features.float()
            
            # FORWARD AND BACK PROP
            logits = model(features)
            
            loss = loss_fn(logits, features)
            optimizer.zero_grad()
            loss.backward()
            
            # UPDATE MODEL PARAMETERS
            optimizer.step()
#             mean_loss_batch+=loss.item()
            
            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())
#             if not batch_idx % logging_interval:
#                 print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'% (epoch+1, num_epochs, batch_idx,len(train_loader), loss.item()))
            
            
#         mean_loss_enc += (mean_loss_batch/len(train_loader))
        print('Epoch: %03d/%03d | Loss: %.4f'% (epoch+1, num_epochs, loss.item()))
        if(init_loss > loss.item()):
            init_loss = loss.item()
            count_patience = 0
        else:
            count_patience = count_patience+1
            if count_patience >= patience:
                print("Early stopping (patience = {}). Lowest loss achieved: {}".format(patience, init_loss))
                log_dict['final_loss'] = init_loss
                break
        
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
#     print("Overall avg loss = %.2f" %(mean_loss_enc/num_epochs))
#     log_dict['avg_loss']=mean_loss_enc/num_epochs
    log_dict['final_loss']=loss.item()
    return log_dict



##===================================================================================##
#
# Compute pearson correlation 
#
##===================================================================================##

## https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html

def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)


def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))    


## https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/
def pearson_correlation(model, data_loader, device):
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (features, _) in enumerate(data_loader):
            features = features.to(device)
            features = features.float()
            logits = model(features)
            true_weights = features.detach().cpu().numpy()  ## assume it true_X
            autoencoder_weights = logits.detach().cpu().numpy()  ## assume it pred_X
            
            ## now start computing pearson correlation
            true_X = true_weights
            pred_X = autoencoder_weights

            pearson_corr = ncc(true_X, pred_X)
            return pearson_corr
            
       
            

##===================================================================================##
#
# Plot encoder loss
#
##===================================================================================##

def plot_training_loss(minibatch_losses, num_epochs, averaging_iterations=100, custom_label=''):

    iter_per_epoch = len(minibatch_losses) // num_epochs
    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_losses)),(minibatch_losses), label=f'Minibatch Loss{custom_label}')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000
    ax1.set_ylim([0, np.max(minibatch_losses[num_losses:])*1.5])
    ax1.plot(np.convolve(minibatch_losses,
                         np.ones(averaging_iterations,)/averaging_iterations,mode='valid'),label=f'Running Average{custom_label}')
    
    ax1.legend()
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))
    newpos = [e*iter_per_epoch for e in newlabel]
    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    
    plt.tight_layout()
    
    
    
    
    
##===================================================================================##
#
# Training Classifier
#
##===================================================================================##    

def train_classifier(
    num_epochs, 
    model,
    optimizer,
    device,
    train_loader,
    valid_loader,
    patience = 5
):
    
    log_dict = {'train_loss_per_batch':[], 'train_loss_per_epoch':[], 'train_acc':[], 'valid_acc':[]}

    loss_fn = F.cross_entropy
    
    count_patience = 0
    init_loss = 1000
    model_copy = []
    epoch_num = 0
    init_acc_train = 0.
    init_acc_valid = 0.
    start_time = time.time()
    for epoch in tqdm(range(num_epochs)):


        # ----- model training ------

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.float()
            features = features.to(device)
            targets = torch.tensor(targets, dtype=torch.long, device=device)
            
            # FORWARD AND BACK PROP
            logits = model(features)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            
            # UPDATE MODEL PARAMETERS
            optimizer.step()
            
            # LOGGING

            log_dict['train_loss_per_batch'].append(loss.item())
        
        
        # ----- model evaluation ------
        
        
        train_acc = compute_accuracy(model, train_loader, device)
        log_dict['train_acc'].append(train_acc.item())

        valid_acc = compute_accuracy(model, valid_loader, device)
        log_dict['valid_acc'].append(valid_acc.item())
        
        print('Epoch: %03d/%03d | Loss: %.4f | Acc_train: %.4f | Acc_valid: %.4f'%
              (epoch+1, num_epochs, loss.item(),train_acc.item(),valid_acc.item()))
        
        ## https://pythonguides.com/pytorch-early-stopping/
        
        if(init_loss > loss.item()):
            init_loss = loss.item()
            init_acc_valid = valid_acc.item()
            init_acc_train = train_acc.item()
            count_patience = 0
            epoch_num = epoch+1
            model_copy = []
            model_copy = copy.deepcopy(model)
        else:
            count_patience = count_patience+1
            if count_patience >= patience:
                print('==========================================================================================')
                print("Early stopping @ epoch: %03d. Min Loss: %.4f | Noted Train Acc: %.4f | Noted Valid Acc: %.4f"%
                      (epoch_num, init_loss, init_acc_train, init_acc_valid))
                print('==========================================================================================')
                model = model_copy
                break

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    return log_dict






##===================================================================================##
#
# Compute Accuracy
#
##===================================================================================##

def compute_accuracy(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):
            features = features.float()
            features = features.to(device)
            targets = torch.tensor(targets, dtype=torch.long, device=device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        
    return (correct_pred.float()/num_examples * 100)






# ##===================================================================================##
# #
# # Plot accuracy
# #
# ##===================================================================================##

# def plot_accuracy(train_acc, valid_acc):
    
#     num_epochs = len(train_acc)
#     # plt.figure(figsize=(10,10))
#     plt.plot(np.arange(1, num_epochs+1), train_acc, label='Training')
#     plt.plot(np.arange(1, num_epochs+1), valid_acc, label='Validation')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.show()

    

    
    
##===================================================================================##
#
# Read/Write Genes
#
##===================================================================================##

def read_essential_genes(PATH):
    essential_genes=[]
    gene_list_path = PATH+"essential_genes_set/essential_genes_custom.kd"
    with open(gene_list_path, "r") as file:
        for gene in file:
            gene=gene.strip()
            essential_genes.append(gene)
    return essential_genes


def write_seed_genes(PATH, seed, common_genes):
    gene_list_path = PATH+"project_summary_seed_wise/seed="+str(seed)+"/deg.kd"
    with open(gene_list_path, "w") as file:
        for gene in list(common_genes):
            file.write("%s\n" % gene)
    display(HTML("Common Genes written successfully !"))
        
        
def write_luad_lusu_seed_genes(PATH, seed, luad_seed_genes, lusu_seed_genes):
    gene_list_path_luad = PATH+"project_summary_seed_wise/seed="+str(seed)+"/deg_LUAD.kd"
    gene_list_path_lusu = PATH+"project_summary_seed_wise/seed="+str(seed)+"/deg_LUSU.kd"
    
    with open(gene_list_path_luad, "w") as file:
        for gene in list(luad_seed_genes):
            file.write("%s\n" % gene)
    display(HTML("LUAD Genes written successfully !"))
    with open(gene_list_path_lusu, "w") as file:
        for gene in list(lusu_seed_genes):
            file.write("%s\n" % gene)
    display(HTML("LUSU Genes written successfully !"))
    


        
        
        
##===================================================================================##
#
# Plot Train/Test/K-FOLD Accuracy
#
##===================================================================================##

def plot_train_test_k_fold_accuracy(
    val1, 
    val2,
    N,
    width,
    width_mult,
    fig_size,
    title,
    x_ticks,
    legends,
    file_path
):
    
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)
    
    bar1 = val1
    bar2 = val2

    ind = np.arange(N)  # the x locations for the groups
    width = width       # the width of the bars

    fig, ax = plt.subplots(figsize = fig_size)
    rects1 = ax.bar(ind, bar1, width, color='r', alpha=0.55)
    rects2 = ax.bar(ind + width*width_mult, bar2, width, color=color, alpha=0.55)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy',fontsize='large', fontweight='bold')
    ax.set_title(title,fontsize='xx-large', fontweight='bold')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(x_ticks, fontsize='large', fontweight='bold')

    ax.legend((rects1[0], rects2[0]), legends)


    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 0.5*height,
                    '%.4f' % float(height),
                    ha='center', va='bottom', fontsize='large', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()
    