from IPython.core.display import display, HTML

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, LeaveOneOut
from sklearn.metrics import roc_curve, auc, plot_confusion_matrix, plot_precision_recall_curve, classification_report

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from tqdm.notebook import tqdm

plt.style.use('seaborn-whitegrid')

import os
import gc
import time
import copy
import torch
import cnv_model_utils as u

torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

import warnings
warnings.simplefilter('ignore')






##===================================================================================##
#
# Get dataframe w.r.t shap genes. naming convention could be changed later.
#
##===================================================================================##

def get_shap_dataset(PATH, df):
    
    essential_genes=[]
    gene_list_path = PATH+"essential_genes_set/XENA_Analysis/SHAP_ANALYSIS/seed=322/shap_mlp_top_20.kd"
    with open(gene_list_path, "r") as file:
        for gene in file:
            gene=gene.strip()
            essential_genes.append(gene)
    df_final_promo_genes = df[df.columns.intersection(essential_genes)]
    return df_final_promo_genes, essential_genes


def get_XAI_genes_dataset(PATH, df):
    
    essential_genes=[]
    gene_list_path = PATH+"essential_genes_set/essential_genes.kd"
    with open(gene_list_path, "r") as file:
        for gene in file:
            gene=gene.strip()
            essential_genes.append(gene)
    df_final_promo_genes = df[df.columns.intersection(essential_genes)]
    return df_final_promo_genes, essential_genes


##===================================================================================##
#
# Compute K-Fold
#
##===================================================================================##


def get_sklearn_classifiers(seed):
    
    epochs = 100
    learning_rate = 1e-5
    batch_size_df = 16
    
    # original
#     sklearn_classifiers = [

#         MLPClassifier(hidden_layer_sizes=(512, 256, 128), random_state=seed, batch_size=batch_size_df, verbose=False, learning_rate_init=learning_rate, max_iter=epochs, activation='tanh'),  ## predict_proba
#         LogisticRegression(random_state=seed, max_iter=epochs, solver='saga', penalty='elasticnet', l1_ratio=0.4),  ## predict_proba
#         RandomForestClassifier(max_depth=100, random_state=seed), ## predict_proba
#         SVC(C=0.2, kernel='linear', random_state=seed), ## decision_func
#     ]
    sklearn_classifiers = [

        MLPClassifier(hidden_layer_sizes=(512, 256, 128), random_state=seed, batch_size=batch_size_df, verbose=False, learning_rate_init=learning_rate, max_iter=epochs, alpha=0.02),  ## predict_proba
        LogisticRegression(random_state=seed, max_iter=epochs, penalty='elasticnet', l1_ratio=0.04, solver='saga'),  ## predict_proba
        XGBClassifier(eta=0.1, max_depth=10, alpha=0.5, random_state=seed, verbosity=0),
        SVC(kernel='rbf', random_state=seed), ## decision_func
    ]
    return sklearn_classifiers
    

def k_fold(X_train, y_train, colors, seed, PATH):
    k = 5
#     k=len(X_train)
    kfold = StratifiedKFold(n_splits=k, shuffle=False)   
#     kfold = LeaveOneOut()
    
    train_acc_dict={
        'MLPClassifier':0.,
        'LogisticRegression':0.,
        'XGBClassifier':0.,
        'SVC':0.
    }
    test_acc_dict={
        'MLPClassifier':0.,
        'LogisticRegression':0.,
        'XGBClassifier':0.,
        'SVC':0.
    }
    roc_score_dict={
        'MLPClassifier':0.,
        'LogisticRegression':0.,
        'XGBClassifier':0.,
        'SVC':0.
    }

    
    classifier_name = ['MLPClassifier', 'LogisticRegression', 'XGBClassifier', 'Support Vector Classifier']
    for fold, (train_index, test_index) in enumerate(kfold.split(X_train, y_train)):
        
        
        sklearn_classifiers = get_sklearn_classifiers(seed)

        test_score = []
        train_score = []
        xtrain, xtest = X_train.iloc[X_train.index[train_index]], X_train.iloc[X_train.index[test_index]]
        ytrain, ytest = np.array(y_train)[train_index], np.array(y_train)[test_index]

        xtrain_scaled = xtrain.to_numpy()
        xtest_scaled = xtest.to_numpy()
        scaler = StandardScaler()
        xtrain_scaled = scaler.fit_transform(xtrain_scaled)
        xtest_scaled = scaler.transform(xtest_scaled)

        heading = HTML("<b>Fold: {}</b>".format(fold+1))
        display(heading)

        for classifier in tqdm(sklearn_classifiers):
            classifier.fit(xtrain_scaled, ytrain)

            train_acc_dict[f'{type(classifier).__name__}'] += classifier.score(xtrain_scaled, ytrain)
            test_acc_dict[f'{type(classifier).__name__}'] += classifier.score(xtest_scaled, ytest)
            
            print(classifier.score(xtrain_scaled, ytrain), classifier.score(xtest_scaled, ytest))

            # compute auc roc and plot
            if type(classifier).__name__ in ['SVC']:
                fpr, tpr, _ = roc_curve(ytest, classifier.decision_function(xtest_scaled))
            else:
                fpr, tpr, _ = roc_curve(ytest, classifier.predict_proba(xtest_scaled)[:,1])

            auc_score = auc(fpr, tpr)
            roc_score_dict[f'{type(classifier).__name__}'] += auc_score
        print("*********\n")

    temp_pd = pd.DataFrame({'Avg Train Accuracy':list(train_acc_dict.values()), 'Avg Test Accuracy':list(test_acc_dict.values()), 'Avg ROC Score':list(roc_score_dict.values())}, index=classifier_name)
    temp_pd = temp_pd/k
    display(temp_pd)
    ruler = HTML("<hr><hr>")
    display(ruler)
    plot_k_fold_summary(k, temp_pd, classifier_name, colors, PATH)
#     return temp_pd
    
    
##===================================================================================##
#
# Plot K-FOLD Summary
#
##===================================================================================##


def plot_k_fold_summary(k, df, classifier_name, colors, PATH):
    avg_roc = df['Avg ROC Score'].values
    avg_train_acc = df['Avg Train Accuracy'].values
    avg_val_acc = df['Avg Test Accuracy'].values

    fig, ax = plt.subplots(figsize=(10, 5), clear=True)

    N = 4
    ind = np.arange(N) 
    width = 0.25
    bar1 = ax.bar(ind, avg_roc, width, color = colors[0], edgecolor='black', alpha=0.8, linewidth=1, label='ROC')
    bar2 = ax.bar(ind+width, avg_train_acc, width, color=colors[1], alpha=0.8, edgecolor='black', linewidth=1, label='Avg. Train Acc.')
    bar3 = ax.bar(ind+width*2, avg_val_acc, width, color = colors[2], alpha=0.8, edgecolor='black', linewidth=1, label='Avg. Validate Acc.')

    ax.set_ylabel('Scores')
    ax.set_title('Avg ROC/Train/Validation Score by all classifiers on K-Fold CV (K={})'.format(k), fontsize='x-large')
    ax.legend(ncol=3, loc="upper right")


    ## shut down the xticks and labels 
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off


    cell_text=[]
    row_text=[]
    for item in avg_roc:
        row_text.append("{0:.3f}".format(item))
    cell_text.append(row_text)
    row_text=[]
    for item in avg_train_acc:
        row_text.append("{0:.3f}".format(item))
    cell_text.append(row_text)
    row_text=[]
    for item in avg_val_acc:
        row_text.append("{0:.3f}".format(item))
    cell_text.append(row_text)

    the_table = plt.table(cellText=cell_text,
                          rowLabels=['Avg. ROC Score', 'Avg. Train Score', ' Avg. Validate Score'],
                          colLabels=classifier_name,
                          colLoc='center',
                          rowLoc='center',
                          cellLoc='center',
                          rowColours=[colors[0], colors[1], colors[2]],
                          colColours=['gainsboro', 'gainsboro', 'gainsboro','gainsboro'],
                          loc='bottom')
    the_table.set_fontsize(12)
    plt.subplots_adjust(left=-0.3, bottom=-0.4)
#     plt.savefig(PATH+"/k_fold", dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
    
    
##===================================================================================##
#
# Plot Classification_Report/Confusion_Matrix/AUC_ROC
#
##===================================================================================##
    
def plot_report_cm_auc(X_train, y_train, X_test, y_test, classifier, color_yb, color_roc, reverse_color_roc, PATH):
    
    roc_color = color_roc
    visuals = color_yb
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 4), dpi=300, clear=True)

    # Specify the target classes
    classes = ["LUSC", "LUAD"]

    ## Classification Report
    report = ClassificationReport(classifier, ax=axes[0], classes=classes, cmap=visuals, support=True)
    report.fit(X_train, y_train)
    report.score(X_test, y_test)
    report.finalize()
    
    
    ## Confusion Matrix
    cm = ConfusionMatrix(classifier, ax=axes[1], classes=classes, cmap=visuals)
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    cm.finalize()
    
    
    ## ROC-AUC
    if type(classifier).__name__ in ['SVC']:
        fpr, tpr, _ = roc_curve(y_test, classifier.decision_function(X_test))
    else:
        fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
    auc_score = auc(fpr, tpr)
    axes[2].plot(fpr, tpr, marker='X', alpha=0.6, label="{}: {:.3f}".format(type(classifier).__name__, auc_score), color='{}'.format(roc_color))
    axes[2].title.set_text("{} ROC-AUC Curve".format(type(classifier).__name__))
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].legend(prop={'size': 16})

    
    
    ## train/eval accuracy
    ## return classifier.score at the end
    classifier.fit(X_train, y_train)
    
    
#     ## train/test bar
#     N = 1
#     ind = np.arange(N)
#     width = 0.001
#     bar1 = axes[3].bar(ind, classifier.score(X_train, y_train), width/3, color = reverse_color_roc, edgecolor='black', alpha=0.8, linewidth=1, label='Avg Train Acc.={:.3f}'.format(classifier.score(X_train, y_train)))
#     bar2 = axes[3].bar(ind+width*1.5, classifier.score(X_test, y_test), width/3, color = color_roc, edgecolor='black', alpha=0.8, linewidth=1, label='Avg. Test Acc.={:.3f}'.format(classifier.score(X_test, y_test)))
#     axes[3].set_ylabel('Accuracy')
#     axes[3].title.set_text('Avg Train/Test Accuracy of {}'.format( type(classifier).__name__ ))
#     axes[3].legend(loc="lower center")
#     axes[3].legend(prop={'size': 12})


#     ## shut down the xticks and labels 
#     axes[3].tick_params(
#         axis='x',          # changes apply to the x-axis
#         which='both',      # both major and minor ticks are affected
#         bottom=False,      # ticks along the bottom edge are off
#         top=False,         # ticks along the top edge are off
#         labelbottom=False) # labels along the bottom edge are off

    
    plt.subplots_adjust(left=0.1,
                        bottom=0.2, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.3, 
                        hspace=0.4)
    plt.savefig(PATH+"/{}".format(type(classifier).__name__), dpi=300, bbox_inches='tight')
    plt.show()
    return (classifier.score(X_test, y_test), auc_score)
    
    
    

    
    
    
    
##===================================================================================##
#
# Prepare ICGC Dataaset
#
##===================================================================================##
    
def prepare_icgc_dataset(icgc_luad, icgc_lusu, promo_genes):
    
    all_columns = list(icgc_luad.columns)
    all_columns.remove('submitted_sample_id')
    all_columns.remove('gene_id')
    all_columns.remove('normalized_read_count')
    all_columns.remove('raw_read_count')
    icgc_luad.drop(columns=all_columns, inplace=True)
    icgc_lusu.drop(columns=all_columns, inplace=True)
    display(HTML("Removed all unnecessary columns. Retained submitted_sample_id, gene_id, normalized_read_count, raw_read_count."))
    
    luad_grouped_by_tcga_id = icgc_luad.groupby('submitted_sample_id')
    lusu_grouped_by_tcga_id = icgc_lusu.groupby('submitted_sample_id')
    display(HTML("Grouped dataset by submitted_sample_id (i.e. the TCGA id)"))
    
    ls_luad = copy.deepcopy(promo_genes)
    ls_lusu = copy.deepcopy(promo_genes)
    ls_luad.append('label')
    ls_luad.append('pid')
    ls_lusu.append('label')
    ls_lusu.append('pid')
    display(HTML("Appended label and pid to icgc_luad and icgc_lusc"))
    
    icgc_luad_df=pd.DataFrame(columns=ls_luad)
    icgc_lusu_df=pd.DataFrame(columns=ls_lusu)
    display(HTML("Curating ICGC LUAD..."))
    for i, (name, group) in enumerate(luad_grouped_by_tcga_id):
        _=[] 
        for gene in promo_genes:
            group.reset_index(drop=True, inplace=True)
            gene_loc = group['gene_id'][group['gene_id']==gene].index[0]
            gene_val = group['normalized_read_count'].iloc[gene_loc]
            _.append(gene_val)
        _.append(1)
        _.append(name)
        icgc_luad_df.loc[i]=_
    display(HTML("ICGC_LUAD curated !!"))
    display(icgc_luad_df)
    display(HTML("Curating ICGC LUSC..."))
    for i, (name, group) in enumerate(lusu_grouped_by_tcga_id):
        _=[] 
        for gene in promo_genes:
            group.reset_index(drop=True, inplace=True)
            gene_loc = group['gene_id'][group['gene_id']==gene].index[0]
            gene_val = group['normalized_read_count'].iloc[gene_loc]
            _.append(gene_val)
        _.append(0)
        _.append(name)
        icgc_lusu_df.loc[i]=_
    display(HTML("ICGC_LUSC curated !!"))
    display(icgc_lusu_df)
        
    icgc_df = pd.concat([icgc_luad_df, icgc_lusu_df])
    display(HTML("ICGC_LUAD and ICGC_LUSC concatenated."))
    
    icgc_df.reset_index(drop=True, inplace=True)
    icgc_df.drop(labels='pid', axis=1, inplace=True)
    icgc_labels = list(icgc_df['label'])
    display(icgc_df)
    icgc_df.drop(labels='label', axis=1, inplace=True)
    return icgc_df, icgc_labels





##===================================================================================##
#
# Plot feature importance using Pearson Shapiro, Correlation and Spearman correlation
#
##===================================================================================##

def plot_feature_importance(X_train, y_train, PATH):
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 4), dpi=300, clear=True)
    X, y = X_train.astype(float), y_train
    
    visualizer_pearson = Rank2D(ax=axes[0], algorithm ='pearson', colormap='RdYlGn')
    visualizer_spearman = Rank2D(ax=axes[1], algorithm ='spearman', colormap='YlOrBr')
    visualizer_shapiro = Rank1D(ax=axes[2], orient = 'v')
    
    axes[0].set_xticklabels(axes[0].get_xticks(), fontsize=5)
    axes[1].set_xticklabels(axes[1].get_xticks(), fontsize=5)
    axes[2].set_xticklabels(axes[2].get_xticks(), fontsize=5)
    axes[0].set_yticklabels(axes[0].get_yticks(), fontsize=5)
    axes[1].set_yticklabels(axes[1].get_yticks(), fontsize=5)
    
    visualizer_pearson.fit(X, y)        
    visualizer_pearson.transform(X)
    visualizer_pearson.finalize()
    
    visualizer_spearman.fit(X, y)        
    visualizer_spearman.transform(X)
    visualizer_spearman.finalize()
    
    visualizer_shapiro.fit(X, y)        
    visualizer_shapiro.transform(X)
    visualizer_shapiro.finalize()
    
    plt.savefig(PATH+"/feature_importance", dpi=300, bbox_inches='tight')
    plt.show()
    





    
    
##===================================================================================##
#
# Plot accuracy summary
#
##===================================================================================##


def plot_accuracy_summary(roc_list, accuracy_list, colors, PATH):
    classifier_name = ['MLPClassifier', 'LogisticRegression', 'RandomForestClassifier', 'Support Vector Classifier']
    avg_roc = roc_list
    avg_val_acc = accuracy_list

    fig, ax = plt.subplots(figsize=(8, 5), clear=True)

    N = 4
    ind = np.arange(N) 
    width = 0.30
    bar1 = ax.bar(ind, avg_roc, width, color = colors[0], edgecolor='black', alpha=0.8, linewidth=1, label='ROC Score.')
    bar2 = ax.bar(ind+width, avg_val_acc, width, color=colors[1], alpha=0.8, edgecolor='black', linewidth=1, label='Evaluation Score.')

    ax.set_ylabel('Scores')
    ax.set_title('AU-ROC Curve Score/Evaluation Score by all classifiers', fontsize='x-large')
    ax.legend(ncol=3, loc="upper right")


    ## shut down the xticks and labels 
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off


    cell_text=[]
    row_text=[]
    for item in avg_roc:
        row_text.append("{0:.3f}".format(item))
    cell_text.append(row_text)
    row_text=[]
    for item in avg_val_acc:
        row_text.append("{0:.3f}".format(item))
    cell_text.append(row_text)
    row_text=[]

    the_table = plt.table(cellText=cell_text,
                          rowLabels=['AU-ROC Curve Score',' Evaluation Score'],
                          colLabels=classifier_name,
                          colLoc='center',
                          rowLoc='center',
                          cellLoc='center',
                          rowColours=[colors[0], colors[1]],
                          colColours=['gainsboro', 'gainsboro', 'gainsboro','gainsboro'],
                          loc='bottom')
    the_table.set_fontsize(12)
    plt.subplots_adjust(left=-0.3, bottom=-0.4)
    plt.savefig(PATH+"/accuracy_summary_plot", dpi=300, bbox_inches='tight')
    plt.show()




















