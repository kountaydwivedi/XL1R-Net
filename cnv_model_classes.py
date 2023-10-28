import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils, transforms
from torch.utils.data import Dataset, DataLoader



class LUNG_CNV(Dataset):
    
    def __init__(self, labels, df, transform=None):
        # dataloading
        self.labels = labels
        self.df = df
        self.length = len(self.labels)
    
    def __len__(self):
        # len(dataset)
        return self.length
    
    def __getitem__(self, index):
        # dataset[idx]
        return (self.df[index], self.labels[index])
        
        
        
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

'''
Few tutorials:
Simple AutoEncoder in PyTorch:(https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py)
PyTorch Model on CIFAR: (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
'''

'''
using nn.Sequential is important as it helps to use encoder as feature extractor
https://towardsdatascience.com/batch-normalization-and-dropout-in-neural-networks-explained-with-pytorch-47d7a8459bcd
https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
use of BatchNorm1D is essential before ReLU to assure normalisation of batches

nn.LeakyReLU() not giving good convergence.
nn.Sigmoid() not giving good results
nn.GELU() performing quite similar as that to ReLU

Dropout is giving issues. Not letting the model converge. However, with empirical results, 
test accuracy have been noted as awesome with dropout. Hence, after studying, conclusion made that since 
Dropouts are used as regulariser terms, therefore, incorporating them would be a good idea.

'''


class CNV_AutoEncoder(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        
        self.encoder = nn.Sequential(
            
            nn.Linear(input_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.Tanh(),
            
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            
            nn.Linear(512, output_dim)
        )
        
        self.decoder = nn.Sequential(
            
            nn.Linear(output_dim, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
            
            nn.Linear(1024, 4096),
            nn.BatchNorm1d(4096),
            nn.Tanh(),
            
            nn.Linear(4096, input_dim)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #


class CNV_Classifier(nn.Module):


    def __init__(self, only_encoder, output_dim):
        super().__init__()
        
        self.output_dim = output_dim
        self.encoder = only_encoder

        self.classify = nn.Sequential(
            
            
            nn.Linear(self.output_dim, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 2),
            nn.Sigmoid()

        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.classify(x)

        return x

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #


# class AutoEncoder(nn.Module):
# optim: (500, 250, 125, 100)
# (5000, 1000, 500)(500, 1000, 5000)
# (5000, 1000, 500, 250, 100)(na, 0.4, 0.2, 0.1) loss = 0.6
# (1000, 500, 100)(tanh)(na, 0.4, 0.2)
# good enough: (4096, 2048, 1024, 512, 100)(tanh)(no dropout): loss = 0.28





# class Classifier()
# (200, 100, 2)(0.1, 0.1) acc 93.62
# (500, 250, 125, 2) acc 93.62















# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# -------------------------- above is the ------------------------------------- #
# --------------------------- proper execution with---------------------------- #
# ------------------------------- very less accuracy -------------------------- #
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #













# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import utils, transforms
# from torch.utils.data import Dataset, DataLoader



# class LUNG_CNV(Dataset):
    
#     def __init__(self, labels, df, transform=None):
#         # dataloading
#         self.labels = labels
#         self.df = df
#         self.length = len(self.labels)
    
#     def __len__(self):
#         # len(dataset)
#         return self.length
    
#     def __getitem__(self, index):
#         # dataset[idx]
#         return (self.df[index], self.labels[index])
        
        
        
# # ----------------------------------------------------------------------------- #
# # ----------------------------------------------------------------------------- #

# '''
# Few tutorials:
# Simple AutoEncoder in PyTorch:(https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py)
# PyTorch Model on CIFAR: (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
# '''

# '''
# using nn.Sequential is important as it helps to use encoder as feature extractor
# https://towardsdatascience.com/batch-normalization-and-dropout-in-neural-networks-explained-with-pytorch-47d7a8459bcd
# https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
# use of BatchNorm1D is essential before ReLU to assure normalisation of batches

# nn.LeakyReLU() not giving good convergence.
# nn.Sigmoid() not giving good results
# nn.GELU() performing quite similar as that to ReLU

# Dropout is giving issues. Not letting the model converge. However, with empirical results, 
# test accuracy have been noted as awesome with dropout. Hence, after studying, conclusion made that since 
# Dropouts are used as regulariser terms, therefore, incorporating them would be a good idea.

# '''


# class CNV_AutoEncoder(nn.Module):
    
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
        
        
#         self.encoder = nn.Sequential(
            
#             nn.Linear(input_dim, 4096),
#             nn.BatchNorm1d(4096),
#             nn.Tanh(),
            
#             nn.Linear(4096, 2048),
#             nn.BatchNorm1d(2048),
#             nn.Tanh(),
            
#             nn.Linear(2048, output_dim),
#         )
        
#         self.decoder = nn.Sequential(
            
#             nn.Linear(output_dim, 2048),
#             nn.BatchNorm1d(2048),
#             nn.Tanh(),
            
#             nn.Linear(2048, 4096),
#             nn.BatchNorm1d(4096),
#             nn.Tanh(),
            
#             nn.Linear(4096, input_dim),
#         )
        
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

    
# # ----------------------------------------------------------------------------- #
# # ----------------------------------------------------------------------------- #


# class CNV_Classifier(nn.Module):


#     def __init__(self, only_encoder, output_dim):
#         super().__init__()
        
#         self.output_dim = output_dim
#         self.encoder = only_encoder

#         self.classify = nn.Sequential(
            
            
#             nn.Linear(self.output_dim, 1024),
#             nn.BatchNorm1d(1024),
#             nn.Tanh(),
            
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.Tanh(),
            
#             nn.Linear(1024, 2),
#             nn.Sigmoid()

#         )
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.classify(x)

#         return x

# # ----------------------------------------------------------------------------- #
# # ----------------------------------------------------------------------------- #

