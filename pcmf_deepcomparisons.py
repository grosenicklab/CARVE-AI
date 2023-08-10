from __future__ import print_function, division
from pcmf_dataloaders import load_NCI, load_SRBCT, load_mouseorgans, load_gbmBreastLung, load_penguins

import pandas as pd
def load_data(X_in,true_clusters_in):
    from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
    scalerX = StandardScaler()
    scalerX.fit(X_in)
    x = scalerX.transform(X_in).astype(np.float32) #np.float32

    y = pd.factorize(true_clusters_in)[0].astype(np.int32)
    print(y.shape)

    print('samples', x.shape, y.shape)
    return x, y

## FOR IDEC; DEC?
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class FashionMNISTDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_data(X_in,true_clusters_in)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))


class SyntheticDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_data(X_in,true_clusters_in)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))


class MNISTDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_data(X_in,true_clusters_in)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))

class PenguinsDataset(Dataset):
    def __init__(self):
        from torch.utils.data import Dataset
        from pcmf_dataloaders import load_penguins
        self.x, self.y, _, _ = load_penguins()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))

import pandas as pd

class COVIDDataset(Dataset):
    def __init__(self):
        self.x, self.y = load_COVID()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))
    
import pandas as pd


class MouseOrgansDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_mouseorgans()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))


import pandas as pd


class GbmBreastLungDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_gbmBreastLung()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))

import pandas as pd

class NCIDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_NCI()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))

import pandas as pd

class SRBCTDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_SRBCT()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))


##################################################################
##################################################################
###### IDEC #######
##################################################################
##################################################################

import argparse
import numpy as np
# from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()

        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)

        self.z_layer = Linear(n_enc_3, n_z)

        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)

        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):

        # encoder
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        # decoder
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


class IDEC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 alpha=1,
                 pretrain_path=''):

        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, dataset, args, pretrain_epochs=200):
        print(self.pretrain_path)
        if self.pretrain_path == '':
            print('pretraining ae weights and saving to path',self.pretrain_path)
            pretrain_ae(self.ae,dataset,args,pretrain_epochs)
        # load pretrain weights
        else:
            try:
                self.ae.load_state_dict(torch.load(self.pretrain_path))
                print('load pretrained ae from', self.pretrain_path)
            except:
                print('pretraining ae weights and saving to path',self.pretrain_path)
                pretrain_ae(self.ae,dataset,args,pretrain_epochs)

    def forward(self, x):

        x_bar, z = self.ae(x)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain_ae(model,dataset,args,device='cpu',pretrain_epochs=200):
    '''
    pretrain autoencoder
    '''
    from torch.optim import Adam
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.nn import Linear
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(model)
    print(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    print('pretraining')
    for epoch in range(pretrain_epochs):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            # x = x.to(device) # uncomment for gpu/CUDA option...

            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("model saved to {}.".format(args.pretrain_path))


def train_idec(model,dataset,args,pretrain_epochs=200,train_epochs=100,device='cpu'):
#     model = IDEC(
#         n_enc_1=500,
#         n_enc_2=500,
#         n_enc_3=1000,
#         n_dec_1=1000,
#         n_dec_2=500,
#         n_dec_3=500,
#         n_input=args.n_input,
#         n_z=args.n_z,
#         n_clusters=args.n_clusters,
#         alpha=1.0,
#         pretrain_path=args.pretrain_path).to(device)

    #  model.pretrain('data/ae_mnist.pkl')
    print('training')
    from torch.optim import Adam
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.nn import Linear
    model.pretrain(dataset, args, pretrain_epochs=pretrain_epochs)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # cluster parameter initiate
    data = dataset.x
    y = dataset.y
    data = torch.Tensor(data)#.to(device) # uncomment for cuda
    x_bar, hidden = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    nmi_k = nmi_score(y_pred, y)
    print("nmi score={:.4f}".format(nmi_k))

    hidden = None
    x_bar = None

    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    model.train()
    for epoch in range(train_epochs):

        if epoch % args.update_interval == 0:

            _, tmp_q = model(data)

            # update target distribution p
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0] #np.float32
            y_pred_last = y_pred

            acc = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)
            print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),
                  ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

            if epoch > 0 and delta_label < args.tol:
                print('delta_label {:.4f}'.format(delta_label), '< tol',
                      args.tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        for batch_idx, (x, _, idx) in enumerate(train_loader):

            x = x.to(device)
            idx = idx.to(device)

            x_bar, q = model(x)

            reconstr_loss = F.mse_loss(x_bar, x)
            kl_loss = F.kl_div(q.log(), p[idx])
            loss = args.gamma * kl_loss + reconstr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

#######################################################
# Evaluate Critiron
#######################################################
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI_score, adjusted_rand_score as ARI_score, rand_score as rand_score

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
#     from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
#     return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    return sum([w[i, j] for i, j in zip(ind[0],ind[1])])*1.0/y_pred.size


def fit_IDEC_deep_cluster(dataset_in='FashionMNIST', data_dir='/athena/listonlab/store/amb2022/PCMF', batch_size_options=[15, 30], pretrain_epochs_options=[100, 1000], train_epochs_options=[100, 1000], device = 'cpu'):
    import argparse
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
    from sklearn.metrics import adjusted_rand_score as ari_score
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parameter import Parameter
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from torch.nn import Linear
    tic0 = time.time()
    from types import SimpleNamespace
    args = SimpleNamespace(
    )
    # batch_size_options = [15, 30]
    # pretrain_epochs_options = [100, 1000]
    # train_epochs_options = [100, 1000]
    print('dataset_in', dataset_in)
    accuracies = []
    idx = 0
    for batch_size in batch_size_options:
        for pretrain_epochs in pretrain_epochs_options:
            for train_epochs in train_epochs_options:
                args.n_z = 10
                args.lr = 0.001
                args.n_clusters = 6
                args.dataset = dataset_in #
                args.gamma = 0.1
                args.update_interval = 1
                args.tol = 0.001
                args.batch_size = batch_size

                os.chdir(data_dir)

                if args.dataset == 'mnist':
                    args.pretrain_path = 'data/ae_mnist.pkl'
                    dataset = MNISTDataset()
                    args.n_input = dataset.x.shape[1]
                    args.n_clusters = len(np.unique(dataset.y))

                if args.dataset == 'FashionMNIST':
                    args.pretrain_path = 'data/ae_FashionMNIST.pkl'
                    dataset = FashionMNISTDataset()
                    args.n_input = dataset.x.shape[1]
                    args.n_clusters = len(np.unique(dataset.y))

                if args.dataset == 'Synthetic':
                    args.pretrain_path = 'data/ae_Synthetic.pkl'
                    dataset = SyntheticDataset()
                    args.n_input = dataset.x.shape[1]
                    args.n_clusters = len(np.unique(dataset.y))

                if args.dataset == 'GBMbreastLung':
                    args.pretrain_path = 'data/ae_GBMbreastLung.pkl'
                    dataset = GBMbreastLungDataset()
                    args.n_input = dataset.x.shape[1]
                    args.n_clusters = len(np.unique(dataset.y))

                if args.dataset == 'penguins':
                    from pcmf_dataloaders import load_penguins
                    args.pretrain_path = 'data/ae_penguins.pkl'
                    dataset = PenguinsDataset()
                    args.n_input = dataset.x.shape[1]
                    args.n_clusters = len(np.unique(dataset.y))

                if args.dataset == 'mouseorgans':
                    args.pretrain_path = 'data/ae_mouseorgans.pkl'
                    dataset = MouseOrgansDataset()
                    args.n_input = dataset.x.shape[1]

                if args.dataset == 'gbmBreastLung':
                    args.pretrain_path = 'data/ae_gbmBreastLung.pkl'
                    dataset = GbmBreastLungDataset()
                    args.n_input = dataset.x.shape[1]
                    args.n_clusters = len(np.unique(dataset.y))

                if args.dataset == 'NCI':
                    args.pretrain_path = 'data/ae_NCI.pkl'
                    dataset = NCIDataset()
                    args.n_input = dataset.x.shape[1]
                    args.n_clusters = len(np.unique(dataset.y))

                if args.dataset == 'SRBCT':
                    args.pretrain_path = 'data/ae_SRBCT.pkl'
                    dataset = SRBCTDataset()
                    args.n_input = dataset.x.shape[1]
                    args.n_clusters = len(np.unique(dataset.y))

                if args.dataset == 'COVID':
                    args.pretrain_path = 'data/ae_COVID.pkl'
                    dataset = COVIDDataset()
                    args.n_input = dataset.x.shape[1]
                    args.n_clusters = len(np.unique(dataset.y))
                ### !!! ADD OTHER DATSETS HERE !!!

                print(args)

                tic = time.time()
                TRAIN_model = IDEC(
                        n_enc_1=500,
                        n_enc_2=500,
                        n_enc_3=1000,
                        n_dec_1=1000,
                        n_dec_2=500,
                        n_dec_3=500,
                        n_input=args.n_input,
                        n_z=args.n_z,
                        n_clusters=args.n_clusters,
                        alpha=1.0,
                        pretrain_path=args.pretrain_path)#.to(device)

                train_idec(TRAIN_model, dataset, args, pretrain_epochs=pretrain_epochs, train_epochs=train_epochs)

                toc = time.time() - tic
                print('Time elapsed:',toc)

                data = dataset.x
                y = dataset.y
                data = torch.Tensor(data) #.to(device)
                x_bar, hidden = TRAIN_model.ae(data)

                # evaluate clustering performance
                _, tmp_q = TRAIN_model(data)

                # update target distribution p
                tmp_q = tmp_q.data
                p = target_distribution(tmp_q)

                y_pred = tmp_q.cpu().numpy().argmax(1)

                acc = cluster_acc(y, y_pred)
                nmi = nmi_score(y, y_pred)
                ari = ari_score(y, y_pred)
                print('Acc {:.4f}'.format(acc),
                      ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
                
                # Calculate accuracy
                conf_mat_ord = confusion_matrix_ordered(y_pred, y)
                acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
                print('IDX:',idx, 'Accuracy:', acc, 'Batch size:',batch_size, 'pretrain_epochs:',pretrain_epochs, 'train_epochs:',train_epochs)
                idx = idx+1

                accuracies.append([idx, acc, batch_size, pretrain_epochs, train_epochs])

        toc2 = time.time() - tic0
        print('Total time elapsed:',toc)
    return accuracies, toc2, 'IDX: '+str(idx)+' Accuracy: '+str(acc)+' Batch size: '+str(batch_size)+' pretrain_epochs: '+str(pretrain_epochs)+' train_epochs: '+str(train_epochs)


##################################################################
##################################################################
#### END IDEC ####
##################################################################
##################################################################



##################################################################
##################################################################
#### BEGIN CARDEC ####
##################################################################
##################################################################

# def load_MNIST(labels_keep=[0,3,5], plot=False, skip=1, batch_size=50, randomize=False):
#     import numpy as np
#     from keras.datasets import mnist
#     from matplotlib import pyplot
#     import pandas as pd
    
#     print('Loading MNIST')
#     # Loading
#     (train_X, train_y), (test_X, test_y) = mnist.load_data()

#     # Shape of dataset
#     print('X_train: ' + str(train_X.shape))
#     print('Y_train: ' + str(train_y.shape))
#     print('X_test:  '  + str(test_X.shape))
#     print('Y_test:  '  + str(test_y.shape))

#     if plot is True:
#         # Plotting
#         from matplotlib import pyplot
#         for i in range(9):  
#             pyplot.subplot(330 + 1 + i)
#             pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
#             pyplot.show()

#     # Select data corresponding to a subset of the latbels
#     inds = []
#     inds_test = []
#     for ind in labels_keep:
#         inds.append(np.where(train_y==ind)[0])
#         inds_test.append(np.where(test_y==ind)[0])
        
#     inds = np.hstack(inds)
#     inds_test = np.hstack(inds_test)

#     # Select training set
#     X_in = train_X[inds,:].reshape((train_X[inds,:].shape[0],28*28))
#     true_clusters_in = pd.factorize(train_y[inds])[0]
    
#     # Randomize training setorder
#     if randomize is True:
#         print('Randomizing order seed 42')
#         idxs = np.random.RandomState(seed=42).permutation(X_in.shape[0])
#         X_in2 = X_in[idxs,:]
#         true_clusters_in2 = true_clusters_in[idxs]
        
#         X_in = X_in2
#         true_clusters_in = true_clusters_in2
#         del X_in2, true_clusters_in2

#     # subset evenly sampled across remaining n_X using skip
#     X_in = X_in[::skip]
#     true_clusters_in = true_clusters_in[::skip]
#     true_clusters_labels = pd.factorize(train_y[inds])[1]
#     num_clusters = len(np.unique(true_clusters_in))
#     print('X_train',X_in.shape, 'Y_train', true_clusters_in.shape)    
#     print('Class labels',true_clusters_labels,'indexed as',np.unique(true_clusters_in))    
#     # Select test set
#     X_in_test = test_X[inds_test,:].reshape((test_X[inds_test,:].shape[0],28*28))
#     true_clusters_in_test = pd.factorize(test_y[inds_test])[0]
#     print('X_test',X_in.shape, 'Y_test', true_clusters_in.shape)

#     # Subset training set to be a multiple of the batch size, batch_size
#     n_X = X_in.shape[0]
#     mod = n_X % batch_size
#     n_X_new = n_X - mod
#     print('n_X', n_X, 'batch size', batch_size, 'modulo', mod, 'is now', 'n_X', n_X_new, 'batch size', batch_size, 'modulo', (n_X - mod) % batch_size)

#     X_in = X_in[0:n_X_new,:]
#     true_clusters_in = true_clusters_in[0:n_X_new]

#     print('Training set is now:', X_in.shape, true_clusters_in.shape)
    
#     return X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters

import numpy as np
import torch
from torch.utils.data import Dataset
import scanpy as sc
from anndata import AnnData
import os
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import os
import numpy as np
import pickle
from copy import deepcopy
from shutil import move
import warnings

"""Machine learning and single cell packages"""
import sklearn.metrics as metrics
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
import scanpy as sc
from anndata import AnnData
import seaborn as sns

import numpy as np
import os
from scipy.sparse import issparse

import scanpy as sc
from anndata import AnnData

def normalize_scanpy(adata, batch_key = None, n_high_var = 1000, LVG = True, 
                     normalize_samples = True, log_normalize = True, 
                     normalize_features = True):
    """ This function preprocesses the raw count data.
    
    
    Arguments:
    ------------------------------------------------------------------
    - adata: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to genes.
    - batch_key: `str`, string specifying the name of the column in the observation dataframe which identifies the batch of each cell. If this is left as None, then all cells are assumed to be from one batch.
    - n_high_var: `int`, integer specifying the number of genes to be idntified as highly variable. E.g. if n_high_var = 2000, then the 2000 genes with the highest variance are designated as highly variable.
    - LVG: `bool`, Whether to retain and preprocess LVGs.
    - normalize_samples: `bool`, If True, normalize expression of each gene in each cell by the sum of expression counts in that cell.
    - log_normalize: `bool`, If True, log transform expression. I.e., compute log(expression + 1) for each gene, cell expression count.
    - normalize_features: `bool`, If True, z-score normalize each gene's expression.
    
    Returns:
    ------------------------------------------------------------------
    - adata: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Contains preprocessed data.
    """
    
    n, p = adata.shape
    sparsemode = issparse(adata.X)
    
    if batch_key is not None:
        batch = list(adata.obs[batch_key])
        batch = convert_vector_to_encoding(batch)
        batch = np.asarray(batch)
        batch = batch.astype('float32') #float32
    else:
        batch = np.ones((n,), dtype = 'float32') #float32
        norm_by_batch = False
        
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
        
    count = adata.X.copy()
        
    if normalize_samples:
        out = sc.pp.normalize_total(adata, inplace = False)
        obs_ = adata.obs
        var_ = adata.var
        adata = None
        adata = AnnData(out['X'])
        adata.obs = obs_
        adata.var = var_
        
        size_factors = out['norm_factor'] / np.median(out['norm_factor'])
        out = None
    else:
        size_factors = np.ones((adata.shape[0], ))
        
    if not log_normalize:
        adata_ = adata.copy()
    
    sc.pp.log1p(adata)
    
    if n_high_var is not None:
        sc.pp.highly_variable_genes(adata, inplace = True, min_mean = 0.0125, max_mean = 3, min_disp = 0.5, 
                                          n_bins = 20, n_top_genes = n_high_var, batch_key = batch_key)
        
        hvg = adata.var['highly_variable'].values
        
        if not log_normalize:
            adata = adata_.copy()

    else:
        hvg = [True] * adata.shape[1]
        
    if normalize_features:
        batch_list = np.unique(batch)

        if sparsemode:
            adata.X = adata.X.toarray()

        for batch_ in batch_list:
            indices = [x == batch_ for x in batch]
            sub_adata = adata[indices]
            
            sc.pp.scale(sub_adata)
            adata[indices] = sub_adata.X
        
        adata.layers["normalized input"] = adata.X
        adata.X = count
        adata.var['Variance Type'] = [['LVG', 'HVG'][int(x)] for x in hvg]
            
    else:
        if sparsemode:   
            adata.layers["normalized input"] = adata.X.toarray()
        else:
            adata.layers["normalized input"] = adata.X
            
        adata.var['Variance Type'] = [['LVG', 'HVG'][int(x)] for x in hvg]
        
    if n_high_var is not None:
        del_keys = ['dispersions', 'dispersions_norm', 'highly_variable', 'highly_variable_intersection', 'highly_variable_nbatches', 'means']
        del_keys = [x for x in del_keys if x in adata.var.keys()]
        adata.var = adata.var.drop(del_keys, axis = 1)
            
    y = np.unique(batch)
    num_batch = len(y)
    
    adata.obs['size factors'] = size_factors.astype('float32') #float32
    adata.obs['batch'] = batch
    adata.uns['num_batch'] = num_batch
    
    if sparsemode:
        adata.X = adata.X.toarray()
        
    if not LVG:
        adata = adata[:, adata.var['Variance Type'] == 'HVG']
        
    return adata


def build_dir(dir_path):
    """ This function builds a directory if it does not exist.
    
    
    Arguments:
    ------------------------------------------------------------------
    - dir_path: `str`, The directory to build. E.g. if dir_path = 'folder1/folder2/folder3', then this function will creates directory if folder1 if it does not already exist. Then it creates folder1/folder2 if folder2 does not exist in folder1. Then it creates folder1/folder2/folder3 if folder3 does not exist in folder2.
    """
    
    subdirs = [dir_path]
    substring = dir_path

    while substring != '':
        splt_dir = os.path.split(substring)
        substring = splt_dir[0]
        subdirs.append(substring)
        
    subdirs.pop()
    subdirs = [x for x in subdirs if os.path.basename(x) != '..']

    n = len(subdirs)
    subdirs = [subdirs[n - 1 - x] for x in range(n)]
    
    for dir_ in subdirs:
        if not os.path.isdir(dir_):
            os.mkdir(dir_)


def convert_string_to_encoding(string, vector_key):
    """A function to convert a string to a numeric encoding.
    
    
    Arguments:
    ------------------------------------------------------------------
    - string: `str`, The specific string to convert to a numeric encoding.
    - vector_key: `np.ndarray`, Array of all possible values of string.
    
    Returns:
    ------------------------------------------------------------------
    - encoding: `int`, The integer encoding of string.
    """
    
    return np.argwhere(vector_key == string)[0][0]


def convert_vector_to_encoding(vector):
    """A function to convert a vector of strings to a dense numeric encoding.
    
    
    Arguments:
    ------------------------------------------------------------------
    - vector: `array_like`, The vector of strings to encode.
    
    Returns:
    ------------------------------------------------------------------
    - vector_num: `list`, A list containing the dense numeric encoding.
    """
    
    vector_key = np.unique(vector)
    vector_strings = list(vector)
    vector_num = [convert_string_to_encoding(string, vector_key) for string in vector_strings]
    
    return vector_num


def find_resolution(adata_, n_clusters, random):
    """A function to find the louvain resolution tjat corresponds to a prespecified number of clusters, if it exists.
    
    
    Arguments:
    ------------------------------------------------------------------
    - adata_: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to low dimension features.
    - n_clusters: `int`, Number of clusters.
    - random: `int`, The random seed.
    
    Returns:
    ------------------------------------------------------------------
    - resolution: `float`, The resolution that gives n_clusters after running louvain's clustering algorithm.
    """
    
    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]
    
    while obtained_clusters != n_clusters and iteration < 50:
        current_res = sum(resolutions)/2
        adata = sc.tl.louvain(adata_, resolution = current_res, random_state = random, copy = True)
        labels = adata.obs['louvain']
        obtained_clusters = len(np.unique(labels))
        
        if obtained_clusters < n_clusters:
            resolutions[0] = current_res
        else:
            resolutions[1] = current_res
        
        iteration = iteration + 1


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from pandas import DataFrame

import os


import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import set_floatx

from sklearn.cluster import KMeans

import scanpy as sc
from anndata import AnnData
import pandas as pd

import random
import numpy as np
from math import ceil

import os
from copy import deepcopy
from time import time

set_floatx('float32') #float32

class CarDEC_Model(Model):
    def __init__(self, adata, dims, LVG_dims = None, tol = 0.005, n_clusters = None, random_seed = 201809, 
                 louvain_seed = 0, n_neighbors = 15, pretrain_epochs = 300, batch_size = 64, decay_factor = 1/3, 
                 patience_LR = 3, patience_ES = 9, act = 'relu', actincenter = "tanh", ae_lr = 1e-04, clust_weight = 1., 
                 load_encoder_weights = True, set_centroids = True, weights_dir = "CarDEC Weights"):
        super(CarDEC_Model, self).__init__()
        """ This class creates the TensorFlow CarDEC model architecture.
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to genes.
        - dims: `list`, the number of output features for each layer of the HVG encoder. The length of the list determines the number of layers.
        - LVG_dims: `list`, the number of output features for each layer of the LVG encoder. The length of the list determines the number of layers.
        - tol: `float`, stop criterion, clustering procedure will be stopped when the difference ratio between the current iteration and last iteration larger than tol.
        - n_clusters: `int`, The number of clusters into which cells will be grouped.
        - random_seed: `int`, The seed used for random weight intialization.
        - louvain_seed: `int`, The seed used for louvain clustering intialization.
        - n_neighbors: `int`, The number of neighbors used for building the graph needed for louvain clustering.
        - pretrain_epochs: `int`, The maximum number of epochs for pretraining the HVG autoencoder. In practice, early stopping criteria should stop training much earlier.
        - batch_size: `int`, The batch size used for pretraining the HVG autoencoder.
        - decay_factor: `float`, The multiplicative factor by which to decay the learning rate when validation loss is not decreasing.
        - patience_LR: `int`, the number of epochs which the validation loss is allowed to increase before learning rate is decayed when pretraining the autoencoder.
        - patience_ES: `int`, the number of epochs which the validation loss is allowed to increase before training is halted when pretraining the autoencoder.
        - act: `str`, The activation function used for the intermediate layers of CarDEC, other than the bottleneck layer.
        - actincenter: `str`, The activation function used for the bottleneck layer of CarDEC.
        - ae_lr: `float`, The learning rate for pretraining the HVG autoencoder.
        - clust_weight: `float`, a number between 0 and 2 qhich balances the clustering and reconstruction losses.
        - load_encoder_weights: `bool`, If True, the API will try to load the weights for the HVG encoder from the weight directory.
        - set_centroids: `bool`, If True, intialize the centroids by running Louvain's algorithm.
        - weights_dir: `str`, the path in which to save the weights of the CarDEC model.
        ------------------------------------------------------------------
        """
        
        assert clust_weight <= 2. and clust_weight>=0.
        
        tf.keras.backend.clear_session()
                    
        self.dims = dims
        self.LVG_dims = LVG_dims
        self.tol = tol
        self.input_dim = dims[0]  # for clustering layer 
        self.n_stacks = len(self.dims) - 1
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.activation = act
        self.actincenter = actincenter
        self.load_encoder_weights = load_encoder_weights
        self.clust_weight = clust_weight
        self.weights_dir = weights_dir
        self.preclust_embedding = None
        
        # set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        self.splitseed = round(abs(10000*np.random.randn()))
        
        # build the autoencoder
        self.sae = SAE(dims = self.dims, act = self.activation, actincenter = self.actincenter, 
                       random_seed = random_seed, splitseed = self.splitseed, init="glorot_uniform", optimizer = Adam(), 
                       weights_dir = weights_dir)
        
        build_dir(self.weights_dir)
        
        decoder_seed = round(100 * abs(np.random.normal()))
        if load_encoder_weights:
            if os.path.isfile("./" + self.weights_dir + "/pretrained_autoencoder_weights.index"):
                print("Pretrain weight index file detected, loading weights.")
                self.sae.load_autoencoder()
                print("Pretrained high variance autoencoder weights initialized.")
            else:
                print("Pretrain weight index file not detected, pretraining autoencoder weights.\n")
                self.sae.train(adata, lr = ae_lr, num_epochs = pretrain_epochs, 
                               batch_size = batch_size, decay_factor = decay_factor, 
                               patience_LR = patience_LR, patience_ES = patience_ES)
                self.sae.load_autoencoder()
        else:
            print("Pre-training high variance autoencoder.\n")
            self.sae.train(adata, lr = ae_lr, num_epochs = pretrain_epochs, 
                           batch_size = batch_size, decay_factor = decay_factor, 
                           patience_LR = patience_LR, patience_ES = patience_ES)
            self.sae.load_autoencoder()
        
        features = self.sae.embed(adata)
        self.preclust_emb = deepcopy(features)
        self.preclust_denoised = self.sae.denoise(adata, batch_size)
                
        if not set_centroids:
            self.init_centroid = np.zeros((n_clusters, self.dims[-1]), dtype = 'float32') #float32
            self.n_clusters = n_clusters
            self.init_pred = np.zeros((adata.shape[0], dims[-1]))
            
        elif louvain_seed is None:
            print("\nInitializing cluster centroids using K-Means")

            kmeans = KMeans(n_clusters=n_clusters, n_init = 20)
            Y_pred_init = kmeans.fit_predict(features)

            self.init_pred = deepcopy(Y_pred_init)
            self.n_clusters = n_clusters
            self.init_centroid = kmeans.cluster_centers_
            
        else:
            print("\nInitializing cluster centroids using the louvain method.")
            
            n_cells = features.shape[0]
            
            if n_cells > 10**5:
                subset = np.random.choice(range(n_cells), 10**5, replace = False)
                adata0 = AnnData(features[subset])
            else: 
                adata0 = AnnData(features)

            sc.pp.neighbors(adata0, n_neighbors = self.n_neighbors, use_rep="X")
            self.resolution = find_resolution(adata0, n_clusters, louvain_seed)
            adata0 = sc.tl.louvain(adata0, resolution = self.resolution, random_state = louvain_seed, copy = True)

            Y_pred_init = adata0.obs['louvain']
            self.init_pred = np.asarray(Y_pred_init, dtype=int)

            features = pd.DataFrame(adata0.X, index = np.arange(0, adata0.shape[0]))
            Group = pd.Series(self.init_pred, index = np.arange(0, adata0.shape[0]), name="Group")
            Mergefeature = pd.concat([features, Group],axis=1)

            self.init_centroid = np.asarray(Mergefeature.groupby("Group").mean())
            self.n_clusters = self.init_centroid.shape[0]

            print("\n " + str(self.n_clusters) + " clusters detected. \n")
        
        self.encoder = self.sae.encoder
        self.decoder = self.sae.decoder
        
        if LVG_dims is not None:
            n_stacks = len(dims) - 1

            LVG_encoder_layers = []

            for i in range(n_stacks-1):
                LVG_encoder_layers.append(Dense(LVG_dims[i + 1], kernel_initializer = 'glorot_uniform', activation = self.activation, name='encoder%d' % i))

            LVG_encoder_layers.append(Dense(LVG_dims[-1], kernel_initializer = 'glorot_uniform', activation = self.actincenter, name='embedding'))
            self.encoderLVG = Sequential(LVG_encoder_layers, name = 'encoderLVG')

        if LVG_dims is not None:
            decoder_layers = []
            for i in range(self.n_stacks - 1, 0, -1):
                decoder_layers.append(Dense(self.LVG_dims[i], kernel_initializer = 'glorot_uniform', 
                                            activation = self.activation, name='decoderLVG%d' % (i-1)))
                
            decoder_layers.append(Dense(self.LVG_dims[0], activation = 'linear', name='outputLVG'))
            self.decoderLVG = Sequential(decoder_layers, name = 'decoderLVG')
        
        self.clustering_layer = ClusteringLayer(centroids = self.init_centroid, name = 'clustering')
        
        del self.sae
        
        self.construct()
        
    def construct(self, summarize = True):
        """ This class method fully initalizes the TensorFlow model.
        Arguments:
        ------------------------------------------------------------------
        - summarize: `bool`, If True, then print a summary of the model architecture.
        """
        
        x = [tf.zeros(shape = (1, self.dims[0]), dtype=float), None]
        if self.LVG_dims is not None:
            x[1] = tf.zeros(shape = (1, self.LVG_dims[0]), dtype=float)
            
        out = self(*x)
        
        if summarize:
            print("\n-----------------------CarDEC Architecture-----------------------\n")
            self.summary()

            print("\n--------------------Encoder Sub-Architecture--------------------\n")
            self.encoder.summary()
            
            print("\n------------------Base Decoder Sub-Architecture------------------\n")
            self.decoder.summary()

            if self.LVG_dims is not None:
                print("\n------------------LVG Encoder Sub-Architecture------------------\n")
                self.encoderLVG.summary()

                print("\n----------------LVG Base Decoder Sub-Architecture----------------\n")
                self.decoderLVG.summary()

    def call(self, hvg, lvg, denoise = True):
        """ This is the forward pass of the model.
        
        ***Inputs***
            - hvg: `tf.Tensor`, an input tensor of shape (n_obs, n_HVG).
            - lvg: `tf.Tensor`, (Optional) an input tensor of shape (n_obs, n_LVG).
            - denoise: `bool`, (Optional) If True, return denoised expression values for each cell.
            
        ***Outputs***
            - denoised_output: `dict`, (Optional) Dictionary containing denoised tensors.
            - cluster_output: `tf.Tensor`, a tensor of cell cluster membership probabilities of shape (n_obs, m).
        """
        
        hvg = self.encoder(hvg)

        cluster_output = self.clustering_layer(hvg)
        
        if not denoise:
            return cluster_output

        HVG_denoised_output = self.decoder(hvg)
        denoised_output = {'HVG_denoised': HVG_denoised_output}

        if self.LVG_dims is not None:
            lvg = self.encoderLVG(lvg)
            z = concatenate([hvg, lvg], axis=1)

            LVG_denoised_output = self.decoderLVG(z)

            denoised_output['LVG_denoised'] = LVG_denoised_output

        return denoised_output, cluster_output

    @staticmethod
    def target_distribution(q):
        """ Updates target distribution cluster assignment probabilities given CarDEC output.
        
        
        Arguments:
        ------------------------------------------------------------------
        - q: `tf.Tensor`, a tensor of shape (b, m) identifying the probability that each of b cells is in each of the m clusters. Obtained as output from CarDEC.
        
        Returns:
        ------------------------------------------------------------------
        - p: `tf.Tensor`, a tensor of shape (b, m) identifying the pseudo-label probability that each of b cells is in each of the m clusters.
        """
        
        weight = q ** 2 / np.sum(q, axis = 0)
        p = weight.T / np.sum(weight, axis = 1)
        return p.T
    
    def make_generators(self, adata, val_split, batch_size):
        """ This class method creates training and validation data generators for the current input data and pseudo labels.
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to genes.
        - val_split: `float`, The fraction of cells to be reserved for validation during this step.
        - batch_size: `int`, The batch size used for training the full model.
        - p: `tf.Tensor`, a tensor of shape (b, m) identifying the pseudo-label probability that each of b cells is in each of the m clusters.
        - splitseed: `int`, The seed used to split cells between training and validation. Should be consistent between iterations to ensure the same cells are always used for validation.
        - newseed: `int`, The seed that is set after splitting cells between training and validation. Should be different every iteration so that stochastic operations other than splitting cells between training and validation vary between epochs.
        
        Returns:
        ------------------------------------------------------------------
        - train_dataset: `tf.data.Dataset`, Dataset that returns training examples.
        - val_dataset: `tf.data.Dataset`, Dataset that returns validation examples.
        """
                
        if self.LVG_dims is None:
            hvg_input = adata.layers["normalized input"]
            hvg_target = adata.layers["normalized input"]
            lvg_input = None
            lvg_target = None
        else:
            hvg_input = adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG']
            hvg_target = adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG']
            lvg_input = adata.layers["normalized input"][:, adata.var['Variance Type'] == 'LVG']
            lvg_target = adata.layers["normalized input"][:, adata.var['Variance Type'] == 'LVG']
                    
        return dataloader(hvg_input, hvg_target, lvg_input, lvg_target, val_split, batch_size, self.splitseed)
        
    def train_loop(self, train_dataset):
        """ This class method runs the training loop.
        
        
        Arguments:
        ------------------------------------------------------------------
        - train_dataset: `tf.data.Dataset`, Dataset that returns training examples.
        
        Returns:
        ------------------------------------------------------------------
        - epoch_loss_avg: `float`, The mean training loss for the iteration.
        """

        epoch_loss_avg = tf.keras.metrics.Mean()
        
        for inputs, target, LVG_target, batch_p in train_dataset(val = False):
            loss_value, grads = grad(self, inputs, target, batch_p, total_loss = total_loss,
                                     LVG_target = LVG_target, aeloss_fun = MSEloss, 
                                     clust_weight = self.clust_weight)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            epoch_loss_avg(loss_value)
                
        return epoch_loss_avg.result()
                
    def validation_loop(self, val_dataset):
        """ This class method runs the validation loop.
        
        
        Arguments:
        ------------------------------------------------------------------
        - val_dataset: `tf.data.Dataset`, Dataset that returns validation examples.
        
        Returns:
        ------------------------------------------------------------------
        - epoch_loss_avg: `float`, The mean validation loss for the iteration (reconstruction + clustering loss)
        - epoch_aeloss_avg_val: `float`, The mean validation reconstruction loss for the iteration
        """

        epoch_loss_avg_val = tf.keras.metrics.Mean()
        epoch_aeloss_avg_val = tf.keras.metrics.Mean()
            
        for inputs, target, LVG_target, batch_p in val_dataset(val = True):
            denoised_output, cluster_output = self(*inputs)
            loss_value, aeloss = total_loss(target, denoised_output, batch_p, cluster_output, 
                           LVG_target = LVG_target, aeloss_fun = MSEloss, clust_weight = self.clust_weight)
            epoch_loss_avg_val(loss_value)
            epoch_aeloss_avg_val(aeloss)
                
        return epoch_loss_avg_val.result(), epoch_aeloss_avg_val.result()
    
    def package_output(self, adata, init_pred, preclust_denoised, preclust_emb):
        """ This class adds some quantities to the adata object.
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to genes.
        - init_pred: `np.ndarray`, the array of initial cluster assignments for each cells, of shape (n_obs,).
        - preclust_denoised: `np.ndarray`, This is the array of feature zscores denoised with the pretrained autoencoder of shape (n_obs, n_vars).
        - preclust_emb: `np.ndarray`, This is the latent embedding from the pretrained autoencoder of shape (n_obs, n_embedding).
        """        
        
        adata.obsm['precluster denoised'] = preclust_denoised
        adata.obsm['precluster embedding'] = preclust_emb
        if adata.shape[0] == init_pred.shape[0]:
            adata.obsm['initial assignments'] = init_pred
    
    def embed(self, adata, batch_size):
        """ This class method can be used to compute the low-dimension embedding for HVG features. 
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to genes.
        - batch_size: `int`, The batch size for filling the array of low dimension embeddings.
        
        Returns:
        ------------------------------------------------------------------
        - embedding: `np.ndarray`, Array of shape (n_obs, p_embedding) containing the HVG embedding for every cell in the dataset.
        """
        
        input_ds = simpleloader(adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG'], batch_size)
        
        embedding = np.zeros((adata.shape[0], self.dims[-1]), dtype = 'float32') #float32
        start = 0

        for x in input_ds:
            end = start + x.shape[0]
            embedding[start:end] = self.encoder(x).numpy()
            start = end
            
        return embedding
    
    def embed_LVG(self, adata, batch_size):
        """ This class method can be used to compute the low-dimension embedding for LVG features. 
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to genes.
        - batch_size: `int`, The batch size for filling the array of low dimension embeddings.
        
        Returns:
        ------------------------------------------------------------------
        - embedding: `np.ndarray`, Array of shape (n_obs, n_embedding) containing the LVG embedding for every cell in the dataset.
        """
        
        input_ds = simpleloader(adata.layers["normalized input"][:, adata.var['Variance Type'] == 'LVG'], batch_size)

        LVG_embedded = np.zeros((adata.shape[0], self.LVG_dims[-1]), dtype = 'float32') #float32
        start = 0

        for x in input_ds:
            end = start + x.shape[0]
            LVG_embedded[start:end] = self.encoderLVG(x).numpy()
            start = end

        return np.concatenate((adata.obsm['embedding'], LVG_embedded), axis = 1)
    
    def make_outputs(self, adata, batch_size, denoise = True):
        """ This class method can be used to pack all relvant outputs into the adata object after training.
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The annotated data matrix of shape (n_obs, n_vars).
        - batch_size: `int`, The batch size for filling the array of low dimension embeddings.
        - denoise: `bool`, Whether to provide denoised expression values for all cells.
        """
        
        if not denoise:
            input_ds = simpleloader(adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG'], batch_size)
            adata.obsm["cluster memberships"] = np.zeros((adata.shape[0], self.n_clusters), dtype = 'float32') #float32
            
            start = 0     
            for x in input_ds:
                q_batch = self(x, None, False)
                end = start + q_batch.shape[0]
                adata.obsm["cluster memberships"][start:end] = q_batch.numpy()
            
                start = end
            
            
        elif self.LVG_dims is not None:
            if not ('embedding' in list(adata.obsm) and 'LVG embedding' in list(adata.obsm)):
                        adata.obsm['embedding'] = self.embed(adata, batch_size)
                        adata.obsm['LVG embedding'] = self.embed_LVG(adata, batch_size)
            input_ds = tupleloader(adata.obsm["embedding"], adata.obsm["LVG embedding"], batch_size = batch_size)
            
            adata.obsm["cluster memberships"] = np.zeros((adata.shape[0], self.n_clusters), dtype = 'float32') #float32
            adata.layers["denoised"] = np.zeros(adata.shape, dtype = 'float32') #float32
            
            start = 0     
            for input_ in input_ds:
                denoised_batch = {'HVG_denoised': self.decoder(input_[0]), 'LVG_denoised': self.decoderLVG(input_[1])}
                q_batch = self.clustering_layer(input_[0])
                end = start + q_batch.shape[0]
                
                adata.obsm["cluster memberships"][start:end] = q_batch.numpy()
                adata.layers["denoised"][start:end, adata.var['Variance Type'] == 'HVG'] = denoised_batch['HVG_denoised'].numpy()
                adata.layers["denoised"][start:end, adata.var['Variance Type'] == 'LVG'] = denoised_batch['LVG_denoised'].numpy()
            
                start = end
        
        else:
            if not ('embedding' in list(adata.obsm)):
                adata.obsm['embedding'] = self.embed(adata, batch_size)
            input_ds = simpleloader(adata.obsm["embedding"], batch_size)
            
            adata.obsm["cluster memberships"] = np.zeros((adata.shape[0], self.n_clusters), dtype = 'float32') #float32
            adata.layers["denoised"] = np.zeros(adata.shape, dtype = 'float32') #float32
            
            start = 0
            
            for input_ in input_ds:
                denoised_batch = {'HVG_denoised': self.decoder(input_)}
                q_batch = self.clustering_layer(input_)
                
                end = start + q_batch.shape[0]
                
                adata.obsm["cluster memberships"][start:end] = q_batch.numpy()
                adata.layers["denoised"][start:end] = denoised_batch['HVG_denoised'].numpy()
                
                start = end
                
    def train(self, adata, batch_size = 64, val_split = 0.1, lr = 1e-04, decay_factor = 1/3,
              iteration_patience_LR = 3, iteration_patience_ES = 6, 
              maxiter = 1e3, epochs_fit = 1, optimizer = Adam(), printperiter = None, denoise = True):
        """ This class method can be used to train the main CarDEC model
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The annotated data matrix of shape (n_obs, n_vars).
        - batch_size: `int`, The batch size used for training the full model.
        - val_split: `float`, The fraction of cells to be reserved for validation during this step.
        - lr: `float`, The learning rate for training the full model.
        - decay_factor: `float`, The multiplicative factor by which to decay the learning rate when validation loss is not decreasing.
        - iteration_patience_LR: `int`, The number of iterations tolerated before decaying the learning rate during which the number of cells that change assignment is less than tol.
        - iteration_patience_ES: `int`, The number of iterations tolerated before stopping training during which the number of cells that change assignment is less than tol.
        - maxiter: `int`, The maximum number of iterations allowed to train the full model. In practice, the model will halt training long before hitting this limit.
        - epochs_fit: `int`, The number of epochs during which to fine-tune weights, before updating the target distribution.
        - optimizer: `tensorflow.python.keras.optimizer_v2`, An instance of a TensorFlow optimizer.
        - printperiter: `int`, Optional integer argument. If specified, denoised values will be returned every printperiter epochs, so that the user can evaluate the progress of denoising as training continues.
        - denoise: `bool`, If True, then denoised expression values are provided for all cells.
        
        Returns:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The updated annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to genes. Depending on the arguments of the train call, some outputs will be added to adata.
        """
        
        total_start = time()
        seedlist = list(1000*np.random.randn(int(maxiter)))
        seedlist = [abs(int(x)) for x in seedlist]
        
        self.optimizer = optimizer
        self.optimizer.lr = lr
        
        # Begin deep clustering
        y_pred_last = np.ones((adata.shape[0],), dtype = int) * -1.

        min_delta = np.inf
        current_aeloss_val = np.inf
        delta_patience_ES = 0
        delta_patience_LR = 0
        delta_stop = False
        
        dataset = self.make_generators(adata, val_split = 0.1, batch_size = batch_size)
        
        self.make_outputs(adata, batch_size, denoise = printperiter is not None)
        
        for ite in range(int(maxiter)):
            
            p = self.target_distribution(adata.obsm['cluster memberships'])
            
            dataset.update_p(p)

            best_loss = np.inf
            iter_start = time()
                        
            for epoch in range(epochs_fit):
                current_loss_train = self.train_loop(dataset)
                current_loss_val, current_aeloss_val = self.validation_loop(dataset)
            
            self.make_outputs(adata, batch_size, denoise = printperiter is not None)
            
            y_pred = np.argmax(adata.obsm['cluster memberships'], axis = 1)
                        
            if printperiter is not None:
                if ite % printperiter == 0 and ite > 0:
                    denoising_filename = os.path.join(CarDEC.weights_dir, '/intermediate_denoising/denoised' + ite)
                    outfile = open(denoising_filename,'wb')
                    pickle.dump(adata.layers["denoised"][:, adata.var['Variance Type'] == 'HVG'], outfile)
                    outfile.close()
                    
                    if self.LVG_dims is not None:
                        denoising_filename = os.path.join(CarDEC.weights_dir, '/intermediate_denoising/denoisedLVG' + ite)
                        outfile = open(denoising_filename,'wb')
                        pickle.dump(adata.layers["denoised"][:, adata.var['Variance Type'] == 'LVG'], outfile)
                        outfile.close()
            
            # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0] #float32
            y_pred_last = deepcopy(y_pred)
            
            current_aeloss_val = current_aeloss_val.numpy()
            current_clustloss_val = (current_loss_val.numpy() - (1 - self.clust_weight) * current_aeloss_val)/self.clust_weight
            print("Iter {:03d} Loss: [Training: {:.3f}, Validation Cluster: {:.3f}, Validation AE: {:.3f}], Label Change: {:.3f}, Time: {:.1f} s".format(ite, current_loss_train.numpy(), current_clustloss_val, current_aeloss_val, delta_label, time() - iter_start))
            
            if current_aeloss_val + 10**(-3) < min_delta:
                min_delta = current_aeloss_val
                delta_patience_ES = 0
                delta_patience_LR = 0
                
            if delta_patience_ES >= iteration_patience_ES:
                delta_stop = True
                
            if delta_patience_LR >= iteration_patience_LR:
                self.optimizer.lr = self.optimizer.lr * decay_factor
                delta_patience_LR = 0
                print("\nDecaying Learning Rate to: " + str(self.optimizer.lr.numpy()))

            delta_patience_ES = delta_patience_ES + 1
            delta_patience_LR = delta_patience_LR + 1
            
            if delta_stop and delta_label < self.tol:
                print('\nAutoencoder_loss ', current_aeloss_val, 'not improving.')
                print('Proportion of Labels Changed: ', delta_label, ' is less than tolerance of ', self.tol)
                print('\nReached tolerance threshold. Stop training.')
                break
                
                        
        y0 = pd.Series(y_pred, dtype='category')
        y0.cat.categories = range(0, len(y0.cat.categories))
        print("\nThe final cluster assignments are:")
        x = y0.value_counts()
        print(x.sort_index(ascending=True))
        
        adata.obsm['embedding'] = self.embed(adata, batch_size)
        if self.LVG_dims is not None:
            adata.obsm['LVG embedding'] = self.embed_LVG(adata, batch_size)
            
        del adata.layers['normalized input']
        
        if denoise:
            self.make_outputs(adata, batch_size, denoise = True)
        
        self.save_weights("./" + self.weights_dir + "/tuned_CarDECweights", save_format='tf')
                   
        print("\nTotal Runtime is " + str(time() - total_start))
                
        print("\nThe CarDEC model is now making inference on the data matrix.")
        
        self.package_output(adata, self.init_pred, self.preclust_denoised, self.preclust_emb)
            
        print("Inference completed, results added.")
        
        return adata
    
    def reload_model(self, adata = None, batch_size = 64, denoise = True):
        """ This class method can be used to load the model's saved weights and redo inference.
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, (Optional) The annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to genes. If left as None, model weights will be reloaded but inference will not be made.
        - batch_size: `int`, The batch size for filling the array of low dimension embeddings.
        - denoise: `bool`, Whether to provide denoised expression values for all cells.
        
        Returns:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, (Optional) The annotated data matrix of shape (n_obs, n_vars). If an adata object was provided as input, the adata object will be returned with inference outputs added.
        """
        
        if os.path.isfile("./" + self.weights_dir + "/tuned_CarDECweights.index"):
            print("Weight index file detected, loading weights.")
            self.load_weights("./" + self.weights_dir + "/tuned_CarDECweights").expect_partial()
            print("CarDEC Model weights loaded successfully.")
        
            if adata is not None:
                print("\nThe CarDEC model is now making inference on the data matrix.")
                
                adata.obsm['embedding'] = self.embed(adata, batch_size)
                if self.LVG_dims is not None:
                    adata.obsm['LVG embedding'] = self.embed_LVG(adata, batch_size)
                    
                del adata.layers['normalized input']
                
                if denoise:
                    self.make_outputs(adata, batch_size, True)
                
                self.package_output(adata, self.init_pred, self.preclust_denoised, self.preclust_emb)
                
                print("Inference completed, results returned.")
                
                return adata

        else:
            print("\nWeight index file not detected, please call CarDEC_Model.train to learn the weights\n")


import tensorflow as tf
from tensorflow.keras.layers import Layer

class ClusteringLayer(Layer):
    def __init__(self, centroids = None, n_clusters = None, n_features = None, alpha=1.0, **kwargs):
        """ The clustering layer predicts the a cell's class membership probability for each cell.
        
        
        Arguments:
        ------------------------------------------------------------------
        - centroids: `tf.Tensor`, Initial cluster ceontroids after pretraining the model.
        - n_clusters: `int`, Number of clusters.
        - n_features: `int`, The number of features of the bottleneck embedding space that the centroids live in.
        - alpha: parameter in Student's t-distribution. Default to 1.0.
        """
        
        super(ClusteringLayer, self).__init__(**kwargs)
        self.alpha = alpha
        self.initial_centroids = centroids

        if centroids is not None:
            n_clusters, n_features = centroids.shape

        self.n_features, self.n_clusters = n_features, n_clusters

        assert self.n_clusters is not None
        assert self.n_features is not None

    def build(self, input_shape):
        """ This class method builds the layer fully once it receives an input tensor.
        
        
        Arguments:
        ------------------------------------------------------------------
        - input_shape: `list`, A list specifying the shape of the input tensor.
        """
        
        assert len(input_shape) == 2
        
        self.centroids = self.add_weight(name = 'clusters', shape = (self.n_clusters, self.n_features), initializer = 'glorot_uniform')
        if self.initial_centroids is not None:
            self.set_weights([self.initial_centroids])
            del self.initial_centroids
        
        self.built = True

    def call(self, x, **kwargs):
        """ Forward pass of the clustering layer,
        
        
        ***Inputs***:
            - x: `tf.Tensor`, the embedding tensor of shape = (n_obs, n_var)
        
        ***Returns***:
            - q: `tf.Tensor`, student's t-distribution, or soft labels for each sample of shape = (n_obs, n_clusters)
        """

        q = 1.0 / (1.0 + (tf.reduce_sum(tf.square(tf.expand_dims(x, axis = 1) - self.centroids), axis = 2) / self.alpha))
        q = q**((self.alpha + 1.0) / 2.0)
        q = q / tf.reduce_sum(q, axis = 1, keepdims = True)

        return q

    def compute_output_shape(self, input_shape):
        """ This method infers the output shape from the input shape.
        
        
        Arguments:
        ------------------------------------------------------------------
        - input_shape: `list`, A list specifying the shape of the input tensor.
        
        Returns:
        ------------------------------------------------------------------
        - output_shape: `list`, A tuple specifying the shape of the output for the minibatch (n_obs, n_clusters)
        """
        
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters


import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import set_floatx
from time import time

import random
import numpy as np
from scipy.stats import zscore
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.losses import KLD, MSE


def grad_MainModel(model, input_, target, target_p, total_loss, LVG_target = None, aeloss_fun = None, clust_weight = 1.):
    """Function to do a backprop update to the main CarDEC model for a minibatch.
    
    
    Arguments:
    ------------------------------------------------------------------
    - model: `tensorflow.keras.Model`, The main CarDEC model.
    - input_: `list`, A list containing the input HVG and (optionally) LVG expression tensors of the minibatch for the CarDEC model.
    - target: `tf.Tensor`, Tensor containing the reconstruction target of the minibatch for the HVGs.
    - target_p: `tf.Tensor`, Tensor containing cluster membership probability targets for the minibatch.
    - total_loss: `function`, Function to compute the loss for the main CarDEC model for a minibatch.
    - LVG_target: `tf.Tensor` (Optional), Tensor containing the reconstruction target of the minibatch for the LVGs.
    - aeloss_fun: `function`, Function to compute reconstruction loss.
    - clust_weight: `float`, A float between 0 and 2 balancing clustering and reconstruction losses.
    
    Returns:
    ------------------------------------------------------------------
    - loss_value: `tf.Tensor`: The loss computed for the minibatch.
    - gradients: `a list of Tensors`: Gradients to update the model weights.
    """
    
    with tf.GradientTape() as tape:
        denoised_output, cluster_output = model(*input_)
        loss_value, aeloss = total_loss(target, denoised_output, target_p, cluster_output, 
                                LVG_target, aeloss_fun, clust_weight)
        
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def grad_reconstruction(model, input_, target, loss):
    """Function to compute gradient update for pretrained autoencoder only.
    
    
    Arguments:
    ------------------------------------------------------------------
    - model: `tensorflow.keras.Model`, The main CarDEC model.
    - input_: `list`, A list containing the input HVG expression tensor of the minibatch for the CarDEC model.
    - target: `tf.Tensor`, Tensor containing the reconstruction target of the minibatch for the HVGs.
    - loss: `function`, Function to compute reconstruction loss.
    
    Returns:
    ------------------------------------------------------------------
    - loss_value: `tf.Tensor`: The loss computed for the minibatch.
    - gradients: `a list of Tensors`: Gradients to update the model weights.
    """
    
    if type(input_) != tuple:
        input_ = (input_, )
        
    with tf.GradientTape() as tape:
        output = model(*input_)
        loss_value = loss(target, output)
        
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def total_loss(target, denoised_output, p, cluster_output_q, LVG_target = None, aeloss_fun = None, clust_weight = 1.):
    """Function to compute the loss for the main CarDEC model for a minibatch.
    
    
    Arguments:
    ------------------------------------------------------------------
    - target: `tf.Tensor`, Tensor containing the reconstruction target of the minibatch for the HVGs.
    - denoised_output: `dict`, Dictionary containing the output tensors from the CarDEC main model's forward pass.
    - p: `tf.Tensor`, Tensor of shape (n_obs, n_cluster) containing cluster membership probability targets for the minibatch.
    - cluster_output_q: `tf.Tensor`, Tensor of shape (n_obs, n_cluster) containing predicted cluster membership probabilities
    for each cell.
    - LVG_target: `tf.Tensor` (Optional), Tensor containing the reconstruction target of the minibatch for the LVGs.
    - aeloss_fun: `function`, Function to compute reconstruction loss.
    - clust_weight: `float`, A float between 0 and 2 balancing clustering and reconstruction losses.
    
    Returns:
    ------------------------------------------------------------------
    - net_loss: `tf.Tensor`, The loss computed for the minibatch.
    - aeloss: `tf.Tensor`, The reconstruction loss computed for the minibatch.
    """

    if aeloss_fun is not None:
        
        aeloss_HVG = aeloss_fun(target, denoised_output['HVG_denoised'])
        if LVG_target is not None:
            aeloss_LVG = aeloss_fun(LVG_target, denoised_output['LVG_denoised'])
            aeloss = 0.5*(aeloss_LVG + aeloss_HVG)
        else:
            aeloss = 1. * aeloss_HVG
    else:
        aeloss = 0.
    
    net_loss = clust_weight * tf.reduce_mean(KLD(p, cluster_output_q)) + (2. - clust_weight) * aeloss
    
    return net_loss, aeloss


def MSEloss(netinput, netoutput):
    """Function to compute the MSEloss for the reconstruction loss of a minibatch.
    
    
    Arguments:
    ------------------------------------------------------------------
    - netinput: `tf.Tensor`, Tensor containing the network reconstruction target of the minibatch for the cells.
    - netoutput: `tf.Tensor`, Tensor containing the reconstructed target of the minibatch for the cells.
    
    Returns:
    ------------------------------------------------------------------
    - mse_loss: `tf.Tensor`, The loss computed for the minibatch, averaged over genes and cells.
    """
    
    return tf.math.reduce_mean(MSE(netinput, netoutput))


def NBloss(count, output, eps = 1e-10, mean = True):
    """Function to compute the negative binomial reconstruction loss of a minibatch.
    
    
    Arguments:
    ------------------------------------------------------------------
    - count: `tf.Tensor`, Tensor containing the network reconstruction target of the minibatch for the cells (the original 
    counts).
    - output: `tf.Tensor`, Tensor containing the reconstructed target of the minibatch for the cells.
    - eps: `float`, A small number introduced for computational stability
    - mean: `bool`, If True, average negative binomial loss over genes and cells
    
    Returns:
    ------------------------------------------------------------------
    - nbloss: `tf.Tensor`, The loss computed for the minibatch. If mean was True, it has shape (n_obs, n_var). Otherwise, it has shape (1,).
    """
    
    count = tf.cast(count, tf.float32) #float32
    mu = tf.cast(output[0], tf.float32) #float32

    theta = tf.minimum(output[1], 1e6)

    t1 = tf.math.lgamma(theta + eps) + tf.math.lgamma(count + 1.0) - tf.math.lgamma(count + theta + eps)
    t2 = (theta + count) * tf.math.log(1.0 + (mu/(theta+eps))) + (count * (tf.math.log(theta + eps) - tf.math.log(mu + eps)))

    final = _nan2inf(t1 + t2)
    
    if mean:
        final = tf.reduce_sum(final)/final.shape[0]/final.shape[1]

    return final


def ZINBloss(count, output, eps = 1e-10):
    """Function to compute the negative binomial reconstruction loss of a minibatch.
    
    
    Arguments:
    ------------------------------------------------------------------
    - count: `tf.Tensor`, Tensor containing the network reconstruction target of the minibatch for the cells (the original counts).
    - output: `tf.Tensor`, Tensor containing the reconstructed target of the minibatch for the cells.
    - eps: `float`, A small number introduced for computational stability
    
    Returns:
    ------------------------------------------------------------------
    - zinbloss: `tf.Tensor`, The loss computed for the minibatch. Has shape (1,).
    """
    
    mu = output[0]
    theta = output[1]
    pi = output[2]
    
    NB = NBloss(count, output, eps = eps, mean = False) - tf.math.log(1.0 - pi + eps)
    
    count = tf.cast(count, tf.float32)
    mu = tf.cast(mu, tf.float32)
    
    theta = tf.math.minimum(theta, 1e6)
    
    zero_nb = tf.math.pow(theta/(theta + mu + eps), theta)
    zero_case = -tf.math.log(pi + ((1.0- pi) * zero_nb) + eps)
    final = tf.where(tf.less(count, 1e-8), zero_case, NB)
    
    final = tf.reduce_sum(final)/final.shape[0]/final.shape[1]
            
    return final


def _nan2inf(x):
    """Function to replace nan entries in a Tensor with infinities.
    
    
    Arguments:
    ------------------------------------------------------------------
    - x: `tf.Tensor`, Tensor of arbitrary shape.
    
    Returns:
    ------------------------------------------------------------------
    - x': `tf.Tensor`, Tensor x with nan entries replaced by infinity.
    """
    
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x) + np.inf, x)

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import exp as tf_exp, set_floatx
from time import time

import random
import numpy as np
from scipy.stats import zscore
import os



set_floatx('float32')


class count_model(Model):
    def __init__(self, dims, act = 'relu', random_seed = 201809, splitseed = 215, optimizer = Adam(),
             weights_dir = 'CarDEC Count Weights', n_features = 32, mode = 'HVG'):
        """ This class method initializes the count model.
        Arguments:
        ------------------------------------------------------------------
        - dims: `list`, the number of output features for each layer of the model. The length of the list determines the
        number of layers.
        - act: `str`, The activation function used for the intermediate layers of CarDEC, other than the bottleneck layer.
        - random_seed: `int`, The seed used for random weight intialization.
        - splitseed: `int`, The seed used to split cells between training and validation. Should be consistent between
        iterations to ensure the same cells are always used for validation.
        - optimizer: `tensorflow.python.keras.optimizer_v2`, An instance of a TensorFlow optimizer.
        - weights_dir: `str`, the path in which to save the weights of the CarDEC model.
        - n_features: `int`, the number of input features.
        - mode: `str`, String identifying whether HVGs or LVGs are being modeled.
        """
        
        super(count_model, self).__init__()

        tf.keras.backend.clear_session()
        
        self.mode = mode
        self.name_ = mode + " Count"
        
        if mode == 'HVG':
            self.embed_name = 'embedding'
        else:
            self.embed_name = 'LVG embedding'
        
        self.weights_dir = weights_dir
        
        self.dims = dims
        n_stacks = len(dims) - 1
        
        self.optimizer = optimizer
        self.random_seed = random_seed
        self.splitseed = splitseed
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        self.activation = act
        self.MeanAct = lambda x: tf.clip_by_value(tf_exp(x), 1e-5, 1e6)
        self.DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)
        
        model_layers = []
        for i in range(n_stacks - 1, 0, -1):
            model_layers.append(Dense(dims[i], kernel_initializer = "glorot_uniform", activation = self.activation
                                        , name='base%d' % (i-1)))
        self.base = Sequential(model_layers, name = 'base')

        self.mean_layer = Dense(dims[0], activation = self.MeanAct, name='mean')
        self.disp_layer = Dense(dims[0], activation = self.DispAct, name='dispersion')

        self.rescale = Lambda(lambda l: tf.matmul(tf.linalg.diag(l[0]), l[1]), name = 'sf scaling')
        
        build_dir(self.weights_dir)
        
        self.construct(n_features, self.name_)
        
    def call(self, x, s):
        """ This is the forward pass of the model.
        
        ***Inputs***
            - x: `tf.Tensor`, an input tensor of shape (b, p)
            - s: `tf.Tensor`, and input tensor of shape (b, ) containing the size factor for each cell
            
        ***Outputs***
            - mean: `tf.Tensor`, A (b, p_gene) tensor of negative binomial means for each cell, gene.
            - disp: `tf.Tensor`, A (b, p_gene) tensor of negative binomial dispersions for each cell, gene.
        """
        
        x = self.base(x)
        
        disp = self.disp_layer(x)
        mean = self.mean_layer(x)
        mean = self.rescale([s, mean])
                        
        return mean, disp
        
    def load_model(self, ):
        """ This class method can be used to load the model's weights."""
            
        tf.keras.backend.clear_session()
        
        self.load_weights(os.path.join(self.weights_dir, "countmodel_weights_" + self.name_)).expect_partial()
        
    def construct(self, n_features, name, summarize = False):
        """ This class method fully initalizes the TensorFlow model.
        Arguments:
        ------------------------------------------------------------------
        - n_features: `int`, the number of input features.
        - name: `str`, Model name (to distinguish HVG and LVG models).
        - summarize: `bool`, If True, then print a summary of the model architecture.
        """
        
        x = [tf.zeros(shape = (1, n_features), dtype='float32'), tf.ones(shape = (1,), dtype='float32')]
        out = self(*x)
        
        if summarize:
            print("----------Count Model " + name + " Architecture----------")
            self.summary()

            print("\n----------Base Sub-Architecture----------")
            self.base.summary()
        
    def denoise(self, adata, keep_dispersion = False, batch_size = 64):
        """ This class method can be used to denoise gene expression for each cell on the count scale.
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The annotated data matrix of shape (n_obs, n_vars). Rows correspond
        to cells and columns to genes.
        - keep_dispersion: `bool`, If True, also return the dispersion for each gene, cell (added as a layer to adata)/
        - batch_size: `int`, The batch size used for computing denoised expression.
        
        Returns:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The annotated data matrix of shape (n_obs, n_vars). Negative binomial means (and optionally 
        dispersions) added as layers.
        """
        
        input_ds = tupleloader(adata.obsm[self.embed_name], adata.obs['size factors'], batch_size = batch_size)
        
        if "denoised counts" not in list(adata.layers):
            adata.layers["denoised counts"] = np.zeros(adata.shape, dtype = 'float32')
        
        type_indices = adata.var['Variance Type'] == self.mode
        
        if not keep_dispersion:
            start = 0
            for x in input_ds:
                end = start + x[0].shape[0]
                adata.layers["denoised counts"][start:end, type_indices] = self(*x)[0].numpy()
                start = end
                
        else:
            if "dispersion" not in list(adata.layers):
                adata.layers["dispersion"] = np.zeros(adata.shape, dtype = 'float32')
                
            start = 0
            for x in input_ds:
                end = start + x[0].shape[0]
                batch_output = self(*x)
                adata.layers["denoised counts"][start:end, type_indices] = batch_output[0].numpy()
                adata.layers["dispersion"][start:end, type_indices] = batch_output[1].numpy()
                start = end
            
    def makegenerators(self, adata, val_split, batch_size, splitseed):
        """ This class method creates training and validation data generators for the current input data.
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond
        to cells and columns to genes.
        - val_split: `float`, The fraction of cells to be reserved for validation during this step.
        - batch_size: `int`, The batch size used for training the model.
        - splitseed: `int`, The seed used to split cells between training and validation. Should be consistent between
        iterations to ensure the same cells are always used for validation.
        
        Returns:
        ------------------------------------------------------------------
        - train_dataset: `tf.data.Dataset`, Dataset that returns training examples.
        - val_dataset: `tf.data.Dataset`, Dataset that returns validation examples.
        """
        
        return countloader(adata.obsm[self.embed_name], adata.X[:, adata.var['Variance Type'] == self.mode], adata.obs['size factors'], 
                           val_split, batch_size, splitseed)
    
    def train(self, adata, num_epochs = 2000, batch_size = 64, val_split = 0.1, lr = 1e-03, decay_factor = 1/3,
              patience_LR = 3, patience_ES = 9):
        """ This class method can be used to train the SAE.
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The annotated data matrix of shape (n_obs, n_vars). Rows correspond
        to cells and columns to genes.
        - num_epochs: `int`, The maximum number of epochs allowed to train the full model. In practice, the model will halt
        training long before hitting this limit.
        - batch_size: `int`, The batch size used for training the full model.
        - val_split: `float`, The fraction of cells to be reserved for validation during this step.
        - lr: `float`, The learning rate for training the full model.
        - decay_factor: `float`, The multiplicative factor by which to decay the learning rate when validation loss is not
        decreasing.
        - patience_LR: `int`, The number of epochs tolerated before decaying the learning rate during which the
        validation loss fails to decrease.
        - patience_ES: `int`, The number of epochs tolerated before stopping training during which the validation loss fails to
        decrease.
        """
        
        tf.keras.backend.clear_session()
                
        loss = NBloss
        
        dataset = self.makegenerators(adata, val_split = 0.1, batch_size = batch_size, splitseed = self.splitseed)
        
        counter_LR = 0
        counter_ES = 0
        best_loss = np.inf
        
        self.optimizer.lr = lr
        
        total_start = time()
        
        for epoch in range(num_epochs):
            epoch_start = time()
            
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_loss_avg_val = tf.keras.metrics.Mean()
            
            # Training loop - using batches of batch_size
            for x, target in dataset(val = False):
                loss_value, grads = grad(self, x, target, loss)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                epoch_loss_avg(loss_value)  # Add current batch loss
            
            # Validation Loop
            for x, target in dataset(val = True):
                output = self(*x)
                loss_value = loss(target, output)
                epoch_loss_avg_val(loss_value)
            
            current_loss_val = epoch_loss_avg_val.result()

            epoch_time = round(time() - epoch_start, 1)
            
            print("Epoch {:03d}: Training Loss: {:.3f}, Validation Loss: {:.3f}, Time: {:.1f} s".format(epoch, epoch_loss_avg.result().numpy(), epoch_loss_avg_val.result().numpy(), epoch_time))
            
            if(current_loss_val + 10**(-3) < best_loss):
                counter_LR = 0
                counter_ES = 0
                best_loss = current_loss_val
            else:
                counter_LR = counter_LR + 1
                counter_ES = counter_ES + 1

            if patience_ES <= counter_ES:
                break

            if patience_LR <= counter_LR:
                self.optimizer.lr = self.optimizer.lr * decay_factor
                counter_LR = 0
                print("\nDecaying Learning Rate to: " + str(self.optimizer.lr.numpy()))
                
            # End epoch
        
        total_time = round(time() - total_start, 2)
        
        if not os.path.isdir("./" + self.weights_dir):
            os.mkdir("./" + self.weights_dir)
        
        self.save_weights(os.path.join(self.weights_dir, "countmodel_weights_" + self.name_), save_format='tf')
                
        print('\nTraining Completed')
        print("Total training time: " + str(total_time) + " seconds")

from tensorflow import convert_to_tensor as tensor
from numpy import setdiff1d
from numpy.random import choice, seed

class batch_sampler(object):
    def __init__(self, array, val_frac, batch_size, splitseed):
        seed(splitseed)
        self.val_indices = choice(range(len(array)), round(val_frac * len(array)), False)
        self.train_indices = setdiff1d(range(len(array)), self.val_indices)
        self.batch_size = batch_size
        
    def __iter__(self):
        batch = []
        
        if self.val:
            for idx in self.val_indices:
                batch.append(idx)
                
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    
        else:
            train_idx = choice(self.train_indices, len(self.train_indices), False)
            
            for idx in train_idx:
                batch.append(idx)
                
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    
        if batch:
            yield batch
            
    def __call__(self, val):
        self.val = val
        return self
            
class simpleloader(object):
    def __init__(self, array, batch_size):
        self.array = array
        self.batch_size = batch_size
        
    def __iter__(self):
        batch = []
        
        for idx in range(len(self.array)):
            batch.append(idx)
            
            if len(batch) == self.batch_size:
                yield tensor(self.array[batch].copy())
                batch = []
                
        if batch:
            yield self.array[batch].copy()
            
class tupleloader(object):
    def __init__(self, *arrays, batch_size):
        self.arrays = arrays
        self.batch_size = batch_size
        
    def __iter__(self):
        batch = []
        
        for idx in range(len(self.arrays[0])):
            batch.append(idx)
            
            if len(batch) == self.batch_size:
                yield [tensor(arr[batch].copy()) for arr in self.arrays]
                batch = []
                
        if batch:
            yield [tensor(arr[batch].copy()) for arr in self.arrays]
            
class aeloader(object):
    def __init__(self, *arrays, val_frac, batch_size, splitseed):
        self.arrays = arrays
        self.batch_size = batch_size
        self.sampler = batch_sampler(arrays[0], val_frac, batch_size, splitseed)
        
    def __iter__(self):
        for idxs in self.sampler(self.val):
            yield [tensor(arr[idxs].copy()) for arr in self.arrays]
            
    def __call__(self, val):
        self.val = val
        return self
            
class countloader(object):
    def __init__(self, embedding, target, sizefactor, val_frac, batch_size, splitseed):
        self.sampler = batch_sampler(embedding, val_frac, batch_size, splitseed)
        self.embedding = embedding
        self.target = target
        self.sizefactor = sizefactor
        
    def __iter__(self):
        for idxs in self.sampler(self.val):
            yield (tensor(self.embedding[idxs].copy()), tensor(self.sizefactor[idxs].copy())), tensor(self.target[idxs].copy())
            
    def __call__(self, val):
        self.val = val
        return self
            
class dataloader(object):
    def __init__(self, hvg_input, hvg_target, lvg_input = None, lvg_target = None, val_frac = 0.1, batch_size = 128, splitseed = 0):
        self.sampler = batch_sampler(hvg_input, val_frac, batch_size, splitseed)
        self.hvg_input = hvg_input
        self.hvg_target = hvg_target
        self.lvg_input = lvg_input
        self.lvg_target = lvg_target
        
    def __iter__(self):
        for idxs in self.sampler(self.val):
            hvg_input = tensor(self.hvg_input[idxs].copy())
            hvg_target = tensor(self.hvg_target[idxs].copy())
            p_target = tensor(self.p_target[idxs].copy())
            
            if (self.lvg_input is not None) and (self.lvg_target is not None):
                lvg_input = tensor(self.lvg_input[idxs].copy())
                lvg_target = tensor(self.lvg_target[idxs].copy())
            else:
                lvg_input = None
                lvg_target = None
                
            yield [hvg_input, lvg_input], hvg_target, lvg_target, p_target
            
    def __call__(self, val):
        self.val = val
        return self
    
    def update_p(self, new_p_target):
        self.p_target = new_p_target

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import set_floatx
from time import time

import random
import numpy as np
from scipy.stats import zscore
import os


set_floatx('float32')


class SAE(Model):
    def __init__(self, dims, act = 'relu', actincenter = "tanh", 
                 random_seed = 201809, splitseed = 215, init = "glorot_uniform", optimizer = Adam(),
                 weights_dir = 'CarDEC Weights'):
        """ This class method initializes the SAE model.
        Arguments:
        ------------------------------------------------------------------
        - dims: `list`, the number of output features for each layer of the HVG encoder. The length of the list determines the number of layers.
        - act: `str`, The activation function used for the intermediate layers of CarDEC, other than the bottleneck layer.
        - actincenter: `str`, The activation function used for the bottleneck layer of CarDEC.
        - random_seed: `int`, The seed used for random weight intialization.
        - splitseed: `int`, The seed used to split cells between training and validation. Should be consistent between iterations to ensure the same cells are always used for validation.
        - init: `str`, The weight initialization strategy for the autoencoder.
        - optimizer: `tensorflow.python.keras.optimizer_v2`, An instance of a TensorFlow optimizer.
        - weights_dir: `str`, the path in which to save the weights of the CarDEC model.
        """
        
        super(SAE, self).__init__()
        
        tf.keras.backend.clear_session()
        
        self.weights_dir = weights_dir
        
        self.dims = dims
        self.n_stacks = len(dims) - 1
        self.init = init
        self.optimizer = optimizer
        self.random_seed = random_seed
        self.splitseed = splitseed
        
        self.activation = act
        self.actincenter = actincenter #hidden layer activation function
        
        #set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
            
        encoder_layers = []
        for i in range(self.n_stacks-1):
            encoder_layers.append(Dense(self.dims[i + 1], kernel_initializer = self.init, activation = self.activation, name='encoder_%d' % i))
                
        encoder_layers.append(Dense(self.dims[-1], kernel_initializer=self.init, activation=self.actincenter, name='embedding'))
        self.encoder = Sequential(encoder_layers, name = 'encoder')

        decoder_layers = []
        for i in range(self.n_stacks - 1, 0, -1):
            decoder_layers.append(Dense(self.dims[i], kernel_initializer = self.init, activation = self.activation
                                        , name = 'decoder%d' % (i-1)))
            
        decoder_layers.append(Dense(self.dims[0], activation = 'linear', name='output'))
        
        self.decoder = Sequential(decoder_layers, name = 'decoder')
        
        self.construct()

    def call(self, x):
        """ This is the forward pass of the model.
        
        
        ***Inputs***
            - x: `tf.Tensor`, an input tensor of shape (n_obs, p_HVG).
            
        ***Outputs***
            - output: `tf.Tensor`, A (n_obs, p_HVG) tensor of denoised HVG expression.
        """
        
        c = self.encoder(x)

        output = self.decoder(c)
                    
        return output
    
    def load_encoder(self, random_seed = 2312):
        """ This class method can be used to load the encoder weights, while randomly reinitializing the decoder weights.
        Arguments:
        ------------------------------------------------------------------
        - random_seed: `int`, Seed for reinitializing the decoder.
        """
        
        tf.keras.backend.clear_session()
        
        #set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
     
        self.encoder.load_weights("./" + self.weights_dir + "/pretrained_encoder_weights").expect_partial()
        
        decoder_layers = []
        for i in range(self.n_stacks - 1, 0, -1):
            decoder_layers.append(Dense(self.dims[i], kernel_initializer = self.init, activation = self.activation
                                        , name='decoder%d' % (i-1)))
        self.decoder_base = Sequential(decoder_layers, name = 'decoderbase')
        
        self.output_layer = Dense(self.dims[0], activation = 'linear', name='output')
            
        self.construct(summarize = False)
        
    def load_autoencoder(self, ):
        """ This class method can be used to load the full model's weights."""
        
        tf.keras.backend.clear_session()
        
        self.load_weights("./" + self.weights_dir + "/pretrained_autoencoder_weights").expect_partial()
        
    def construct(self, summarize = False):
        """ This class method fully initalizes the TensorFlow model.
        Arguments:
        ------------------------------------------------------------------
        - summarize: `bool`, If True, then print a summary of the model architecture.
        """
        
        x = tf.zeros(shape = (1, self.dims[0]), dtype=float)
        out = self(x)
        
        if summarize:
            print("----------Autoencoder Architecture----------")
            self.summary()

            print("\n----------Encoder Sub-Architecture----------")
            self.encoder.summary()

            print("\n----------Base Decoder Sub-Architecture----------")
            self.decoder.summary()
        
    def denoise(self, adata, batch_size = 64):
        """ This class method can be used to denoise gene expression for each cell.
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The annotated data matrix of shape (n_obs, n_vars).
        - batch_size: `int`, The batch size used for computing denoised expression.
        
        Returns:
        ------------------------------------------------------------------
        - output: `np.ndarray`, Numpy array of denoised expression of shape (n_obs, n_vars)
        """
        
        input_ds = simpleloader(adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG'], batch_size)
        
        output = np.zeros((adata.shape[0], self.dims[0]), dtype = 'float32')
        start = 0
        
        for x in input_ds:
            end = start + x.shape[0]
            output[start:end] = self(x).numpy()
            start = end
        
        return output
        
    def embed(self, adata, batch_size = 64):
        """ This class method can be used to compute the low-dimension embedding for HVG features. 
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The annotated data matrix of shape (n_obs, n_vars).
        - batch_size: `int`, The batch size for filling the array of low dimension embeddings.
        
        Returns:
        ------------------------------------------------------------------
        - embedding: `np.ndarray`, Array of shape (n_obs, n_vars) containing the cell HVG embeddings.
        """
        
        input_ds = simpleloader(adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG'], batch_size)
        
        embedding = np.zeros((adata.shape[0], self.dims[-1]), dtype = 'float32')
        
        start = 0
        for x in input_ds:
            end = start + x.shape[0]
            embedding[start:end] = self.encoder(x).numpy()
            start = end
            
        return embedding
    
    def makegenerators(self, adata, val_split, batch_size, splitseed):
        """ This class method creates training and validation data generators for the current input data.
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars).
        - val_split: `float`, The fraction of cells to be reserved for validation during this step.
        - batch_size: `int`, The batch size used for training the model.
        - splitseed: `int`, The seed used to split cells between training and validation.
        
        Returns:
        ------------------------------------------------------------------
        - train_dataset: `tf.data.Dataset`, Dataset that returns training examples.
        - val_dataset: `tf.data.Dataset`, Dataset that returns validation examples.
        """
        
        return aeloader(adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG'], adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG'], val_frac = val_split, batch_size = batch_size, splitseed = splitseed)
    
    def train(self, adata, num_epochs = 2000, batch_size = 64, val_split = 0.1, lr = 1e-03, decay_factor = 1/3,
              patience_LR = 3, patience_ES = 9, save_fullmodel = True):
        """ This class method can be used to train the SAE.
        
        
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, The annotated data matrix of shape (n_obs, n_vars).
        - num_epochs: `int`, The maximum number of epochs allowed to train the full model. In practice, the model will halt training long before hitting this limit.
        - batch_size: `int`, The batch size used for training the full model.
        - val_split: `float`, The fraction of cells to be reserved for validation during this step.
        - lr: `float`, The learning rate for training the full model.
        - decay_factor: `float`, The multiplicative factor by which to decay the learning rate when validation loss is not decreasing.
        - patience_LR: `int`, The number of epochs tolerated before decaying the learning rate during which the validation loss fails to decrease.
        - patience_ES: `int`, The number of epochs tolerated before stopping training during which the validation loss fails to decrease.
        - save_fullmodel: `bool`, If True, save the full model's weights, not just the encoder.
        """
        
        tf.keras.backend.clear_session()
        
        dataset = self.makegenerators(adata, val_split = 0.1, batch_size = batch_size, splitseed = self.splitseed)
        
        counter_LR = 0
        counter_ES = 0
        best_loss = np.inf
        
        self.optimizer.lr = lr
        
        total_start = time()
        for epoch in range(num_epochs):
            epoch_start = time()
            
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_loss_avg_val = tf.keras.metrics.Mean()
            
            # Training loop - using batches of batch_size
            for x, target in dataset(val = False):
                loss_value, grads = grad(self, x, target, MSEloss)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                epoch_loss_avg(loss_value)  # Add current batch loss
            
            # Validation Loop
            for x, target in dataset(val = True):
                output = self(x)
                loss_value = MSEloss(target, output)
                epoch_loss_avg_val(loss_value)
            
            current_loss_val = epoch_loss_avg_val.result()

            epoch_time = round(time() - epoch_start, 1)
            
            print("Epoch {:03d}: Training Loss: {:.3f}, Validation Loss: {:.3f}, Time: {:.1f} s".format(epoch, epoch_loss_avg.result().numpy(), epoch_loss_avg_val.result().numpy(), epoch_time))
            
            if(current_loss_val + 10**(-3) < best_loss):
                counter_LR = 0
                counter_ES = 0
                best_loss = current_loss_val
            else:
                counter_LR = counter_LR + 1
                counter_ES = counter_ES + 1

            if patience_ES <= counter_ES:
                break

            if patience_LR <= counter_LR:
                self.optimizer.lr = self.optimizer.lr * decay_factor
                counter_LR = 0
                print("\nDecaying Learning Rate to: " + str(self.optimizer.lr.numpy()))
                
            # End epoch
        
        total_time = round(time() - total_start, 2)
        
        if not os.path.isdir("./" + self.weights_dir):
            os.mkdir("./" + self.weights_dir)
        
        self.save_weights("./" + self.weights_dir + "/pretrained_autoencoder_weights", save_format='tf')
        self.encoder.save_weights("./" + self.weights_dir + "/pretrained_encoder_weights", save_format='tf')
        
        print('\nTraining Completed')
        print("Total training time: " + str(total_time) + " seconds")


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from pandas import DataFrame

import os

class CarDEC_API:
    def __init__(self, adata, preprocess=True, weights_dir = "CarDEC Weights", batch_key = None, n_high_var = 2000, LVG = True,
                     normalize_samples = True, log_normalize = True, normalize_features = True):
        """ Main CarDEC API the user can use to conduct batch correction and denoising experiments.
        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to genes.
        - preprocess: `bool`, If True, then preprocess the data.
        - weights_dir: `str`, the path in which to save the weights of the CarDEC model.
        - batch_key: `str`, string specifying the name of the column in the observation dataframe which identifies the batch of each cell. If this is left as None, then all cells are assumed to be from one batch.
        - n_high_var: `int`, integer specifying the number of genes to be idntified as highly variable. E.g. if n_high_var = 2000, then the 2000 genes with the highest variance are designated as highly variable.
        - LVG: `bool`, If True, also model LVGs. Otherwise, only model HVGs.
        - normalize_samples: `bool`, If True, normalize expression of each gene in each cell by the sum of expression counts in that cell.
        - log_normalize: `bool`, If True, log transform expression. I.e., compute log(expression + 1) for each gene, cell expression count.
        - normalize_features: `bool`, If True, z-score normalize each gene's expression.
        """
    
        if n_high_var is None:
            n_high_var = None
            LVG = False

        self.weights_dir = weights_dir
        self.LVG = LVG

        self.norm_args = (batch_key, n_high_var, LVG, normalize_samples, log_normalize, normalize_features)

        if preprocess:
            self.dataset = normalize_scanpy(adata, *self.norm_args)
        else:
            assert 'Variance Type' in adata.var.keys()
            assert 'normalized input' in adata.layers
            self.dataset = adata

        self.loaded = False
        self.count_loaded = False

    def build_model(self, load_fullmodel = True, dims = [128, 32], LVG_dims = [128, 32], tol = 0.005, n_clusters = None, 
                    random_seed = 201809, louvain_seed = 0, n_neighbors = 15, pretrain_epochs = 2000, batch_size_pretrain = 64,
                    act = 'relu', actincenter = "tanh", ae_lr = 1e-04, ae_decay_factor = 1/3, ae_patience_LR = 3, 
                    ae_patience_ES = 9, clust_weight = 1., load_encoder_weights = True):
        """ Initializes the main CarDEC model.
        Arguments:
        ------------------------------------------------------------------
        - load_fullmodel: `bool`, If True, the API will try to load the weights for the full model from the weight directory.
        - dims: `list`, the number of output features for each layer of the HVG encoder. The length of the list determines the number of layers.
        - LVG_dims: `list`, the number of output features for each layer of the LVG encoder. The length of the list determines the number of layers.
        - tol: `float`, stop criterion, clustering procedure will be stopped when the difference ratio between the current iteration and last iteration larger than tol.
        - n_clusters: `int`, The number of clusters into which cells will be grouped.
        - random_seed: `int`, The seed used for random weight intialization.
        - louvain_seed: `int`, The seed used for louvain clustering intialization.
        - n_neighbors: `int`, The number of neighbors used for building the graph needed for louvain clustering.
        - pretrain_epochs: `int`, The maximum number of epochs for pretraining the HVG autoencoder. In practice, early stopping criteria should stop training much earlier.
        - batch_size_pretrain: `int`, The batch size used for pretraining the HVG autoencoder.
        - act: `str`, The activation function used for the intermediate layers of CarDEC, other than the bottleneck layer.
        - actincenter: `str`, The activation function used for the bottleneck layer of CarDEC.
        - ae_lr: `float`, The learning rate for pretraining the HVG autoencoder.
        - ae_decay_factor: `float`, The multiplicative factor by which to decay the learning rate when validation loss is not decreasing.
        - ae_patience_LR: `int`, the number of epochs which the validation loss is allowed to increase before learning rate is decayed when pretraining the autoencoder.
        - ae_patience_ES: `int`, the number of epochs which the validation loss is allowed to increase before training is halted when pretraining the autoencoder.
        - clust_weight: `float`, a number between 0 and 2 qhich balances the clustering and reconstruction losses.
        - load_encoder_weights: `bool`, If True, the API will try to load the weights for the HVG encoder from the weight directory.
        """
        
        assert n_clusters is not None
        
        if 'normalized input' not in list(self.dataset.layers):
            self.dataset = normalize_scanpy(self.dataset, *self.norm_args)
        
        p = sum(self.dataset.var["Variance Type"] == 'HVG')
        self.dims = [p] + dims
        
        if self.LVG:
            LVG_p = sum(self.dataset.var["Variance Type"] == 'LVG')
            self.LVG_dims = [LVG_p] + LVG_dims
        else:
            self.LVG_dims = None
        
        self.load_fullmodel = load_fullmodel
        self.weights_exist = os.path.isfile("./" + self.weights_dir + "/tuned_CarDECweights.index")
        
        set_centroids = not (self.load_fullmodel and self.weights_exist)
        
        self.model = CarDEC_Model(self.dataset, self.dims, self.LVG_dims, tol, n_clusters, random_seed, louvain_seed, 
                                  n_neighbors, pretrain_epochs, batch_size_pretrain, ae_decay_factor, 
                                  ae_patience_LR, ae_patience_ES, act, actincenter, ae_lr, 
                                  clust_weight, load_encoder_weights, set_centroids, self.weights_dir)
        
    def make_inference(self, batch_size = 64, val_split = 0.1, lr = 1e-04, decay_factor = 1/3,
                       iteration_patience_LR = 3, iteration_patience_ES = 6, maxiter = 1e3, epochs_fit = 1, 
                       optimizer = Adam(), printperiter = None, denoise_all = True, denoise_list = None):
        """ This class method makes inference on the data (batch correction + denoising) with the main CarDEC model
        Arguments:
        ------------------------------------------------------------------
        - batch_size: `int`, The batch size used for training the full model.
        - val_split: `float`, The fraction of cells to be reserved for validation during this step.
        - lr: `float`, The learning rate for training the full model.
        - decay_factor: `float`, The multiplicative factor by which to decay the learning rate when validation loss is not decreasing.
        - iteration_patience_LR: `int`, The number of iterations tolerated before decaying the learning rate during which the number of cells that change assignment is less than tol.
        - iteration_patience_ES: `int`, The number of iterations tolerated before stopping training during which the number of cells that change assignment is less than tol.
        - maxiter: `int`, The maximum number of iterations allowed to train the full model. In practice, the model will halt training long before hitting this limit.
        - epochs_fit: `int`, The number of epochs during which to fine-tune weights, before updating the target distribution.
        - optimizer: `tensorflow.python.keras.optimizer_v2`, An instance of a TensorFlow optimizer.
        - printperiter: `int`, Optional integer argument. If specified, denoised values will be returned every printperiter epochs, so that the user can evaluate the progress of denoising as training continues.
        - denoise_all: `bool`, If True, then denoised expression values are provided for all cells.
        - denoise_list: `list`, An optional list of cell names (as strings). If provided, denoised values will be computed only for cells in this list.
        Returns:
        ------------------------------------------------------------------
        - denoised: `pd.DataFrame`, (Optional) If denoise_list was specified, then this will be an array of denoised expression provided only for listed cells. If denoise_all was instead specified as True, then denoised expression for all cells will be added as a layer to adata.
        """

        if denoise_list is not None:
            denoise_all = False
            
        if not self.loaded:
            if self.load_fullmodel and self.weights_exist:
                self.dataset = self.model.reload_model(self.dataset, batch_size, denoise_all)

            elif not self.weights_exist:
                print("CarDEC Model Weights not detected. Training full model.\n")
                self.dataset = self.model.train(self.dataset, batch_size, val_split, lr, decay_factor,
                               iteration_patience_LR, iteration_patience_ES, maxiter,
                               epochs_fit, optimizer, printperiter, denoise_all)

            else:
                print("Training full model.\n")
                self.dataset = self.model.train(self.dataset, batch_size, val_split, lr, decay_factor, 
                                                iteration_patience_LR, iteration_patience_ES, 
                                                maxiter, epochs_fit, optimizer, printperiter, denoise_all)
            
            
            self.loaded = True
            
        elif denoise_all:
            self.dataset = self.model.make_outputs(self.dataset, batch_size, True)
            
        if denoise_list is not None:
            denoise_list = list(denoise_list)
            indices = [x in denoise_list for x in self.dataset.obs.index]
            denoised = DataFrame(np.zeros((len(denoise_list), self.dataset.shape[1]), dtype = 'float32'))
            denoised.index = self.dataset.obs.index[indices]
            denoised.columns = self.dataset.var.index
            
            
            if self.LVG:
                hvg_ds = tf.data.Dataset.from_tensor_slices(self.dataset.obsm["embedding"][indices])
                lvg_ds = tf.data.Dataset.from_tensor_slices(self.dataset.obsm["LVG embedding"][indices])
            
                input_ds = tf.data.Dataset.zip((hvg_ds, lvg_ds))
                input_ds = input_ds.batch(batch_size)

                start = 0     
                for x in input_ds:
                    denoised_batch = {'HVG_denoised': self.model.decoder(x[0]), 'LVG_denoised': self.model.decoderLVG(x[1])}
                    q_batch = self.model.clustering_layer(x[0])
                    end = start + q_batch.shape[0]

                    denoised.iloc[start:end, np.where(self.dataset.var['Variance Type'] == 'HVG')[0]] = denoised_batch['HVG_denoised'].numpy()
                    denoised.iloc[start:end, np.where(self.dataset.var['Variance Type'] == 'LVG')[0]] = denoised_batch['LVG_denoised'].numpy()

                    start = end

            else:
                input_ds = tf.data.Dataset.from_tensor_slices(self.dataset.obsm["embedding"])

                input_ds = input_ds.batch(batch_size)

                start = 0

                for x in input_ds:
                    denoised_batch = {'HVG_denoised': self.model.decoder(x)}
                    q_batch = self.model.clustering_layer(x)
                    end = start + q_batch.shape[0]

                    denoised.iloc[start:end] = denoised_batch['HVG_denoised'].numpy()

                    start = end
            
            return denoised
            
        print(" ")
            
    def model_counts(self, load_weights = True, act = 'relu', random_seed = 201809,
                     optimizer = Adam(), keep_dispersion = False, num_epochs = 2000, batch_size_count = 64,
                     val_split = 0.1, lr = 1e-03, decay_factor = 1/3, patience_LR = 3, patience_ES = 9, 
                     denoise_all = True, denoise_list = None):
        """ This class method makes inference on the data on the count scale.
        Arguments:
        ------------------------------------------------------------------
        - load_weights: `bool`, If true, the API will attempt to load the weights for the count model.
        - act: `str`, A string specifying the activation function for intermediate layers of the count models.
        - random_seed: `int`, A seed used for weight initialization.
        - optimizer: `tensorflow.python.keras.optimizer_v2`, An instance of a TensorFlow optimizer.
    - keep_dispersion: `bool`, If True, the gene, cell dispersions will be returned as well.
        - num_epochs: `int`, The maximum number of epochs allowed to train each count model. In practice, the model will halt
        training long before hitting this limit.
        - batch_size_count: `int`, The batch size used for training the count models.
        - val_split: `float`, The fraction of cells to be reserved for validation during this step.
        - lr: `float`, The learning rate for training the count models.
        - decay_factor: `float`, The multiplicative factor by which to decay the learning rate when validation loss is not decreasing.
        - patience_LR: `int`, The number of epochs tolerated before decaying the learning rate during which the validation loss does not decrease.
        - patience_ES: `int`, The number of iterations tolerated before stopping training during which the validation loss does not decrease.
        - denoise_all: `bool`, If True, then denoised expression values are provided for all cells.
        - denoise_list: `list`, An optional list of cell names (as strings). If provided, denoised values will be computed only for cells in this list.
        Returns:
        ------------------------------------------------------------------
        - denoised: `pd.DataFrame`, (Optional) If denoise_list was specified, then this will be an array of denoised expression on the count scale provided only for listed cells. If denoise_all was instead specified as True, then denoised expression for all cells will be added as a layer to adata.
        - denoised_dispersion: `pd.DataFrame`, (Optional) If denoise_list was specified and "keep_dispersion" was set to True, then this will be an array of dispersions from the fitted negative binomial model provided only for listed cells. If denoise_all was instead specified as False, but "keep_dispersion" was still True then dispersions for all cells will be added as a layer to adata.
        """
    
        if denoise_list is not None:
            denoise_all = False
        
        if not self.count_loaded:
            weights_dir = os.path.join(self.weights_dir, 'count weights')
            weight_files_exist = os.path.isfile(weights_dir + "/countmodel_weights_HVG Count.index")
            if self.LVG:
                weight_files_exist = weight_files_exist and os.path.isfile(weights_dir + "/countmodel_weights_LVG Count.index")

            init_args = (act, random_seed, self.model.splitseed, optimizer, weights_dir)
            train_args = (num_epochs, batch_size_count, val_split, lr, decay_factor, patience_LR, patience_ES)

            self.nbmodel = count_model(self.dims, *init_args, n_features = self.dims[-1], mode = 'HVG')

            if load_weights and weight_files_exist:
                print("Weight files for count models detected, loading weights.")
                self.nbmodel.load_model()

            elif load_weights:
                print("Weight files for count models not detected. Training HVG count model.\n")
                self.nbmodel.train(self.dataset, *train_args)

            else:
                print("Training HVG count model.\n")
                self.nbmodel.train(self.dataset, *train_args)

            if self.LVG:
                self.nbmodel_lvg = count_model(self.LVG_dims, *init_args, 
                    n_features = self.dims[-1] + self.LVG_dims[-1], mode = 'LVG')

                if load_weights and weight_files_exist:
                    self.nbmodel_lvg.load_model()
                    print("Count model weights loaded successfully.")

                elif load_weights:
                    print("\n \n \n")
                    print("Training LVG count model.\n")
                    self.nbmodel_lvg.train(self.dataset, *train_args)

                else:
                    print("\n \n \n")
                    print("Training LVG count model.\n")
                    self.nbmodel_lvg.train(self.dataset, *train_args)
            
            self.count_loaded = True
            
        if denoise_all:
            self.nbmodel.denoise(self.dataset, keep_dispersion, batch_size_count)
            if self.LVG:
                self.nbmodel_lvg.denoise(self.dataset, keep_dispersion, batch_size_count)
                
        elif denoise_list is not None:
            denoise_list = list(denoise_list)
            indices = [x in denoise_list for x in self.dataset.obs.index]
            denoised = DataFrame(np.zeros((len(denoise_list), self.dataset.shape[1]), dtype = 'float32'))
            denoised.index = self.dataset.obs.index[indices]
            denoised.columns = self.dataset.var.index
            if keep_dispersion:
                denoised_dispersion = DataFrame(np.zeros((len(denoise_list), self.dataset.shape[1]), dtype = 'float32'))
                denoised_dispersion.index = self.dataset.obs.index[indices]
                denoised_dispersion.columns = self.dataset.var.index
            
            input_ds_embed = tf.data.Dataset.from_tensor_slices(self.dataset.obsm['embedding'][indices])
            input_ds_sf = tf.data.Dataset.from_tensor_slices(self.dataset.obs['size factors'][indices])
            input_ds = tf.data.Dataset.zip((input_ds_embed, input_ds_sf))
            input_ds = input_ds.batch(batch_size_count)

            type_indices = np.where(self.dataset.var['Variance Type'] == 'HVG')[0]

            if not keep_dispersion:
                start = 0
                for x in input_ds:
                    end = start + x[0].shape[0]
                    denoised.iloc[start:end, type_indices] = self.nbmodel(*x)[0].numpy()
                    start = end

            else:
                start = 0
                for x in input_ds:
                    end = start + x[0].shape[0]
                    batch_output = self.nbmodel(*x)
                    denoised.iloc[start:end, type_indices] = batch_output[0].numpy()
                    denoised_dispersion.iloc[start:end, type_indices] = batch_output[1].numpy()
                    start = end
            
            if self.LVG:
                input_ds_embed = tf.data.Dataset.from_tensor_slices(self.dataset.obsm['LVG embedding'][indices])
                input_ds_sf = tf.data.Dataset.from_tensor_slices(self.dataset.obs['size factors'][indices])
                input_ds = tf.data.Dataset.zip((input_ds_embed, input_ds_sf))
                input_ds = input_ds.batch(batch_size_count)

                type_indices = np.where(self.dataset.var['Variance Type'] == 'LVG')[0]

                if not keep_dispersion:
                    start = 0
                    for x in input_ds:
                        end = start + x[0].shape[0]
                        denoised.iloc[start:end, type_indices] = self.nbmodel_lvg(*x)[0].numpy()
                        start = end

                else:
                    start = 0
                    for x in input_ds:
                        end = start + x[0].shape[0]
                        batch_output = self.nbmodel_lvg(*x)
                        denoised.iloc[start:end, type_indices] = batch_output[0].numpy()
                        denoised_dispersion.iloc[start:end, type_indices] = batch_output[1].numpy()
                        start = end
                        
            if not keep_dispersion:
                return denoised
            else:
                return denoised, denoised_dispersion


import sklearn.metrics as metrics

def purity_score(y_true, y_pred):
    """A function to compute cluster purity"""
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
def convert_string_to_encoding(string, vector_key):
    """A function to convert a string to a numeric encoding"""
    return np.argwhere(vector_key == string)[0][0]

def convert_vector_to_encoding(vector):
    """A function to convert a vector of strings to a dense numeric encoding"""
    vector_key = np.unique(vector)
    vector_strings = list(vector)
    vector_num = [convert_string_to_encoding(string, vector_key) for string in vector_strings]
    
    return vector_num

def build_dir(dir_path):
    subdirs = [dir_path]
    substring = dir_path

    while substring != '':
        splt_dir = os.path.split(substring)
        substring = splt_dir[0]
        subdirs.append(substring)
        
    subdirs.pop()
    subdirs = [x for x in subdirs if os.path.basename(x) != '..']

    n = len(subdirs)
    subdirs = [subdirs[n - 1 - x] for x in range(n)]
    
    for dir_ in subdirs:
        if not os.path.isdir(dir_):
            os.mkdir(dir_)
    
def find_resolution(adata_, n_clusters, random = 0): 
    adata = adata_.copy()
    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]
    
    while obtained_clusters != n_clusters and iteration < 12:
        current_res = sum(resolutions)/2
        sc.tl.louvain(adata, resolution = current_res, random_state = random)
        labels = adata.obs['louvain']
        obtained_clusters = len(np.unique(labels))
        
        if obtained_clusters < n_clusters:
            resolutions[0] = current_res
        else:
            resolutions[1] = current_res
        
        iteration = iteration + 1
        
    return current_res




import pandas as pd
def load_data(X_in,true_clusters_in):
    from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
    scalerX = StandardScaler()
    scalerX.fit(X_in)
    x = scalerX.transform(X_in)

    y = pd.factorize(true_clusters_in)[0].astype(np.int32)
    print(y.shape)

    print('samples', x.shape, y.shape)
    return x, y

from torch.utils.data import Dataset
class inDataset(Dataset):
    def __init__(self, X_in,true_clusters_in):
        self.x, self.y = load_data(X_in,true_clusters_in)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))

def fit_cardec(X_in,true_clusters_in, data_dir='/athena/listonlab/store/amb2022/PCMF/',dataName="MNIST", n_top_genes_options = [2], n_neighbors_options = [5, 10, 15, 20, 25]):
    ''' dataName="MNIST", "Penguins" ''' 

    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    import numpy as np
    from pandas import DataFrame

    import scanpy as sc
    import time
    import os
    import numpy as np
    import pandas as pd
    from anndata import AnnData
    from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
    tic = time.time()
    os.chdir(data_dir)
    dataset = inDataset(X_in,true_clusters_in)
    scaler = MinMaxScaler()
    dataset.x = scaler.fit_transform(dataset.x)
    adata = AnnData(dataset.x)

    # n_top_genes_options = [2]
    # n_neighbors_options = [5, 10, 15, 20, 25]
    idx = 0
    accuracies = []
    for n_neighbors in n_neighbors_options:
        for n_top_genes in n_top_genes_options:
            from sklearn.preprocessing import MinMaxScaler
            sc.pp.pca(adata)
            sc.pp.neighbors(adata,n_neighbors=n_neighbors)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata) # ,n_top_genes=n_top_genes
            # Initialize and pretrain weights
            CarDEC = CarDEC_API(adata, weights_dir = "./data/"+dataName+"_Weights_NN"+str(n_neighbors)+'_NG'+str(n_top_genes), preprocess=True, normalize_samples=False, log_normalize=False, normalize_features=False, batch_key = None, LVG = True, n_high_var = 1)
            # Build model
            grad = grad_reconstruction
            CarDEC.build_model(n_clusters = len(np.unique(dataset.y)))
            # # Train CarDEC model using autodifferentiation and save weights
            grad = grad_MainModel
            CarDEC.make_inference()

            temporary = AnnData(CarDEC.dataset.obsm['embedding'])
            temporary.obs = CarDEC.dataset.obs
            sc.tl.pca(temporary, svd_solver='arpack')
            sc.pp.neighbors(temporary, n_neighbors = n_neighbors)
            y_pred = np.argmax(CarDEC.dataset.obsm['cluster memberships'],axis=1)

            # Calculate accuracy
            conf_mat_ord = confusion_matrix_ordered(dataset.y, y_pred)
            acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
            print('IDX:',idx, 'Accuracy:', acc, 'n_neighbors:',n_neighbors, 'n_top_genes:',n_top_genes)
            idx = idx+1

            accuracies.append([idx, acc, n_neighbors, n_top_genes])
    toc = time.time() - tic
    print('Time elapsed:',toc)
    return accuracies, toc, 'IDX: '+str(idx)+' Accuracy: '+str(acc)+' Batch size: '+str(batch_size)+' n_neighbors: '+str(n_neighbors)+' n_top_genes: '+str(n_top_genes)

##################################################################
##################################################################
#### END CARDEC ####
##################################################################
##################################################################


##################################################################
##################################################################
#### BEGIN DEC ####
##################################################################
##################################################################

# import tensorflow
import sys
import numpy as np
import keras.backend as K
from keras.initializers import RandomNormal
# from keras.engine.topology import Layer, InputSpec
from tensorflow.keras.layers import Layer, InputSpec

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input

try:
    from tensorflow.keras.optimizers.legacy import SGD
except:
    try:
        from keras.optimizers import SGD
        # from tensorflow.keras.optimizers import SGD
    except:
        from keras.optimizers import SGD

from sklearn.preprocessing import normalize
from keras.callbacks import LearningRateScheduler
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
if (sys.version[0] == 2):
    import cPickle as pickle
else:
    import pickle
    import numpy as np

class ClusteringLayer(Layer):
    '''
    Clustering layer which converts latent space Z of input layer
    into a probability vector for each cluster defined by its centre in
    Z-space. Use Kullback-Leibler divergence as loss, with a probability
    target distribution.
    # Arguments
        output_dim: int > 0. Should be same as number of clusters.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        alpha: parameter in Student's t-distribution. Default is 1.0.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    '''
    def __init__(self, output_dim, input_dim=None, weights=None, alpha=1.0, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.alpha = alpha
        # kmeans cluster centre locations
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(ClusteringLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = K.variable(self.initial_weights)
        self._trainable_weights = [self.W]

    def call(self, x, mask=None):
        q = 1.0/(1.0 + K.sqrt(K.sum(K.square(K.expand_dims(x, 1) - self.W), axis=2))**2 /self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = K.transpose(K.transpose(q)/K.sum(q, axis=1))
        return q

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeepEmbeddingClustering(object):
    def __init__(self,
                 n_clusters,
                 input_dim,
                 encoded=None,
                 decoded=None,
                 alpha=1.0,
                 pretrained_weights=None,
                 cluster_centres=None,
                 batch_size=256,
                 **kwargs):

        super(DeepEmbeddingClustering, self).__init__()

        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.encoded = encoded
        self.decoded = decoded
        self.alpha = alpha
        self.pretrained_weights = pretrained_weights
        self.cluster_centres = cluster_centres
        self.batch_size = batch_size

        self.learning_rate = 0.1
        self.iters_lr_update = 20000
        self.lr_change_rate = 0.1

        # greedy layer-wise training before end-to-end training:

        self.encoders_dims = [self.input_dim, 500, 500, 2000, 10]

        self.input_layer = Input(shape=(self.input_dim,), name='input')
        dropout_fraction = 0.2
        init_stddev = 0.01

        self.layer_wise_autoencoders = []
        self.encoders = []
        self.decoders = []
        for i  in range(1, len(self.encoders_dims)):
            
            encoder_activation = 'linear' if i == (len(self.encoders_dims) - 1) else 'relu'
            encoder = Dense(self.encoders_dims[i], activation=encoder_activation,
                            input_shape=(self.encoders_dims[i-1],),
                            kernel_initializer=RandomNormal(mean=0.0, stddev=init_stddev, seed=None),
                            bias_initializer='zeros', name='encoder_dense_%d'%i)
            self.encoders.append(encoder)

            decoder_index = len(self.encoders_dims) - i
            decoder_activation = 'linear' if i == 1 else 'relu'
            decoder = Dense(self.encoders_dims[i-1], activation=decoder_activation,
                            kernel_initializer=RandomNormal(mean=0.0, stddev=init_stddev, seed=None),
                            bias_initializer='zeros',
                            name='decoder_dense_%d'%decoder_index)
            self.decoders.append(decoder)

            autoencoder = Sequential([
                Dropout(dropout_fraction, input_shape=(self.encoders_dims[i-1],), 
                        name='encoder_dropout_%d'%i),
                encoder,
                Dropout(dropout_fraction, name='decoder_dropout_%d'%decoder_index),
                decoder
            ])
            autoencoder.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=0, momentum=0.9))
            self.layer_wise_autoencoders.append(autoencoder)

        # build the end-to-end autoencoder for finetuning
        # Note that at this point dropout is discarded
        self.encoder = Sequential(self.encoders)
        self.encoder.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=0, momentum=0.9))
        self.decoders.reverse()
        self.autoencoder = Sequential(self.encoders + self.decoders)
        self.autoencoder.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=0, momentum=0.9))

        if cluster_centres is not None:
            assert cluster_centres.shape[0] == self.n_clusters
            assert cluster_centres.shape[1] == self.encoder.layers[-1].output_dim

        if self.pretrained_weights is not None:
            self.autoencoder.load_weights(self.pretrained_weights)

    def p_mat(self, q):
        weight = q**2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def initialize(self, X, save_autoencoder=False, layerwise_pretrain_iters=50000, finetune_iters=100000):
        if self.pretrained_weights is None:

            iters_per_epoch = int(len(X) / self.batch_size)
            layerwise_epochs = max(int(layerwise_pretrain_iters / iters_per_epoch), 1)
            finetune_epochs = max(int(finetune_iters / iters_per_epoch), 1)

            print('layerwise pretrain')
            current_input = X
            lr_epoch_update = max(1, self.iters_lr_update / float(iters_per_epoch))
            
            def step_decay(epoch):
                initial_rate = self.learning_rate
                factor = int(epoch / lr_epoch_update)
                lr = initial_rate / (10 ** factor)
                return lr
            lr_schedule = LearningRateScheduler(step_decay)

            for i, autoencoder in enumerate(self.layer_wise_autoencoders):
                if i > 0:
                    weights = self.encoders[i-1].get_weights()
                    dense_layer = Dense(self.encoders_dims[i], input_shape=(current_input.shape[1],),
                                        activation='relu', weights=weights,
                                        name='encoder_dense_copy_%d'%i)
                    encoder_model = Sequential([dense_layer])
                    encoder_model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=0, momentum=0.9))
                    current_input = encoder_model.predict(current_input)

                autoencoder.fit(current_input, current_input, 
                                batch_size=self.batch_size, epochs=layerwise_epochs, callbacks=[lr_schedule])
                self.autoencoder.layers[i].set_weights(autoencoder.layers[1].get_weights())
                self.autoencoder.layers[len(self.autoencoder.layers) - i - 1].set_weights(autoencoder.layers[-1].get_weights())
            
            print('Finetuning autoencoder')
            
            #update encoder and decoder weights:
            self.autoencoder.fit(X, X, batch_size=self.batch_size, epochs=finetune_epochs, callbacks=[lr_schedule])

            if save_autoencoder:
                self.autoencoder.save_weights('autoencoder.h5')
        else:
            print('Loading pretrained weights for autoencoder.')
            self.autoencoder.load_weights(self.pretrained_weights)

        # update encoder, decoder
        # TODO: is this needed? Might be redundant...
        for i in range(len(self.encoder.layers)):
            self.encoder.layers[i].set_weights(self.autoencoder.layers[i].get_weights())

        # initialize cluster centres using k-means
        print('Initializing cluster centres with k-means.')
        if self.cluster_centres is None:
            print('setting cluster centers')
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            print('predicting')
            print('encoder',self.encoder.predict(X))
            try:
                self.y_pred = kmeans.fit_predict(self.encoder.predict(X))
            except:
                return
            self.cluster_centres = kmeans.cluster_centers_

        # prepare DEC model
        #self.DEC = Model(inputs=self.input_layer,
        #                 outputs=ClusteringLayer(self.n_clusters,
        #                                        weights=self.cluster_centres,
        #                                        name='clustering')(self.encoder))
        print('sequential1')
        self.DEC = Sequential([self.encoder,
                             ClusteringLayer(self.n_clusters,
                                                weights=self.cluster_centres,
                                                name='clustering')])
        print('compile')
        self.DEC.compile(loss='kullback_leibler_divergence', optimizer='adadelta')
        return

    def cluster_acc(self, y_true, y_pred):
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max())+1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
#         return sum([w[i, j] for i, j in ind])*1.0/y_pred.size, w
        return sum([w[i, j] for i, j in zip(ind[0],ind[1])])*1.0/y_pred.size, w


    def cluster(self, X, y=None,
                tol=0.01, update_interval=None,
                iter_max=1e6,
                save_interval=None,
                **kwargs):

        if update_interval is None:
            # 1 epochs
            update_interval = X.shape[0]/self.batch_size
        print('Update interval', update_interval)

        if save_interval is None:
            # 50 epochs
            save_interval = X.shape[0]/self.batch_size*50
        print('Save interval', save_interval)

        assert save_interval >= update_interval

        train = True
        iteration, index = 0, 0
        self.accuracy = []

        while train:
            sys.stdout.write('\r')
            # cutoff iteration
            if iter_max < iteration:
                print('Reached maximum iteration limit. Stopping training.')
                return self.y_pred

            # update (or initialize) probability distributions and propagate weight changes
            # from DEC model to encoder.
            if iteration % update_interval == 0:
                self.q = self.DEC.predict(X, verbose=0)
                self.p = self.p_mat(self.q)

                y_pred = self.q.argmax(1)
                delta_label = ((y_pred == self.y_pred).sum().astype(np.float32) / y_pred.shape[0]) #np.float32
                if y is not None:
#                     acc, w = self.cluster_acc(y, y_pred)
                    acc = self.cluster_acc(y, y_pred)[0]
                    self.accuracy.append(acc)
                    print('Iteration '+str(iteration)+', Accuracy '+str(np.round(acc, 5)))
                else:
                    print(str(np.round(delta_label*100, 5))+'% change in label assignment')

                if delta_label < tol:
                    print('Reached tolerance threshold. Stopping training.')
                    train = False
                    continue
                else:
                    self.y_pred = y_pred

                for i in range(len(self.encoder.layers)):
                    self.encoder.layers[i].set_weights(self.DEC.layers[0].layers[i].get_weights())
                self.cluster_centres = self.DEC.layers[-1].get_weights()[0]

            # train on batch
            sys.stdout.write('Iteration %d, ' % iteration)
            if (index+1)*self.batch_size > X.shape[0]:
                loss = self.DEC.train_on_batch(X[index*self.batch_size::], self.p[index*self.batch_size::])
                index = 0
                sys.stdout.write('Loss %f' % loss)
            else:
                loss = self.DEC.train_on_batch(X[index*self.batch_size:(index+1) * self.batch_size],
                                               self.p[index*self.batch_size:(index+1) * self.batch_size])
                sys.stdout.write('Loss %f' % loss)
                index += 1

            # save intermediate
            if iteration % save_interval == 0:
                z = self.encoder.predict(X)
                pca = PCA(n_components=2).fit(z)
                z_2d = pca.transform(z)
                clust_2d = pca.transform(self.cluster_centres)
                # save states for visualization
                pickle.dump({'z_2d': z_2d, 'clust_2d': clust_2d, 'q': self.q, 'p': self.p},
                            open('c'+str(iteration)+'.pkl', 'wb'))
                # save DEC model checkpoints
                self.DEC.save('DEC_model_'+str(iteration)+'.h5')

            iteration += 1
            sys.stdout.flush()
        return

import os
import sys
import multiprocessing  
import time
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cross_decomposition import CCA
from sklearn.cluster import SpectralClustering
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI_score, adjusted_rand_score as ARI_score, rand_score as rand_score
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score, mean_squared_error # cohen_kappa_score, hinge_loss, coverage_error, consensus_score
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linprog
from scipy.optimize import linear_sum_assignment as linear_assignment

def calculate_scores_nonpath(pred_clusters, true_clusters):
    """
    Calculate cluster accuracy scores and requires true clusters
    Calls cluster_along_path on each index i along penalties in path
    Params:
        true_clusters
        path_length
    Returns:
        path_clusters = list of clusterings at each penalty on path
        nmi_score = np-array of normalized mutual information score for each penalty
        adj_rand_score = np-array of adjusted rand score for each penalty
        rand_score = np-array of rand score for each penalty
        mse_score = np-array of mean squared error score for each penalty
    """
    nmi_score = NMI_score(true_clusters, pred_clusters, average_method='arithmetic')
    adj_rand_score = ARI_score(true_clusters, pred_clusters)
    ri_score = rand_score(true_clusters, pred_clusters)
    mse_score = mean_squared_error(true_clusters, pred_clusters)
     
    return nmi_score, adj_rand_score, ri_score, mse_score

def confusion_matrix_ordered(pred, true):
    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)
    conf_mat = confusion_matrix(pred,true)
    indexes = linear_assignment(_make_cost_m(conf_mat))
    js = [e for e in sorted(indexes, key=lambda x: x[0])[1]]
    conf_mat_ord = conf_mat[:, js]
    return conf_mat_ord


    #     # Load MNIST
    # X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters = load_MNIST(labels_keep=[0,1,2,3,4,5], plot=False, skip=1, batch_size=50, randomize=False)

def fit_dec(X_in, true_clusters_in, batch_size_options=[15, 30], finetune_iters_options=[100, 1000], layerwise_pretrain_iters_options=[100, 1000], cluster_iter_max_options=[100, 200]):
    import sys
    import numpy as np
    import keras.backend as K
    from keras.initializers import RandomNormal
    # from keras.engine.topology import Layer, InputSpec
    from tensorflow.keras.layers import Layer, InputSpec

    from keras.models import Model, Sequential
    from keras.layers import Dense, Dropout, Input

    try:
        from tensorflow.keras.optimizers.legacy import SGD
    except:
        try:
            from tensorflow.keras.optimizers import SGD
        except:
            from keras.optimizers import SGD

    from sklearn.preprocessing import normalize
    from keras.callbacks import LearningRateScheduler
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    if (sys.version[0] == 2):
        import cPickle as pickle
    else:
        import pickle
    import numpy as np

    tic = time.time()


    from sklearn.preprocessing import StandardScaler
    scalerX = StandardScaler()
    scalerX.fit(X_in)

    # RUN DEC Grid Search
    X = scalerX.transform(X_in).astype(np.float32) #np.float32
    Y = true_clusters_in.astype(np.int32)
    print(X.shape)

    # batch_size_options = [15] #[15, 30]
    # finetune_iters_options= [10]# [100, 1000]
    # layerwise_pretrain_iters_options= [10] #[100, 1000]
    # cluster_iter_max_options = [10] #[100, 200]
    idx = 0
    accuracies = []
    acc = []
    for batch_size in batch_size_options:
        for finetune_iters in finetune_iters_options:
            for layerwise_pretrain_iters in layerwise_pretrain_iters_options:
                    for cluster_iter_max in cluster_iter_max_options:
                        # print('batch_size',batch_size ,'finetune_iters',finetune_iters, 'cluster_iter_max:', cluster_iter_max)
                        c = DeepEmbeddingClustering(n_clusters=len(np.unique(Y)), input_dim=X.shape[1], batch_size=batch_size)
                        c.initialize(X, finetune_iters=finetune_iters, layerwise_pretrain_iters=layerwise_pretrain_iters)
                        print(np.unique(Y))
                        print(np.max(X))
                        try:
                            labels = c.cluster(X, y=Y, iter_max=cluster_iter_max)
                            # print(c.accuracy)
                        except:
                            acc = np.nan
                            print('failed to converge')

                        if ~np.isnan(acc):
                            # Calculate accuracy
                            conf_mat_ord = confusion_matrix_ordered(labels,Y)
                            acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
                            print('IDX:',idx, 'Accuracy:', acc, 'Batch size:',batch_size, 'finetune_iters:',finetune_iters, 'layerwise_pretrain_iters:',layerwise_pretrain_iters, 'cluster_iter_max:',cluster_iter_max)
                            idx = idx+1
                            
                        accuracies.append([idx, acc, batch_size, finetune_iters, layerwise_pretrain_iters, cluster_iter_max,])

                        # make it so string returned is the best accuracy...
        # pd.DataFrame(accuracies, columns = ['idx','Accuracy', 'BatchSize','finetune_iters', 'layerwise_pretrain_iters', 'cluster_iter_max'])
    toc = time.time() - tic
    print('Time elapsed:',toc)
    return accuracies, toc, 'IDX: '+str(idx)+' Accuracy: '+str(acc)+' Batch size: '+str(batch_size)+' finetune_iters: '+str(finetune_iters)+' layerwise_pretrain_iters: '+str(layerwise_pretrain_iters)+' cluster_iter_max: '+str(cluster_iter_max)



