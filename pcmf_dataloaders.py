# IMPORTS

# Paths
import time
import os
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::ConvergenceWarning,ignore::RuntimeWarning')
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
import math

# os.chdir('..')
from pcmf import pcmf_full, pcmf_approx_uV, path_plot, plot_ordercolor, plot_cluster_assignments, prox_numba_arr, chol_D, SVD, SVD2, get_weights
from p3ca import cluster_metrics, calculate_scores_nonpath, admm_CCA_new

import numpy as np
# %load_ext autoreload
import nilearn
from nilearn import image
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.decomposition import NMF as sk_NMF

import time
from datetime import datetime

# %matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

import rpy2

from torch.utils.data import Dataset
import h5py

def load_macaqueLGNrnaseq(data_path='/Users/amandabuch/Documents/2_Research/Postdoc/Code/PhaseTransitions/Datasets/data/', plot=False, skip=1, batch_size=50, randomize=False):
    import numpy as np
    import os
    from matplotlib import pyplot
    import pandas as pd
    import h5py


    f=h5py.File(data_path+'macaque_LGN_2021.h5','r')
    labels = np.load(data_path+'macaque_LGN_2021.npz', allow_pickle=True)
    Xall = f['X']
    Yall = labels['labels']
    Xall = Xall[Yall!='Low Quality',0:15000]
    Yall = Yall[Yall!='Low Quality']
    
    print(np.unique(Yall))
    
    train_X = Xall[0:int(np.floor(Xall.shape[0]/2)),:]
    test_X = Xall[int(np.floor(Xall.shape[0]/2)):Xall.shape[0],:]
    train_y = Yall[0:int(np.floor(Xall.shape[0]/2))]
    test_y = Yall[int(np.floor(Xall.shape[0]/2)):Xall.shape[0]]
    
    # Select training set
    X_in = train_X
    true_clusters_in = pd.factorize(train_y)[0]
    
    # Randomize training setorder
    if randomize is True:
        print('Randomizing order seed 42')
        idxs = np.random.RandomState(seed=42).permutation(X_in.shape[0])
        X_in2 = X_in[idxs,:]
        true_clusters_in2 = true_clusters_in[idxs]
        
        X_in = X_in2
        true_clusters_in = true_clusters_in2
        del X_in2, true_clusters_in2
        
#     # Select training set
#     X_in = train_X
#     true_clusters_in = pd.factorize(train_y)[0]
    
    # Shape of dataset
    print('Xall: ' + str(Xall.shape))
    print('Yall: ' + str(Yall.shape))
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))
    
    # subset evenly sampled across remaining n_X using skip
    X_in = X_in[::skip]
    true_clusters_in = true_clusters_in[::skip]
    true_clusters_labels = pd.factorize(train_y)[1]
    num_clusters = len(np.unique(true_clusters_in))
    print('X_train',X_in.shape, 'Y_train', true_clusters_in.shape)    
    print('Class labels',true_clusters_labels,'indexed as',np.unique(true_clusters_in))
    
    # Select test set
    X_in_test = test_X
    true_clusters_in_test = pd.factorize(test_y)[0]
    print('X_test',X_in.shape, 'Y_test', true_clusters_in.shape)

    # Subset training set to be a multiple of the batch size, batch_size
    n_X = X_in.shape[0]
    mod = n_X % batch_size
    n_X_new = n_X - mod
    print('n_X', n_X, 'batch size', batch_size, 'modulo', mod, 'is now', 'n_X', n_X_new, 'batch size', batch_size, 'modulo', (n_X - mod) % batch_size)

    X_in = X_in[0:n_X_new,:]
    true_clusters_in = true_clusters_in[0:n_X_new]

    print('Training set is now:', X_in.shape, true_clusters_in.shape)
    
    return X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters, Xall


def load_mouseLGNrnaseq(data_path='/Users/amandabuch/Documents/2_Research/Postdoc/Code/PhaseTransitions/Datasets/data/', plot=False, skip=1, batch_size=50, randomize=False):
    import numpy as np
    import os
    from matplotlib import pyplot
    import pandas as pd
    import h5py

    f=h5py.File(data_path+'mouse_LGN_2021.h5','r')
    labels = np.load(data_path+'mouse_LGN_2021.npz', allow_pickle=True)
    Xall = f['X']
    Yall = labels['labels']
    Yall = Yall[0:Xall.shape[0]] # !! TEMPORARY FIX BECAUSE YALL IS 3 LONGER THAN XALL!
    Xall = Xall[Yall!='Low Quality',0:5000]
    Yall = Yall[Yall!='Low Quality']
    Xall = Xall[Yall!='Non-Neuronal',:]
    Yall = Yall[Yall!='Non-Neuronal']
    
    print(np.unique(Yall))
    
    train_X = Xall[0:int(np.floor(Xall.shape[0]*0.95)),:]
    test_X = Xall[int(np.floor(Xall.shape[0]*0.95)):Xall.shape[0],:]
    train_y = Yall[0:int(np.floor(Xall.shape[0]*0.95))]
    test_y = Yall[int(np.floor(Xall.shape[0]*0.95)):Xall.shape[0]]
    
    # Select training set
    X_in = train_X
    true_clusters_in = pd.factorize(train_y)[0]
    
    # Randomize training setorder
    if randomize is True:
        print('Randomizing order seed 42')
        idxs = np.random.RandomState(seed=42).permutation(X_in.shape[0])
        X_in2 = X_in[idxs,:]
        true_clusters_in2 = true_clusters_in[idxs]
        
        X_in = X_in2
        true_clusters_in = true_clusters_in2
        del X_in2, true_clusters_in2
        
    # Shape of dataset
    print('Xall: ' + str(Xall.shape))
    print('Yall: ' + str(Yall.shape))
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print(np.unique(train_y))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))
    print(np.unique(test_y))
    
    # subset evenly sampled across remaining n_X using skip
    X_in = X_in[::skip]
    true_clusters_in = true_clusters_in[::skip]
    true_clusters_labels = pd.factorize(train_y)[1]
    num_clusters = len(np.unique(true_clusters_in))
    print('X_train',X_in.shape, 'Y_train', true_clusters_in.shape)    
    print('Class labels',true_clusters_labels,'indexed as',np.unique(true_clusters_in))
    
    # Select test set
    X_in_test = test_X
    true_clusters_in_test = pd.factorize(test_y)[0]
    print('X_test',X_in.shape, 'Y_test', true_clusters_in.shape)

    # Subset training set to be a multiple of the batch size, batch_size
    n_X = X_in.shape[0]
    mod = n_X % batch_size
    n_X_new = n_X - mod
    print('n_X', n_X, 'batch size', batch_size, 'modulo', mod, 'is now', 'n_X', n_X_new, 'batch size', batch_size, 'modulo', (n_X - mod) % batch_size)

    X_in = X_in[0:n_X_new,:]
    true_clusters_in = true_clusters_in[0:n_X_new]

    print('Training set is now:', X_in.shape, true_clusters_in.shape)
    
    return X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters, Xall


def load_humanATAC(data_path='/Users/amandabuch/Documents/2_Research/Postdoc/Code/PhaseTransitions/Datasets/data/', plot=False, skip=1, batch_size=50, randomize=False):
    import numpy as np
    import os
    from matplotlib import pyplot
    import pandas as pd
    import h5py
    print('test2')
    # f=h5py.File(data_path+'All-snATAC-Spatial multi-omic map of human myocardial infarction.h5','r')
    f=h5py.File(data_path+'All-snATAC-Spatial multi-omic map of human myocardial infarction_wCellLabels.h5','r')
    labels = np.load(data_path+'All-snATAC-Spatial multi-omic map of human myocardial infarction_wCellLabels.npz', allow_pickle=True)
    Xall = f['X']
    # Yall = labels['labels'] # patient groups
    Yall = labels['cell_labels'] # cell types
    print(np.unique(Yall))
    Xall = Xall[Yall!='Low Quality',0:1500]#,0:500]
    Yall = Yall[Yall!='Low Quality']

    # Xall = Xall[Yall!='Cardiomyocyte',:]
    # Yall = Yall[Yall!='Cardiomyocyte']

    Xall = Xall[Yall!='Fibroblast',:]
    Yall = Yall[Yall!='Fibroblast']

    Xall = Xall[Yall!='Lymphoid',:]
    Yall = Yall[Yall!='Lymphoid']

    Xall = Xall[Yall!='vSMCs',:] 
    Yall = Yall[Yall!='vSMCs']

    Xall = Xall[Yall!='Pericyte',:] ## Cardiomyocyte' 'Endothelial' 'Fibroblast' 'Lymphoid' 'Myeloid''Neuronal' 'Pericyte' 'vSMCs']
    Yall = Yall[Yall!='Pericyte']


#     Xall = Xall[0:5000,:]
#     Yall = Yall[0:5000]
    n,p = Xall.shape

    print(Xall.shape, Yall.shape)
    
    inds=np.arange(Xall.shape[0])
    test_inds=inds[::10]
    train_inds=list(set(inds)-set(test_inds))
    print(train_inds[0:10])

    train_X = Xall[train_inds,:]
    test_X = Xall[test_inds,:]
    train_y = Yall[train_inds]
    test_y = Yall[test_inds]
    
    # Select training set
    X_in = train_X
    true_clusters_in = pd.factorize(train_y)[0]
    
    # Randomize training setorder
    if randomize is True:
        print('Randomizing order seed 42')
        idxs = np.random.RandomState(seed=42).permutation(X_in.shape[0])
        X_in2 = X_in[idxs,:]
        true_clusters_in2 = true_clusters_in[idxs]
        
        X_in = X_in2
        true_clusters_in = true_clusters_in2
        del X_in2, true_clusters_in2
        
    # Shape of dataset
    print('Xall: ' + str(Xall.shape))
    print('Yall: ' + str(Yall.shape))
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))
    
    # subset evenly sampled across remaining n_X using skip
    X_in = X_in[::skip]
    true_clusters_in = true_clusters_in[::skip]
    true_clusters_labels = pd.factorize(train_y)[1]
    num_clusters = len(np.unique(true_clusters_in))
    print('X_train',X_in.shape, 'Y_train', true_clusters_in.shape)    
    print('Class labels',true_clusters_labels,'indexed as',np.unique(true_clusters_in))
    
    # Select test set
    X_in_test = test_X
    true_clusters_in_test = pd.factorize(test_y)[0]
    print('X_test',X_in.shape, 'Y_test', true_clusters_in.shape)
    print('Class labels',true_clusters_labels,'indexed as',np.unique(true_clusters_in_test))

    # Subset training set to be a multiple of the batch size, batch_size
    n_X = X_in.shape[0]
    mod = n_X % batch_size
    n_X_new = n_X - mod
    print('n_X', n_X, 'batch size', batch_size, 'modulo', mod, 'is now', 'n_X', n_X_new, 'batch size', batch_size, 'modulo', (n_X - mod) % batch_size)

    X_in = X_in[0:n_X_new,:]
    true_clusters_in = true_clusters_in[0:n_X_new]

    print('Training set is now:', X_in.shape, true_clusters_in.shape)
    
    return X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters, Xall


def load_mouseOrganRNAseq(data_path='/Users/amandabuch/Documents/2_Research/Postdoc/Code/PhaseTransitions/Datasets/data/', plot=False, skip=1, batch_size=50, randomize=False):
    import numpy as np
    import os
    from matplotlib import pyplot
    import pandas as pd
    import h5py

    # Load Mouse Organ data
    f=h5py.File(data_path+'mouse_organs.h5','r')
    labels = np.load(data_path+'mouse_organs_labels.npz', allow_pickle=True)
    # print(list(f.keys()), list(labels.keys()))

    Xtr = f['training_data_scaled']
    Xts = f['test_data_scaled']
    Xall = np.concatenate((Xtr,Xts))
    Ytr = labels['training_labels']
    Yts = labels['test_labels']
    Yall = np.concatenate((Ytr,Yts))
    n,p = Xall.shape
    
    Xall = Xall[Yall!='Liver',:] #['Liver' 'Thymus' 'Heart' 'Lung' 'Spleen' 'Kidney' 'Large_Intestine']
    Yall = Yall[Yall!='Liver']
    
#     Xall = Xall[Yall!='Lung',:] #['Liver' 'Thymus' 'Heart' 'Lung' 'Spleen' 'Kidney' 'Large_Intestine']
#     Yall = Yall[Yall!='Lung']
    
    Xall = Xall[Yall!='Spleen',:] #['Liver' 'Thymus' 'Heart' 'Lung' 'Spleen' 'Kidney' 'Large_Intestine']
    Yall = Yall[Yall!='Spleen']

    Xall = Xall[Yall!='Kidney',:] #['Liver' 'Thymus' 'Heart' 'Lung' 'Spleen' 'Kidney' 'Large_Intestine']
    Yall = Yall[Yall!='Kidney']
    
#     Xall = Xall[Yall!='Large_Intestine',:] #['Liver' 'Thymus' 'Heart' 'Lung' 'Spleen' 'Kidney' 'Large_Intestine']
#     Yall = Yall[Yall!='Large_Intestine']
    
    Xall = Xall[Yall!='Thymus',:] #['Liver' 'Thymus' 'Heart' 'Lung' 'Spleen' 'Kidney' 'Large_Intestine']
    Yall = Yall[Yall!='Thymus']
    
    Xall = Xall[Yall!='Heart',:] #['Liver' 'Thymus' 'Heart' 'Lung' 'Spleen' 'Kidney' 'Large_Intestine']
    Yall = Yall[Yall!='Heart']
    
    print(Xall.shape, Yall.shape)
    
    inds=np.arange(Xall.shape[0])
    test_inds=inds[::10]
    train_inds=list(set(inds)-set(test_inds))
    print(train_inds[0:10])

    train_X = Xall[train_inds,:]
    test_X = Xall[test_inds,:]
    train_y = Yall[train_inds]
    test_y = Yall[test_inds]
    
    # Select training set
    X_in = train_X
    true_clusters_in = pd.factorize(train_y)[0]
    
    # Randomize training setorder
    if randomize is True:
        print('Randomizing order seed 42')
        idxs = np.random.RandomState(seed=42).permutation(X_in.shape[0])
        X_in2 = X_in[idxs,:]
        true_clusters_in2 = true_clusters_in[idxs]
        
        X_in = X_in2
        true_clusters_in = true_clusters_in2
        del X_in2, true_clusters_in2
        
    # Shape of dataset
    print('Xall: ' + str(Xall.shape))
    print('Yall: ' + str(Yall.shape))
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))
    
    # subset evenly sampled across remaining n_X using skip
    X_in = X_in[::skip]
    true_clusters_in = true_clusters_in[::skip]
    true_clusters_labels = pd.factorize(train_y)[1]
    num_clusters = len(np.unique(true_clusters_in))
    print('X_train',X_in.shape, 'Y_train', true_clusters_in.shape)    
    print('Class labels',true_clusters_labels,'indexed as',np.unique(true_clusters_in))
    
    # Select test set
    X_in_test = test_X
    true_clusters_in_test = pd.factorize(test_y)[0]
    print('X_test',X_in.shape, 'Y_test', true_clusters_in.shape)
    print('Class labels',true_clusters_labels,'indexed as',np.unique(true_clusters_in_test))

    # Subset training set to be a multiple of the batch size, batch_size
    n_X = X_in.shape[0]
    mod = n_X % batch_size
    n_X_new = n_X - mod
    print('n_X', n_X, 'batch size', batch_size, 'modulo', mod, 'is now', 'n_X', n_X_new, 'batch size', batch_size, 'modulo', (n_X - mod) % batch_size)

    X_in = X_in[0:n_X_new,:]
    true_clusters_in = true_clusters_in[0:n_X_new]

    print('Training set is now:', X_in.shape, true_clusters_in.shape)
    
    return X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters, Xall




def load_GbmBreastLungCancergenomics_geneProteinMIRNA_3class(data_path='data/'):
    import pandas as pd
    # Load GbmBreastLungCancer dataset
    # GENES
    kidney = pd.read_csv(os.path.join(data_path, 'kidney_Z_gene.txt'), sep='\t', header=0)
    lung = pd.read_csv(os.path.join(data_path, 'lung_Z_gene.txt'), sep='\t', header=0)
    breast = pd.read_csv(os.path.join(data_path, 'breast_Z_gene.txt'), sep='\t', header=0)
    labels = np.hstack((np.repeat('kidney',kidney.shape[0]),np.repeat('lung',lung.shape[0]),np.repeat('breast',breast.shape[0])))
    kidney_T = kidney.T
    kidney_T['variables'] = kidney.columns
    lung_T = lung.T
    lung_T['variables'] = lung.columns
    breast_T = breast.T
    breast_T['variables'] = breast.columns
    data_T = pd.merge(kidney_T,pd.merge(lung_T,breast_T,on='variables'),on='variables')
    data = data_T.drop(columns='variables').T
    data.columns = data_T['variables']

    data_gene = data
    labels_gene = labels
    del data, data_T, labels, kidney, lung, breast, kidney_T, lung_T, breast_T

  
    # miRNAs
    kidney = pd.read_csv(os.path.join(data_path, 'kidney_Z_mirna.txt'), sep='\t', header=0)
    lung = pd.read_csv(os.path.join(data_path, 'lung_Z_mirna.txt'), sep='\t', header=0)
    breast = pd.read_csv(os.path.join(data_path, 'breast_Z_mirna.txt'), sep='\t', header=0)
    labels = np.hstack((np.repeat('kidney',kidney.shape[0]),np.repeat('lung',lung.shape[0]),np.repeat('breast',breast.shape[0])))
    kidney_T = kidney.T
    kidney_T['variables'] = kidney.columns
    lung_T = lung.T
    lung_T['variables'] = lung.columns
    breast_T = breast.T
    breast_T['variables'] = breast.columns
    data_T = pd.merge(kidney_T,pd.merge(lung_T,breast_T,on='variables'),on='variables')
    data = data_T.drop(columns='variables').T
    data.columns = data_T['variables']
    
    data_mirna = data
    labels_mirna = labels
    del data, data_T, labels, kidney, lung, breast, kidney_T, lung_T, breast_T


    return data_gene, labels_gene, data_mirna, labels_mirna

def load_GbmBreastLungCancergenomics_geneProteinMIRNA(data_path='data/'):
    import pandas as pd
    # Load GbmBreastLungCancer dataset
    # GENES
#     gbm = pd.read_csv(os.path.join(data_path, 'gbm_Z_gene.txt'), sep='\t', header=0)
#     gbm = pd.read_csv(os.path.join(data_path, 'gbm_Z_gene.txt'), sep='\t', header=0)
    lung = pd.read_csv(os.path.join(data_path, 'lung_Z_gene.txt'), sep='\t', header=0)
    breast = pd.read_csv(os.path.join(data_path, 'breast_Z_gene.txt'), sep='\t', header=0)
    labels = np.hstack((np.repeat('lung',lung.shape[0]),np.repeat('breast',breast.shape[0])))
#     labels = np.hstack((np.repeat('gbm',gbm.shape[0]),np.repeat('lung',lung.shape[0]),np.repeat('breast',breast.shape[0])))
#     gbm_T = gbm.T
#     gbm_T['variables'] = gbm.columns
    lung_T = lung.T
    lung_T['variables'] = lung.columns
    breast_T = breast.T
    breast_T['variables'] = breast.columns
    data_T = pd.merge(lung_T,breast_T,on='variables')
#     data_T = pd.merge(gbm_T,pd.merge(lung_T,breast_T,on='variables'),on='variables')
    data = data_T.drop(columns='variables').T
    data.columns = data_T['variables']
    print(data.shape)

    data_gene = data
    labels_gene = labels
    del data, data_T, labels, lung, breast, lung_T, breast_T
#     del data, data_T, labels, gbm, lung, breast, gbm_T, lung_T, breast_T
    
    # PROTEINS
    gbm = pd.read_csv(os.path.join(data_path, 'gbm_Z_methy.txt'), sep='\t', header=0)
    lung = pd.read_csv(os.path.join(data_path, 'lung_Z_methy.txt'), sep='\t', header=0)
    breast = pd.read_csv(os.path.join(data_path, 'breast_Z_methy.txt'), sep='\t', header=0)
    labels = np.hstack((np.repeat('lung',lung.shape[0]),np.repeat('breast',breast.shape[0])))
#     labels = np.hstack((np.repeat('gbm',gbm.shape[0]),np.repeat('lung',lung.shape[0]),np.repeat('breast',breast.shape[0])))
#     gbm_T = gbm.T
#     gbm_T['variables'] = gbm.columns
    lung_T = lung.T
    lung_T['variables'] = lung.columns
    breast_T = breast.T
    breast_T['variables'] = breast.columns
    data_T = pd.merge(lung_T,breast_T,on='variables')
#     data_T = pd.merge(gbm_T,pd.merge(lung_T,breast_T,on='variables'),on='variables')
    data = data_T.drop(columns='variables').T
    data.columns = data_T['variables']

    data_protein = data
    labels_protein = labels
    del data, data_T, labels, lung, breast, lung_T, breast_T
#     del data, data_T, labels, gbm, lung, breast, gbm_T, lung_T, breast_T
  
    # miRNAs
    gbm = pd.read_csv(os.path.join(data_path, 'gbm_Z_mirna.txt'), sep='\t', header=0)
    lung = pd.read_csv(os.path.join(data_path, 'lung_Z_mirna.txt'), sep='\t', header=0)
    breast = pd.read_csv(os.path.join(data_path, 'breast_Z_mirna.txt'), sep='\t', header=0)
    labels = np.hstack((np.repeat('lung',lung.shape[0]),np.repeat('breast',breast.shape[0])))
#     labels = np.hstack((np.repeat('gbm',gbm.shape[0]),np.repeat('lung',lung.shape[0]),np.repeat('breast',breast.shape[0])))
#     gbm_T = gbm.T
#     gbm_T['variables'] = gbm.columns
    lung_T = lung.T
    lung_T['variables'] = lung.columns
    breast_T = breast.T
    breast_T['variables'] = breast.columns
    data_T = pd.merge(lung_T,breast_T,on='variables')
#     data_T = pd.merge(gbm_T,pd.merge(lung_T,breast_T,on='variables'),on='variables')
    data = data_T.drop(columns='variables').T
    data.columns = data_T['variables']

    data_mirna = data
    labels_mirna = labels
    del data, data_T, labels, lung, breast, lung_T, breast_T
#     del data, data_T, labels, gbm, lung, breast, gbm_T, lung_T, breast_T


    return data_gene, labels_gene, data_protein, labels_protein, data_mirna, labels_mirna


def load_GBMBreastLung_N400(data_path='/Users/amandabuch/Documents/clusterCCA/revision1/clusterCCA/data/', plot=False, skip=1, batch_size=50, randomize=False):
    import numpy as np
    import os
    from matplotlib import pyplot
    import pandas as pd

    print('Loading GBM Breast Lung')
    # Load GbmBreastLungCancer dataset
    gbm = pd.read_csv(os.path.join(data_path, 'gbm_z.txt'), sep='\t', header=0)
    lung = pd.read_csv(os.path.join(data_path, 'lung_z.txt'), sep='\t', header=0)
    breast = pd.read_csv(os.path.join(data_path, 'breast_z.txt'), sep='\t', header=0)
    labels = np.hstack((np.repeat('gbm',gbm.shape[0]),np.repeat('lung',lung.shape[0]),np.repeat('breast',breast.shape[0])))
    gbm_T = gbm.T
    gbm_T['variables'] = gbm.columns
    lung_T = lung.T
    lung_T['variables'] = lung.columns
    breast_T = breast.T
    breast_T['variables'] = breast.columns
    data_T = pd.merge(gbm_T,pd.merge(lung_T,breast_T,on='variables'),on='variables')
    data = data_T.drop(columns='variables').T
    data.columns = data_T['variables']
    
    train_X = data.to_numpy()[0:400,:]
    test_X = data.to_numpy()[400:data.shape[0],:]
    train_y = labels[0:400]
    test_y = labels[400:data.shape[0]]

    # Shape of dataset
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))
    
    num_clusters = 3

    if plot is True:
        # Plotting
        from matplotlib import pyplot as plt
        # Set data generation parameters
        colors = ['darkblue','darkorange','red','green','purple','gray','pink']
        plot_idxs = [0,1]
        scatter_cmap = 'tab20b'
        scatter_alpha = 0.3
        plot_idxs = [0,1]

        # Plot clusters
        plt.figure(figsize=(6,6))
        for nc in range(num_clusters):
            plt.scatter(train_X[train_y==nc,plot_idxs[0]],train_X[train_y==nc,plot_idxs[1]], c=colors[nc])
        plt.xlabel('First Variable')
        plt.ylabel('Second Variable')
        plt.gca().patch.set_facecolor('white')

    # Select training set
    X_in = train_X
    true_clusters_in = pd.factorize(train_y)[0]
    
    # Randomize training setorder
    if randomize is True:
        print('Randomizing order seed 42')
        idxs = np.random.RandomState(seed=42).permutation(X_in.shape[0])
        X_in2 = X_in[idxs,:]
        true_clusters_in2 = true_clusters_in[idxs]
        
        X_in = X_in2
        true_clusters_in = true_clusters_in2
        del X_in2, true_clusters_in2
    
    # subset evenly sampled across remaining n_X using skip
    X_in = X_in[::skip]
    true_clusters_in = true_clusters_in[::skip]
    true_clusters_labels = pd.factorize(train_y)[1]
    num_clusters = len(np.unique(true_clusters_in))
    print('X_train',X_in.shape, 'Y_train', true_clusters_in.shape)    
    print('Class labels',true_clusters_labels,'indexed as',np.unique(true_clusters_in))
    
    # Select test set
    X_in_test = test_X
    true_clusters_in_test = pd.factorize(test_y)[0]
    print('X_test',X_in.shape, 'Y_test', true_clusters_in.shape)

    # Subset training set to be a multiple of the batch size, batch_size
    n_X = X_in.shape[0]
    mod = n_X % batch_size
    n_X_new = n_X - mod
    print('n_X', n_X, 'batch size', batch_size, 'modulo', mod, 'is now', 'n_X', n_X_new, 'batch size', batch_size, 'modulo', (n_X - mod) % batch_size)

    X_in = X_in[0:n_X_new,:]
    true_clusters_in = true_clusters_in[0:n_X_new]

    print('Training set is now:', X_in.shape, true_clusters_in.shape)
    
    return X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters, data


    

# MNIST
def load_MNIST(labels_keep=[0,3,5], plot=False, skip=1, batch_size=50, randomize=False):
    import numpy as np
    from keras.datasets import mnist
    from matplotlib import pyplot
    import pandas as pd
    
    print('Loading MNIST')
    # Loading
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # Shape of dataset
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))

    if plot is True:
        # Plotting
        from matplotlib import pyplot
        for i in range(9):  
            pyplot.subplot(330 + 1 + i)
            pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
            pyplot.show()

    # Select data corresponding to a subset of the latbels
    inds = []
    inds_test = []
    for ind in labels_keep:
        inds.append(np.where(train_y==ind)[0])
        inds_test.append(np.where(test_y==ind)[0])
        
    inds = np.hstack(inds)
    inds_test = np.hstack(inds_test)

    # Select training set
    X_in = train_X[inds,:].reshape((train_X[inds,:].shape[0],28*28))
    true_clusters_in = pd.factorize(train_y[inds])[0]
    
    # Randomize training setorder
    if randomize is True:
        print('Randomizing order seed 42')
        idxs = np.random.RandomState(seed=42).permutation(X_in.shape[0])
        X_in2 = X_in[idxs,:]
        true_clusters_in2 = true_clusters_in[idxs]
        
        X_in = X_in2
        true_clusters_in = true_clusters_in2
        del X_in2, true_clusters_in2

    # subset evenly sampled across remaining n_X using skip
    X_in = X_in[::skip]
    true_clusters_in = true_clusters_in[::skip]
    true_clusters_labels = pd.factorize(train_y[inds])[1]
    num_clusters = len(np.unique(true_clusters_in))
    print('X_train',X_in.shape, 'Y_train', true_clusters_in.shape)    
    print('Class labels',true_clusters_labels,'indexed as',np.unique(true_clusters_in))    
    # Select test set
    X_in_test = test_X[inds_test,:].reshape((test_X[inds_test,:].shape[0],28*28))
    true_clusters_in_test = pd.factorize(test_y[inds_test])[0]
    print('X_test',X_in.shape, 'Y_test', true_clusters_in.shape)

    # Subset training set to be a multiple of the batch size, batch_size
    n_X = X_in.shape[0]
    mod = n_X % batch_size
    n_X_new = n_X - mod
    print('n_X', n_X, 'batch size', batch_size, 'modulo', mod, 'is now', 'n_X', n_X_new, 'batch size', batch_size, 'modulo', (n_X - mod) % batch_size)

    X_in = X_in[0:n_X_new,:]
    true_clusters_in = true_clusters_in[0:n_X_new]

    print('Training set is now:', X_in.shape, true_clusters_in.shape)
    
    return X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters

    
    

# Fashion MNIST
def load_FashionMNIST(labels_keep=[0,3,5], plot=False, skip=1, batch_size=50, randomize=False):
    import numpy as np
    from keras.datasets import fashion_mnist
    from matplotlib import pyplot
    import pandas as pd

    print('Loading Fashion MNIST')
    # Loading
    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()

    # Shape of dataset
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))

    if plot is True:
        # Plotting
        from matplotlib import pyplot
        for i in range(9):  
            pyplot.subplot(330 + 1 + i)
            pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
            pyplot.show()

    # Select data corresponding to a subset of the latbels
    inds = []
    inds_test = []
    for ind in labels_keep:
        inds.append(np.where(train_y==ind)[0])
        inds_test.append(np.where(test_y==ind)[0])
        
    inds = np.hstack(inds)
    inds_test = np.hstack(inds_test)

    # Select training set
    X_in = train_X[inds,:].reshape((train_X[inds,:].shape[0],28*28))
    true_clusters_in = pd.factorize(train_y[inds])[0]
    
    # Randomize training setorder
    if randomize is True:
        print('Randomizing order seed 42')
        idxs = np.random.RandomState(seed=42).permutation(X_in.shape[0])
        X_in2 = X_in[idxs,:]
        true_clusters_in2 = true_clusters_in[idxs]
        
        X_in = X_in2
        true_clusters_in = true_clusters_in2
        del X_in2, true_clusters_in2
    
    # subset evenly sampled across remaining n_X using skip
    X_in = X_in[::skip]
    true_clusters_in = true_clusters_in[::skip]
    true_clusters_labels = pd.factorize(train_y[inds])[1]
    num_clusters = len(np.unique(true_clusters_in))
    print('X_train',X_in.shape, 'Y_train', true_clusters_in.shape)    
    print('Class labels',true_clusters_labels,'indexed as',np.unique(true_clusters_in))
    
    # Select test set
    X_in_test = test_X[inds_test,:].reshape((test_X[inds_test,:].shape[0],28*28))
    true_clusters_in_test = pd.factorize(test_y[inds_test])[0]
    print('X_test',X_in.shape, 'Y_test', true_clusters_in.shape)

    # Subset training set to be a multiple of the batch size, batch_size
    n_X = X_in.shape[0]
    mod = n_X % batch_size
    n_X_new = n_X - mod
    print('n_X', n_X, 'batch size', batch_size, 'modulo', mod, 'is now', 'n_X', n_X_new, 'batch size', batch_size, 'modulo', (n_X - mod) % batch_size)

    X_in = X_in[0:n_X_new,:]
    true_clusters_in = true_clusters_in[0:n_X_new]

    print('Training set is now:', X_in.shape, true_clusters_in.shape)
    
    return X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters


# Synthetic data
def load_syntheticDataConsensus(n=100000, n_test=10000, p=1000, m=25000, m_test=2500, num_clusters=4, means=[-1.0, 1.0, -0.4, 0.4], r=1, sigma=0.075, density=0.5, plot=False, randomize=False, pcmf_dir="/Users/amandabuch/Documents/clusterCCA/PCMF/"):
    import numpy as np
    import os
    os.chdir(pcmf_dir)
    from pcmf import generate_cluster_PMD_data
    print('Loading synthetic data, run: '+str(r))
    seeds = [r,r+1,r+2,r+3]
    
    # Cluster sizes
    ms = [m+m_test,m+m_test,m+m_test,m+m_test]
    
    # Get clustered CCA data
    X_clusters, u_true, v_true, _ = generate_cluster_PMD_data(ms, p, sigma, density, num_clusters, means=means) 
    # Training set
    X_c_train = []
    X_in_test = []
    for nc in range(num_clusters):
        X_c_train.append(X_clusters[nc][0:m,:])
        X_in_test.append(X_clusters[nc][m:m+m_test,:])
    X_c_train = np.vstack(X_c_train)
    true_clusters = np.repeat([0,1,2,3],m)
    # Test set
    X_in_test = np.vstack(X_in_test)
    true_clusters_in_test = np.repeat([0,1,2,3],m_test)
    
    # Shape of dataset
    print('X_c_train: ' + str(X_c_train.shape))
    print('Y_train: ' + str(true_clusters.shape))
    print('X_c_test:  '  + str(X_in_test.shape))
    print('Y_test:  '  + str(true_clusters_in_test.shape))
    
    # Randomize training setorder
    if randomize is True:
        print('Randomizing order seed 1234')
        idx_perm = np.random.RandomState(seed=1234).permutation(X_c_train.shape[0])
        X_in = X_c_train[idx_perm,:]
        true_clusters_in = true_clusters[idx_perm]
    else:
        X_in = X_c_train
        true_clusters_in = true_clusters
    
    return X_in, true_clusters_in, X_in_test, true_clusters_in_test, num_clusters, u_true, v_true


# CIFAR10
def load_cifar10(labels_keep=[0,3,5], plot=False, skip=1, batch_size=50, randomize=False):
    import numpy as np
    from keras.datasets import cifar10
    from matplotlib import pyplot
    import pandas as pd

    print('Loading Cifar10')
    # Loading
    (train_X, train_y), (test_X, test_y) = cifar10.load_data()

    # Shape of dataset
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))

    if plot is True:
        # Plotting
        from matplotlib import pyplot
        for i in range(9):  
            pyplot.subplot(330 + 1 + i)
            pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
            pyplot.show()

    # Select data corresponding to a subset of the latbels
    inds = []
    inds_test = []
    for ind in labels_keep:
        inds.append(np.where(train_y==ind)[0])
        inds_test.append(np.where(test_y==ind)[0])
        
    inds = np.hstack(inds)
    inds_test = np.hstack(inds_test)

    # Select training set
    X_in = train_X[inds,:].reshape((train_X[inds,:].shape[0],32*32*3))
    true_clusters_in = pd.factorize(train_y[inds,0])[0]
    
    # Randomize training setorder
    if randomize is True:
        print('Randomizing order seed 42')
        idxs = np.random.RandomState(seed=42).permutation(X_in.shape[0])
        X_in2 = X_in[idxs,:]
        true_clusters_in2 = true_clusters_in[idxs]
        
        X_in = X_in2
        true_clusters_in = true_clusters_in2
        del X_in2, true_clusters_in2
    
    # subset evenly sampled across remaining n_X using skip
    X_in = X_in[::skip]
    true_clusters_in = true_clusters_in[::skip]
    true_clusters_labels = pd.factorize(train_y[inds,0])[1]
    num_clusters = len(np.unique(true_clusters_in))
    print('X_train',X_in.shape, 'Y_train', true_clusters_in.shape)    
    print('Class labels',true_clusters_labels,'indexed as',np.unique(true_clusters_in))
    
    # Select test set
    X_in_test = test_X[inds_test,:].reshape((test_X[inds_test,:].shape[0],32*32*3))
    true_clusters_in_test = pd.factorize(test_y[inds_test,0])[0]
    print('X_test',X_in.shape, 'Y_test', true_clusters_in.shape)

    # Subset training set to be a multiple of the batch size, batch_size
    n_X = X_in.shape[0]
    mod = n_X % batch_size
    n_X_new = n_X - mod
    print('n_X', n_X, 'batch size', batch_size, 'modulo', mod, 'is now', 'n_X', n_X_new, 'batch size', batch_size, 'modulo', (n_X - mod) % batch_size)

    X_in = X_in[0:n_X_new,:]
    true_clusters_in = true_clusters_in[0:n_X_new]

    print('Training set is now:', X_in.shape, true_clusters_in.shape)
    
    return X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters


import pandas as pd
def load_COVID(path='./results/pcmf_approx_uV_COVIDCancer_genomics_run_gausscoef2.0_neighbors15_rho1.5.npz'):
    from sklearn.preprocessing import StandardScaler
    data_path = 'data/covid_j.cell.2020.05.032'
    proteomics = pd.read_csv(os.path.join(data_path,'proteomics_data_preprocessed_shared.csv'), sep=',',index_col=1, header=0)
    proteomics_batch = pd.read_csv(os.path.join(data_path,'proteomics_batch_shared.csv'), sep=',',index_col=None, header=None)
    proteomics_labels = pd.read_csv(os.path.join(data_path,'proteomics_labels_shared.csv'), sep=',',index_col=None, header=None)

    accessionTogene = pd.read_csv(os.path.join(data_path,'protein-accessionTogene.csv'), sep=',',index_col=None, header=0)


    metabolomic = pd.read_csv(os.path.join(data_path,'metabolomics_data_preprocessed_shared.csv'), sep=',',index_col=0, header=0)
    metabolomic_labels = pd.read_csv(os.path.join(data_path,'metabolomics_labels_shared.csv'), sep=',',index_col=None, header=None)

    metabolomic=metabolomic.iloc[: , 1:]
    proteomics=proteomics.iloc[: , 1:]

    clinical = pd.read_csv(os.path.join(data_path,'clinical_shared.csv'), sep=',',index_col=0, header=0)
    blood_mat = clinical.iloc[:,6:18].to_numpy()
    blood_mat = clinical.iloc[:,np.hstack((np.arange(3,5),np.arange(6,18)))].to_numpy()
    blood_df = clinical.iloc[:,np.hstack((np.arange(3,5),np.arange(6,18)))]
    labels = metabolomic_labels.to_numpy()

    keep = np.where(np.logical_or(labels.flatten()=='ZX',np.logical_or(labels.flatten()=='PT',labels.flatten()=='jkdz')))[0]
    X_c = metabolomic.to_numpy()
    X_c = X_c[keep,:]
    X_c = 2**X_c
    Y_c = proteomics.to_numpy()
    Y_c = Y_c[keep,:]
    labels_c = labels.flatten()[keep]

    scaler = StandardScaler()
    scaler.fit(X_c)
    X_c = scaler.transform(X_c)
    X_c = np.hstack((X_c,np.ones((X_c.shape[0],1))))

    scaler = StandardScaler()
    scaler.fit(Y_c)
    Y_c = scaler.transform(Y_c)
    Y_c = np.hstack((Y_c,np.ones((Y_c.shape[0],1))))

    labels = metabolomic_labels.to_numpy()
    print("X_c",X_c.shape,"Y_c",Y_c.shape,"labels",labels_c.shape)

    true_clusters=pd.factorize(labels_c)[0]
    n_clusters = len(np.unique(true_clusters))
    print("Cluster types:",np.unique(labels_c))

    x = np.hstack((X_c,Y_c)).astype(np.float32)
    y = pd.factorize(labels_c)[0].astype(np.int32)
    print('COVID samples', x.shape)
    return x, y, X_c, Y_c


def load_mouseorgans(path='/Users/amandabuch/Documents/clusterCCA/revision1/clusterCCA/data/'):
    # f = np.load(path, allow_pickle=True)
    import h5py
    f=h5py.File(path+'mouse_organs.h5','r')
    labels = np.load(path+'mouse_organs_labels.npz', allow_pickle=True)
    # print(list(f.keys()), list(labels.keys()))

    # Xtr = f['training_data_scaled']
    # Xts = f['test_data_scaled']
    # Xall = np.concatenate((Xtr,Xts))
    # Ytr = labels['training_labels']
    # Yts = labels['test_labels']
    # Yall = np.concatenate((Ytr,Yts))
    # n,p = Xall.shape
    # x,y = Xall, Yall

    rows_skip = 50
    training_data_scaled_subset = f['training_data_scaled'][::rows_skip]
    training_labels_subset = labels['training_labels'][::rows_skip]
    x,y = training_data_scaled_subset, training_labels_subset

    f.close()
    x = x.astype(np.float32)
    y = pd.factorize(y)[0].astype(np.int32)
    print('Mouse Organs samples', x.shape)
    return x, y


def load_gbmBreastLung(path='/Users/amandabuch/Documents/clusterCCA/revision1/clusterCCA/data/GbmBreastLungCancer.npz'):
    f = np.load(path, allow_pickle=True)
    x,y = f['X'], f['labels']
    f.close()
    x = x.astype(np.float32)
    y = pd.factorize(y)[0].astype(np.int32)
    print('gbmBreastLung samples', x.shape)
    return x, y

def load_NCI(path='/Users/amandabuch/Documents/clusterCCA/revision1/clusterCCA/data/NCI.npz'):
    f = np.load(path, allow_pickle=True)
    x,y = f['X'], f['labels']
    f.close()
    x = x.astype(np.float32)
    y = pd.factorize(y)[0].astype(np.int32)
    print('NCI samples', x.shape)
    return x, y


def load_SRBCT(path='/Users/amandabuch/Documents/clusterCCA/revision1/clusterCCA/data/SRBCT.npz'):
    f = np.load(path, allow_pickle=True)
    x,y = f['X'], f['labels']
    f.close()
    x = x.astype(np.float32)
    y = pd.factorize(y)[0].astype(np.int32)
    print('SRBCT samples', x.shape)
    return x, y

class anyDataset(Dataset): # make sure this still works for selecting datatype
    import torch
    from torch.utils.data import Dataset
    def __init__(self, datatype='NCI'):
        print('Dataset:',datatype)
        if datatype=='NCI':
            self.x, self.y = load_NCI()
        elif datatype=='SRBCT':
            self.x, self.y = load_SRBCT()
        elif datatype=='gbmBreastLung':
            self.x, self.y = load_gbmBreastLung()
        elif datatype=='mouseorgans':
            self.x, self.y = load_mouseorgans()
        else:
            print(str(datatype)+' not implemented')
            return

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))

def load_mouseorgans_MV(path='/Users/amandabuch/Documents/clusterCCA/revision1/clusterCCA/data/'):
    import scanpy as sc
    from anndata import AnnData
    # f = np.load(path, allow_pickle=True)
    f=h5py.File(path+'mouse_organs.h5','r')
    labels = np.load(path+'mouse_organs_labels.npz', allow_pickle=True)
    f=h5py.File(path+'mouse_organs.h5','r')
    labels = np.load(path+'mouse_organs_labels.npz', allow_pickle=True)
    # print(list(f.keys()), list(labels.keys()))

    # Xtr = f['training_data_scaled']
    # Xts = f['test_data_scaled']
    # Xall = np.concatenate((Xtr,Xts))
    # Ytr = labels['training_labels']
    # Yts = labels['test_labels']
    # Yall = np.concatenate((Ytr,Yts))
    # n,p = Xall.shape
    # x,y = Xall, Yall

    rows_skip = 50
    training_data_scaled_subset = f['training_data_scaled'][::rows_skip]
    training_labels_subset = labels['training_labels'][::rows_skip]
    x,y = training_data_scaled_subset, training_labels_subset

    f.close()
    x = x.astype(np.float32)
    y = pd.factorize(y)[0].astype(np.int32)

    dataset = anyDataset(datatype='mouseorgans')
    X = dataset.x[:,0:1000] #[:,0:1000]
    Y = dataset.x[:,1001:1100] #[:,1000:dataset.x.shape[1]]
    # X = dataset.x[:,0:500] #[:,0:1000]
    # Y = dataset.x[:,500:1500] #[:,1000:dataset.x.shape[1]]
    true_clusters = dataset.y

    print('mouseorgans mv samples', X.shape, Y.shape)
    return X, Y, true_clusters


def load_NCI_MV(path='Users/amandabuch/Documents/clusterCCA/revision1/clusterCCA/data/NCI.npz'):
    import scanpy as sc
    from anndata import AnnData
    f = np.load(path, allow_pickle=True)
    x,y = f['X'], f['labels']
    f.close()
    x = x.astype(np.float32)
    y = pd.factorize(y)[0].astype(np.int32)

    dataset = anyDataset(datatype='NCI')
    X = dataset.x[:,0:1000] #[:,0:1000]
    Y = dataset.x[:,1001:1100] #[:,1000:dataset.x.shape[1]]
    # X = dataset.x[:,0:500] #[:,0:1000]
    # Y = dataset.x[:,500:1500] #[:,1000:dataset.x.shape[1]]
    true_clusters = dataset.y

    print('NCI mv samples', X.shape, Y.shape)
    return X, Y, true_clusters


def load_gbmBreastLung_MV(path='Users/amandabuch/Documents/clusterCCA/revision1/clusterCCA/data/GbmBreastLungCancer.npz'):
    import scanpy as sc
    from anndata import AnnData
    f = np.load(path, allow_pickle=True)
    x,y = f['X'], f['labels']
    f.close()
    x = x.astype(np.float32)
    y = pd.factorize(y)[0].astype(np.int32)

    dataset = anyDataset(datatype='gbmBreastLung')
    X = dataset.x[:,0:1000] # why is this different than NCI?
    Y = dataset.x[:,1001:1100] # why is this different than NCI?
    # X = dataset.x[:,0:500] # why is this different than NCI?
    # Y = dataset.x[:,1000:1500] # why is this different than NCI?
    true_clusters = dataset.y

    print('gbmBreastLung mv samples', X.shape, Y.shape)
    return X, Y, true_clusters


def load_SRBCT_MV(path='Users/amandabuch/Documents/clusterCCA/revision1/clusterCCA/data/SRBCT.npz'):
    import scanpy as sc
    from anndata import AnnData
    f = np.load(path, allow_pickle=True)
    x,y = f['X'], f['labels']
    f.close()
    x = x.astype(np.float32)
    y = pd.factorize(y)[0].astype(np.int32)

    dataset = anyDataset(datatype='SRBCT')
    X = dataset.x[:,0:1000] # why is this different than NCI?
    Y = dataset.x[:,1001:1100] # why is this different than NCI?
    # X = dataset.x[:,0:1000] # why is this different than NCI?
    # Y = dataset.x[:,1000:1100] # why is this different than NCI?
    true_clusters = dataset.y

    print('SRBCT mv samples', X.shape, Y.shape)
    return X, Y, true_clusters


def load_penguins():
    from palmerpenguins import load_penguins as load_penguinsPP
    from sklearn.preprocessing import StandardScaler

    # penguins = load_penguinsPP()
    # import seaborn as sns
#     pairplot_figure = sns.pairplot(penguins, hue="species")
#     pairplot_figure.fig.set_size_inches(9, 6.5)

    data,species = load_penguinsPP(return_X_y = True)

    data_arr = data.to_numpy() 
    data_arr = data_arr[~np.isnan(data).any(axis=1)]
    species = species[~np.isnan(data).any(axis=1)]
    labels, true_clusters_penguin = np.unique(species, return_inverse=True)

    X_penguin = data_arr[:,0:2]
    Y_penguin = data_arr[:,2:4]
    print(X_penguin.shape, Y_penguin.shape)

    scalerX = StandardScaler()
    scalerY = StandardScaler()
    scalerX.fit(X_penguin)
    scalerY.fit(Y_penguin)

    X = scalerX.transform(X_penguin)
    Y = scalerY.transform(Y_penguin)

    x = np.hstack((X, Y, X, Y, X, Y,X, Y, X, Y, X, Y)).astype(np.float32)
    y = pd.factorize(true_clusters_penguin)[0].astype(np.int32)

    print('Penguins samples', x.shape, y.shape)
    return x, y, X, Y



