######################################################
### Code for running PCA/embedding with convex clustering using autoencoders ###
## Run a test linear autoencoder for PCA with convex clustering penalty model ##

## FLAG Note: ! Currently untested, but should work after some debugging given my previous code
######################################################
## Key aspects are:
###################
## Can initialize encoder weights as right singular vectors (eigenvectors) of X (by setting initial_weights in model_config)
## Tied weights constraint: encoder weights W = decoder weights W.T (like in PCA X_hat = U U.T X)
## Unit norm constraint: encoder weights are unit normalized along each hidden dimension (like in PCA)
## Orthogonality constraint: linear layers are parameterized such that the weights between hidden dimensions must be orthogonal (like in PCA)
## Convex clustering penalty soft thresholding loss: can apply group soft thresholding (prox_version) or un-thresholded convex clustering penalty on reconstructions

######################################################
## Notes:
## First note: Add a new model class with the LAE-RAG (rotational augmented gradient), or if not possible, then add the option for rotation in train_model() function
## Second note: Could implement convex clustering via a proximal gradient update on the weights instead of as a loss on the reconsructed X (following the standard Pytorch update)
## Third note: Add a new model class with multiple encoder / decoder layers and nonlinear activation function
## Forth note: Add a new model class with variation autoencoder
## Fifth note: Add option to add l2, l1, l2l1, elasticnet proximal gradient update on weights (closed-form regularization) following standard pytorch update (See: https://github.com/KentonMurray/ProxGradPytorch)
## Sixth note: This is a useful reference, though in tensorflow: https://towardsdatascience.com/build-the-right-autoencoder-tune-and-optimize-using-pca-principles-part-i-1f01f821999b
			   # Also here in Pytorch, useful: # https://gist.github.com/InnovArul/500e0c57e88300651f8005f9bd0d12bc on tied weights implementations
			   # https://discuss.pytorch.org/t/autoencoder-with-tied-weights-using-sequential/25296/4; https://discuss.pytorch.org/t/how-to-create-and-train-a-tied-autoencoder/2585/3
			   # You can rescale the weights within a with torch.no_grad(): block to not bother the autograd.

######################################################
##### GET DATA #####
seed=0
X_c, data, loader, hidden_dim = get_4cluster_data(r=seed)

# Initialize weights using right singular vectors of the centered input data, parallel to its eigenvectors.
# Center data
x = X_c; x -= np.mean(x, axis=0)
# Get the right singular vectors of the centered input data
_, _, Vt = randomized_svd(x, n_components=hidden_dim, n_oversamples=10, flip_sign=True)
inital_weights = torch.from_numpy(Vt[:,0:hidden_dim]) 
del x, Vt

#### DEFINE MODEL DICTIONARY #####
model_dict = dict(
    model_name='LAE_PCA',
    model_type='convex_clustering',
    model_class=LAE_PCA,
    extra_model_args = {'weight_reg_type':'convex_cluster', 'gauss_coef':1.0, 'neighbors':None, 'prox_version':True, 'initial_weights':inital_weights},
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    optim_class=torch.optim.SGD,
    extra_optim_args={'momentum': 0.9, 'nesterov': True},
    lr=0.0001,
    seed=seed
)

#### INSTANTIATE MODEL #####
# model config contains the model 
model_config = ModelConfig(
        model_name=model_dict['model_name'],
        model_type=model_dict['model_type'],
        model_class=model_dict['model_class'],
        input_dim=model_dict['input_dim'], 
        hidden_dim=model_dict['hidden_dim'],
        init_scale=model_dict['init_scale'],
        extra_model_args=model_dict['extra_model_args'],
        optim_class=model_dict['optim_class'],
        lr=model_dict['lr'],
        extra_optim_args=model_dict['extra_optim_args']
    )

#### TRAIN ALONG PATH #####
# lambda_path = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(10,-10,50))),axis=0)
lambda_path = np.exp(np.linspace(100,0,50))

 # if 1 batch, otherwise need to modify, need to check differences between this X and X_c
X = next(enumerate(loader))[1].data.detach().numpy()
encoder_weights_list, decoder_weights_list, embeddings_list, reconstructions_list, alignments_list, subspaceDists_list, model = \
train_models_path(X_train=X, data_weigs=data, train_loader=loader, model_config=model_config, \
                  lambda_path=lambda_path, epochs_per_lambda=25)


#### OUTPUT PLOTS #####
output_plots(X_c, encoder_weights_list, decoder_weights_list, embeddings_list, reconstructions_list, alignments_list, subspaceDists_list, lambda_path, true_clusters, hidden_dim, end_idx=10, plot_idx=20)


######################################################
######################################################
## Functions are below. Above is for testing.
######################################################
######################################################

######################################################
### Two implementations of convex clustering loss ###
#####################################################
def convclust_penalty_prox(recons, D, cc_weights, cc_lambda=0.1, q=2, rho=1.0):
    '''Group soft thresholding proximal operator for convex clustering of reconstructions
        Closed-form solution does not rely on gradients
        rho is similar to lr (learning rate)
    '''
    diffs = D.matmul(recons)
    if cc_weights is None:
        cc_weights = np.ones(diffs.shape[0])

    alpha = torch.from_numpy(cc_weights)*cc_lambda/rho
    diffs_norm = diffs.norm(q, dim=1) # Key insight here: apply 2-norm row-wise (by using dim 1)
    
    # Group soft thresholding
    diffs_prox = torch.where(diffs_norm > alpha, diffs - alpha.mul(diffs)/diffs_norm, torch.zeros_like(diffs, device=torch.device(device)) )
    cc_loss = torch.sum(diffs_prox) # diffs_prox.norm(1)

    return cc_loss

def convclust_penalty(recons, D, cc_weights, cc_lambda=0.0, q=2):
    '''
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffs_norm = torch.norm(D.matmul(recons), q, dim=1)  #vec_norm = np.linalg.norm(V[i,:])

    if cc_weights is None:
        cc_weights = np.ones(diffs_norm.shape[0])

    cc_loss = cc_lambda * torch.sum(torch.mul(cc_weights, diffs_norm)) #sum of weighted paired differences between samples
    # torch.norm(torch.mul(cc_weights, diffs_norm), 1) #sum of the absolute values of the columns. 

    return cc_loss

def get_weights(X, gauss_coef=1.0, neighbors=None):
    ''' 
    same function as PCMF with ADMM
    Construct convext clustering weights according to approaches from Hocking et al. 2011 
    and Chi and Lange, 2013. X is the original data matrix with observations in the rows. 
    ''' 
    from scipy.spatial.distance import pdist, squareform
    dist_vec = pdist(X) / X.shape[0]
    w = np.exp(-1*gauss_coef*(dist_vec)**2)
    if neighbors is not None:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm='ball_tree').fit(X)
        _, indices = nbrs.kneighbors(X)
        comb_list = list(combinations(range(X.shape[0]),2))
        neighbors_indicator = []
        for comb in comb_list:
            nn_i = indices[comb[0],:]
            if comb[1] in nn_i:
                neighbors_indicator.append(1.0)
            else:
                neighbors_indicator.append(0.0)
        w *= np.array(neighbors_indicator)
    return w

def sparse_D_AE(n,p):
    '''
    Construct a sparse matrix, that when applied to a vector containing concatenated vectors
    of coefficients b = [b_1 b_2 ... b_n] where each b_i is p=num_var long and there are
    n = num_vec of them. Differences are taken between conformal elements (e.g. b_11 and b_21)
    across all unique pairwise combinations of vectors.
    '''
    comb_list = list(combinations(range(n),2))
    combs_arr = np.array(comb_list)
    num_combs = combs_arr.shape[0]
    data = np.ones_like(combs_arr)
    data[:,1] *= -1
    row = np.repeat(range(num_combs),2)
    col = combs_arr.flatten()
    D = csr_matrix((data.flatten(), (row, col)), shape=(num_combs, n))
    return sparse_mx_to_torch_sparse_tensor(D) # this step is different in PCAE version

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    '''
    Convert a scipy sparse matrix to a torch sparse tensor.
    '''
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


######################################################
## Special Autoencoder Class ##
######################################################
class ModelConfig:
    def __init__(self, model_name, model_type, model_class, input_dim, hidden_dim, init_scale, optim_class, lr,
                 extra_model_args={}, extra_optim_args={}):
        self.model_name = model_name
        self.model_type = model_type
        self.model_class = model_class
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init_scale = init_scale
        self.extra_model_args = extra_model_args

        self.optim_class = optim_class
        self.lr = lr
        self.extra_optim_args = extra_optim_args
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model_class(input_dim=input_dim, hidden_dim=hidden_dim, init_scale=init_scale, **extra_model_args).to(device)

        self.optimizer = optim_class(self.model.parameters(), lr=lr, **extra_optim_args)

    @property
    def name(self):
        return self.model_name

    @property
    def type(self):
        return self.model_type

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer


class LAE_PCA(nn.Module):
    '''Linear autoencoder with 1 encoder layer, 1 decoder layer
        - tied encoder/decoder weights
        - Parameterized layers such that weights must be orthogonal
        - Weights are unit normalized
        - option to implement convex clustering objective as a loss regularizer by setting cc_loss=True
    '''
    def __init__(self, input_dim, hidden_dim, inital_weights=None, cc_loss=True, cc_lambda=0.0, gauss_coef=1.0, neighbors=None, prox_version=True, q=2, rho=1.0):
        super(LAE_PCA, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Define orthogonalized encoder and decoder 
        self.encoder = nn.utils.parametrizations.orthogonal(nn.Linear(inp, out, bias=False))
        self.decoder = nn.utils.parametrizeations.orthogonal(nn.Linear(out, inp, bias=False))

        # Parameterize weights to unit normalize along each hidden_dim and tie the weights
        if inital_weights is None:
        	# use random initialization of weights
        	inital_weights = torch.randn(input_dim, hidden_dim)
        self.encoder.weight = nn.Parameter(UnitNorm(inital_weights,axis=1)) # hidden_dim x p
        self.decoder.weight = nn.Parameter(UnitNorm(inital_weights,axis=1).transpose(0,1)) # p x hidden_dim

        # Convex clustering parameters
        self.cc_loss = cc_loss
        self.prox_version = prox_version
        self.q = q
        self.rho = rho
        self.cc_weights = None
        self.D = None
        self.cc_lambda = cc_lambda
        self.gauss_coef = gauss_coef
        self.neighbors = neighbors

    def UnitNorm(self, weight, axis):
        ''' Unit normalization of weights along axis (each hidden_dim should be unit norm)
            torch.finfo(torch.float32).eps is tf.keras.backend.epsilon()
        '''
        return weight / (
            torch.finfo(torch.float32).eps
            + torch.sqrt(torch.mean(torch.square(weight), axis, keepdims=True))
                        )

    def get_reconstruction_loss(self, x, recon):
        '''Objective 1: compute loss with goal
                        to minimize reconstruction error (MSE sum)
        '''
        ##  ##
        # Compute loss
        recon_loss = torch.sum((x - recon) ** 2) / len(x) # MSE loss, sum reduction
        # criterion = torch.nn.MSELoss(reduction='sum')
        # loss = criterion(recon, x)
        return recon_loss
   
    def get_convex_clustering_loss(self, recon):
        '''Objective 2: loss for minimizing the sum of between-observation distances of reconstruction
        '''
        if prox_version:
        	convclust_loss = convclust_penalty_prox(recon.detach().cpu(), self.D, self.cc_weights, self.cc_lambda, q=self.q, rho=self.rho)
    	else:
        	convclust_loss = convclust_penalty(recon.detach().cpu(), self.D, self.cc_weights, self.cc_lambda, q=self.q)
        return convclust_loss

    def forward(self, x):
        # Forward pass: Compute predicted reconstruction by passing x to the model
        z = self.encoder(x) # encoded_X
        recon = self.decoder(z) # reconstructed X

        if self.cc_loss == True:
            if self.cc_weights is None:
                print('Getting convex clustering weights.')
                x_np = x.detach().cpu().numpy().reshape(x.shape[0], np.prod(x.shape[1::])) # x_np is n x p
                self.cc_weights = torch.from_numpy(get_weights(x_np, gauss_coef=self.gauss_coef, neighbors=self.neighbors) ).to(device)
                del x_np
            if self.D is None:
                # Construct the sparse matrix graph for penalty differencing
                [n,p] = x.shape
                self.D = sparse_D_AE(n,p)
            return self.get_reconstruction_loss(x, recon) + self.get_convex_clustering_loss(recon), recon
        else:
            return self.get_reconstruction_loss(x, recon), recon

######################################################
## Training ##
######################################################

def train_model(data_loader, train_itr, hidden_dim, pbar=None):
	'''Train the model'''
    # Create a progress bar
    if pbar is None:
        pbar = Progbar(target=train_itr)

    for train_i in range(train_itr):
        for x_batch in data_loader:
            x_cuda = x_batch.to(device)

            model = model_config.get_model()
            optimizer = model_config.get_optimizer()

            if len(data_loader)>1:
                # Need to reset convex clustering weights if using batches
                # would be better to store this ahead, like in my consensus version so not recalculating each time
                model.cc_weights = None

            # Forward pass: 
            # Compute reconstruction by passing x to the model and compute the loss
            loss, X_recon = model(x_cuda)

            # Zero gradients, perform a backward pass (gradient descent minimizer), and update the weights.
            # zero the gradients
            optimizer.zero_grad()
            # back propagation
            loss.backward()
            # step to update the weights
            optimizer.step()

        pbar.update(train_i, values=[("lambda",model.cc_lambda), ("loss",list(losses.values())[0])])
    pbar.update(train_itr, values=None)

return model

######################################################
## Training along a path ##
######################################################

def train_model_path(X_train, data_weigs, train_loader, model_config, lambda_path,
                   epochs_per_lambda=5):
    ''' Train linear autoencoder (not setup currently for nested AE)'''
    embeddings_list = []
    encoder_weights_list = []
    decoder_weights_list = []
    reconstructions_list = []
    alignments_list = []
    subspaceDists_list = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, lambd in enumerate(lambda_path):
        print('Index ',i+1,'/',len(lambda_path))
        # Update convex clustering penalty
        model_config.get_model().cc_lambda = lambd
        # Train model over epochs_per_lambda iterations
        train_model(data_loader=train_loader, train_itr=epochs_per_lambda, model_configs=model_config, metrics_dict=None, eval_metrics_list=None)
        
        # Save encoder/decoder weights, encoded embeddings, and decoded reconstructions
        model = model_config.get_model()
        
        encoder_weights_list.append(get_weight_tensor_from_seq(model.encoder).cpu().numpy())
        decoder_weights_list.append(get_weight_tensor_from_seq(model.decoder).cpu().numpy())
#         encoder_weights_list.append(model.encoder.weight.data.detach().cpu().numpy())
#         decoder_weights_list.append(model.encoder.weight.data.detach().cpu().numpy())
        embeddings_list.append(model.encoder(torch.tensor(X_train, dtype=torch.float32).to(device)).detach().cpu().numpy())
        reconstructions_list.append(model.decoder(model.encoder(torch.tensor(X_train, dtype=torch.float32))).data.detach().cpu().numpy())

        alignments_list.append(metric_alignment(model, data_weigs.eigvectors))
        subspaceDists_list.append(metric_subspace(model, data_weigs.eigvectors, data.eigs))
        print('Distance to axis-aligned solution:', metric_alignment(model, data_weigs.eigvectors)) # alignment of decoder columns to ground truth eigenvectors
        print('Distance to optimal subspace):', metric_subspace(model, data_weigs.eigvectors, data.eigs),'\n')

    return encoder_weights_list, decoder_weights_list, embeddings_list, reconstructions_list, alignments_list, subspaceDists_list, model

######################################################
###### DEFINE DATA LOADER FUNCTIONS ######
######################################################
from torch.utils.data import Dataset, DataLoader 
import numpy as np
from scipy.stats import ortho_group

##### GET DATA ####
def get_4cluster_data(r=0, m=20, num_vars=10, hidden_dim=5, means=[-1.0, 1.0, -0.4, 0.4], sigma=0.075,density=1.0, plot_data=True):
	import os
	import torch
	import numpy as np
	from sklearn.preprocessing import StandardScaler
	from pcmf import two_cluster_data, generate_cluster_PMD_data

	print('run: '+str(r))
	n_clusters=4
	ms = [m,m,m,m]
	true_clusters=np.repeat([0,1,2,3],m)

	# Get clustered CCA data
	X_clusters, u_true, v_true, _=generate_cluster_PMD_data(ms, num_vars, sigma, density, n_clusters, means=means) 
	X_c=np.vstack(X_clusters)

	scaler = StandardScaler()
	scaler.fit(X_c)
	X_c = scaler.transform(X_c).astype('float32')

	# set random seed
	np.random.seed(r)
	torch.manual_seed(r)

	input_dim = num_vars

	n_data = X_c.shape[0]
	batch_size = n_data

	train_dataset = synthetic_Dataset(X_c, true_clusters)

	data = DataGeneratorPCA(input_dim, hidden_dim, load_data=train_dataset.X)

	loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

	if plot_data is True:
		import matplotlib
		import matplotlib.pyplot as plt
		scatter_cmap = 'tab20b'
		plt.figure(figsize=(4,4))
		plt.scatter(X_c[:,0],X_c[:,1], c=true_clusters, cmap=scatter_cmap, alpha=1)
		plt.xlabel('First Variable',fontsize=16)
		plt.ylabel('Second Variable',fontsize=16)

	return X_c, data, loader, hidden_dim

class synthetic_Dataset(Dataset):
    ''' Custom data class generated from PCMF synthetic data generator.. '''
    def __init__(self, X, X_labels):
        super(synthetic_Dataset,self).__init__()
        self.X = X
        self.X_labels = X_labels

    def __len__(self):
        return len(self.X_labels)
  
    def __getitem__(self, index):
        data = self.X[index]
        data_label = self.X_labels[index]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(data_label, dtype=torch.float32)


class DataGeneratorPPCA(Dataset):

    def __init__(self, dims, hdims, min_sv=0.11, max_sv=5.0, sigma_sq=0.1, deterministic=True, total=10000):
        self.dims = dims
        self.hdims = hdims

        self.eigs = min_sv + (max_sv - min_sv) * np.linspace(0, 1, hdims)
        self.eigvectors = ortho_group.rvs(dims)[:, :hdims]
        self.w = np.matmul(self.eigvectors, np.diag(np.sqrt(self.eigs - sigma_sq)))

        self.sigma_sq = sigma_sq
        self.sigma = np.sqrt(sigma_sq)

        self.total = total
        self.deterministic = deterministic
        if self.deterministic:
            self.z_sample = np.random.normal(size=(total, self.hdims))
            self.x_sample = np.random.normal(np.matmul(self.z_sample, self.w.T), self.sigma).astype(np.float32)

    def __getitem__(self, i):
        if self.deterministic:
            return self.x_sample[i]
        else:
            z_sample = np.random.normal(size=self.hdims)
            return np.random.normal(self.w.dot(z_sample), self.sigma).astype(np.float32)

    def __len__(self):
        # Return a large number for an epoch
        return self.total


class DataGeneratorPCA(Dataset):
    def __init__(self, dims, hdims, min_sv=0.11, max_sv=5.0, total=10000, sv_list=None,
                 load_data=None):
        self.dims = dims
        self.hdims = hdims

        if load_data is None:
            if isinstance(sv_list, list):
                assert len(sv_list) == dims
                self.full_eigs = np.array(sorted(sv_list, reverse=True))
            else:
                self.full_eigs = min_sv + (max_sv - min_sv) * np.linspace(1, 0, dims)
            self.eigs = self.full_eigs[:hdims]

            self.full_svs = np.sqrt(self.full_eigs)

            self.full_eigvectors = ortho_group.rvs(dims)
            self.eigvectors = self.full_eigvectors[:, :hdims]

            self.total = total

            self.full_z_sample = np.random.normal(size=(total, self.dims))
            self.x_sample = (self.full_eigvectors @ np.diag(self.full_svs) @ self.full_z_sample.T).T.astype(np.float32)

        else:
            self.x_sample = load_data
            u, s, vh = np.linalg.svd(self.x_sample.T, full_matrices=False)
            self.eigs = s[:self.hdims]
            self.eigvectors = u[:, :self.hdims]
            self.total = len(self.x_sample)

    def __getitem__(self, i):
        return self.x_sample[i]

    def __len__(self):
        return self.total

    @property
    def shape(self):
        return self.x_sample.shape


######################################################
###### DEFINE PLOTTING FUNCTIONS ######
######################################################
### Cluster matching ###

from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linprog
from scipy.optimize import linear_sum_assignment as linear_assignment


def cluster_matching(pred, true):
    'New function for hungrarian matching clusters'
    from sklearn.metrics import confusion_matrix, accuracy_score
    from scipy.optimize import linprog
    from scipy.optimize import linear_sum_assignment as linear_assignment
    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)
    conf_mat = confusion_matrix(pred,true)
    indexes = linear_assignment(_make_cost_m(conf_mat))

    pred_new = np.copy(pred)
    for clusterN in range(len(np.unique(pred))):
        pred_new[pred==clusterN]= indexes[1][clusterN]

    return pred_new


### add my video function...
from matplotlib import cm

def path_plot(coefficient_arr, penalty_list, plot_range=[0,-1], cut_vars=False, 
              first_vars_only=False, var_sel=1, true_clusters=None,figsize=(20,10), xticks=None):
    # Crop x axis (e.g, to remove 'burn-in' period at beginning)                                                         
    coefficient_arr = coefficient_arr[plot_range[0]:plot_range[1],:,:]
    penalty_list = penalty_list[plot_range[0]:plot_range[1]]
    if cut_vars is True:
        coefficient_arr = coefficient_arr[:,:,[1,2,coefficient_arr.shape[2]-1]]

    if first_vars_only is True:
        coefficient_arr = coefficient_arr[:,:,[var_sel]]

    # Colormap                                                                                                           
#     cmap = matplotlib.cm.get_cmap('tab20b')

     #cm.get_cmap('viridis', coefficient_arr.shape[2])
    if true_clusters is None:
        cmap = cm.get_cmap('viridis', coefficient_arr.shape[2])
        colors = cmap(np.linspace(0.0,1.0,coefficient_arr.shape[2]))
    else:
        cmap = matplotlib.cm.get_cmap('tab20b')
        colors = cmap(np.linspace(0.0,1.0,len(np.unique(true_clusters))))
   
    # Define x-axis range                                                                                                
    penalty_range = range(len(penalty_list))

    # Make figure
    fig, ax = plt.subplots(1,1, figsize=figsize)

    for i in range(coefficient_arr.shape[2]):
        x = np.round(np.array(penalty_list),8)[penalty_range]
        y = coefficient_arr[penalty_range,:,i]
        if true_clusters is not None:
            # Make different line types for different clusters
            linetypes = ['dotted']*len(true_clusters)
            color_list = [colors[i]]*len(true_clusters)
            for j, tc in enumerate(true_clusters):
                if tc == 1:
                    linetypes[j] = 'solid'
                    color_list[j] = colors[tc]
                elif tc == 2:
                    linetypes[j] = 'dashed'
                    color_list[j] = colors[tc]
                elif tc == 3:
                    linetypes[j] = 'dashdot'
                    color_list[j] = colors[tc]
                else:
                    linetypes[j] = 'dotted'
                    color_list[j] = colors[tc]

            # Plot the lines and set linestyle
            ax.plot(np.arange(x.shape[0]), y, color=colors[i], alpha=0.5)
            for l, line in enumerate(ax.get_lines()):
                line.set_linestyle(linetypes[l])
                line.set_color(color_list[l])
        else:
            ax.plot(np.arange(x.shape[0]), y, color=colors[i], alpha=0.5)
        # Set plot ticks and labels
        ax.set_xticks(range(x.shape[0]), minor=False);
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        x = [str(xx)[0:9] for xx in x]
        ax.set_xticklabels(x) #,fontsize=24)
        evens = np.arange(0,len(ax.xaxis.get_ticklabels())+1,2)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False) 
    ax.tick_params(axis='y') #, labelsize=24)
    plt.xlabel(r'$\lambda$',fontsize=16)
    plt.ylabel('Coefficients',fontsize=16)
    sns.despine()
    
    
def path_plot_video(coefficient_arr, penalty_list, penalty_idx, fig, ax, plot_range=[0,-1], cut_vars=False, 
              first_vars_only=False, var_sel=1, true_clusters=None,figsize=(20,10), xticks=None):
    # Crop x axis (e.g, to remove 'burn-in' period at beginning)                                                         
    coefficient_arr = coefficient_arr[plot_range[0]:plot_range[1],:,:]
    penalty_list = penalty_list[plot_range[0]:plot_range[1]]
    if cut_vars is True:
        coefficient_arr = coefficient_arr[:,:,[1,2,coefficient_arr.shape[2]-1]]

    if first_vars_only is True:
        coefficient_arr = coefficient_arr[:,:,[var_sel]]

    # Colormap                                                                                                           
    cmap = matplotlib.cm.get_cmap('tab20b')

    #cm.get_cmap('viridis', coefficient_arr.shape[2])
    if true_clusters is None:
        colors = cmap(np.linspace(0.0,1.0,3))
    else:
        colors = cmap(np.linspace(0.0,1.0,len(np.unique(true_clusters))))
    
    # Define x-axis range                                                                                                
    penalty_range = range(len(penalty_list))

    # Make figure
#     fig, ax = plt.subplots(1,1, figsize=figsize)

    for i in range(coefficient_arr.shape[2]):
        x = np.round(np.array(penalty_list),8)[penalty_range]
        y = coefficient_arr[penalty_range,:,i]
        if true_clusters is not None:
            # Make different line types for different clusters
            linetypes = ['dotted']*len(true_clusters)
            color_list = [colors[i]]*len(true_clusters)
            for j, tc in enumerate(true_clusters):
                if tc == 1:
                    linetypes[j] = 'solid'
                    color_list[j] = colors[tc]
                elif tc == 2:
                    linetypes[j] = 'dashed'
                    color_list[j] = colors[tc]
                elif tc == 3:
                    linetypes[j] = 'dashdot'
                    color_list[j] = colors[tc]
                else:
                    linetypes[j] = 'dotted'
                    color_list[j] = colors[tc]

            # Plot the lines and set linestyle
            ax.plot(np.arange(x.shape[0]), y, color=colors[i], alpha=0.5)
            for l, line in enumerate(ax.get_lines()):
                line.set_linestyle(linetypes[l])
                line.set_color(color_list[l])
        else:
            ax.plot(np.arange(x.shape[0]), y, color=colors[i], alpha=0.5)
        # Set plot ticks and labels
        ax.set_xticks(range(x.shape[0]), minor=False);
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        x = [str(xx)[0:9] for xx in x]
        ax.set_xticklabels(x) #,fontsize=24)
        evens = np.arange(0,len(ax.xaxis.get_ticklabels())+1,2)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False) 
    ax.tick_params(axis='y') #, labelsize=24)
    plt.xlabel(r'$\lambda$',fontsize=16)
    plt.ylabel('Coefficients',fontsize=16)
    
    plt.plot([penalty_idx,penalty_idx],[-0.1,0.5],color='black')
    plt.locator_params(axis='x',nbins=25)
        
    sns.despine()
###

def output_plots(X_c, encoder_weights_list, decoder_weights_list, embeddings_list, reconstructions_list, alignments_list, subspaceDists_list, lambda_path, true_clusters, hidden_dim, end_idx=10, plot_idx=20):
	import matplotlib
	import matplotlib.pyplot as plt
	import seaborn as sns
	path_plot(np.asarray(reconstructions_list)[:,:,0:end_idx], lambda_path, plot_range=[0,len(lambda_path)],cut_vars=False)

	path_plot(np.asarray(embeddings_list), lambda_path, plot_range=[0,len(lambda_path)],cut_vars=False)

	path_plot(np.asarray(encoder_weights_list)[:,:,0:end_idx], lambda_path, plot_range=[0,len(lambda_path)],cut_vars=False)

	path_plot(np.asarray(decoder_weights_list)[:,:,0:end_idx], lambda_path, plot_range=[0,len(lambda_path)],cut_vars=False)

	path_plot(np.asarray(alignments_list).reshape((len(lambda_path),1,1)), lambda_path, plot_range=[0,len(lambda_path)],cut_vars=False)

	path_plot(np.asarray(subspaceDists_list).reshape((len(lambda_path),1,1)), lambda_path, plot_range=[0,len(lambda_path)],cut_vars=False)

	num_clusters = len(np.unique(true_clusters))
	cluster_pclae_rag_list = []
	for i in range(len(lambda_path)):
	    print(i, end="")
	    print("...", end="")	    
	    clustering = KMeans(n_clusters=num_clusters, random_state=0).fit(np.asarray(reconstructions_list)[plot_idx,:,:])
	    cluster_pclae_rag_list.append(clustering.labels_)

	cluster_pclae_rag_new_list = []
	for idx in range(0,len(cluster_pclae_rag_list)):
	    cluster_pclae_rag_new_list.append( cluster_matching(cluster_pclae_rag_list[idx], true_clusters ) )

	# Plot paths
	path_plot(np.asarray(reconstructions_list)[:,:,0:1], lambda_path, true_clusters=true_clusters)
	path_plot(np.asarray(reconstructions_list)[:,:,0:1], lambda_path, true_clusters=cluster_pclae_rag_new_list[plot_idx])
	plt.locator_params(axis='x',nbins=25)

	num_components = hidden_dim
	colors = ['darkblue','darkorange','red','green','purple']
	plot_idxs = [0,1]
	scatter_cmap = 'tab20b'
	scatter_alpha = 0.3

	# Generate video of solutions along path
	for i in np.arange(len(lambda_path)-1,len(lambda_path)):
	    print(i, end="")
	    print("...", end="")
	    fig = plt.figure(figsize=(6,6))
	    ax=fig.gca()
	    ax.axes.xaxis.set_visible(False)
	    ax.axes.yaxis.set_visible(False)
	    ax = fig.add_subplot(111) #, autoscale_on=False) #, xlim=(-0.6, 0.6), ylim=(-0.6, 0.6))

	    cluster_pcmf = cluster_pclae_rag_new_list[i]

	    ax.scatter(X_c[:,0],X_c[:,1],c=true_clusters, alpha=scatter_alpha, cmap=scatter_cmap)

	    ax.scatter(np.asarray(reconstructions_list)[i,:,0],np.asarray(reconstructions_list)[i,:,1], c=cluster_pcmf, cmap=scatter_cmap)
	    ax.set(xlabel="First Variable")
	    ax.set(ylabel="Second Variable")
	    ax.text(0.95, 0.01, r"$\lambda =$"+str(np.round(lambda_path[i],5)),
	        verticalalignment='bottom', horizontalalignment='right',
	        transform=ax.transAxes,
	        color='black', fontsize=12)

	    # ax.set_xlim(-2.75, 3)
	    # ax.set_ylim(-3, 1.75)

######################################################
###### DEFINE EVALUATION METRICS FUNCTIONS ######
######################################################

import os
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_weight_tensor_from_seq(weight_seq):
    if isinstance(weight_seq, nn.Linear):
        return weight_seq.weight.detach()
    elif isinstance(weight_seq, nn.Sequential):
        weight_tensor = None
        for layer in weight_seq:
            if isinstance(layer, nn.Linear):
                layer_weight = layer.weight.detach()
                if weight_tensor is None:
                    weight_tensor = layer_weight
                else:
                    weight_tensor = layer_weight @ weight_tensor
            elif isinstance(layer, nn.BatchNorm1d):
                bn_weight = layer.weight.detach()

                # ignore bias

                if weight_tensor is None:
                    weight_tensor = torch.diag(bn_weight)
                else:
                    weight_tensor = torch.diag(bn_weight) @ weight_tensor
                    
            elif isinstance(layer,nn.ReLU):
                weight_tensor = weight_tensor
#                 print('ok')
#                 rl_weight = layer.weight.detach()
#                 if weight_tensor is None:
#                     weight_tensor = torch.diag(rl_weight)
#                 else:
#                     weight_tensor = torch.diag(rl_weight) @ weight_tensor
 
            else:
                raise ValueError("Layer type {} not supported!".format(type(layer)))
        return weight_tensor


def metric_transpose_theorem(model):
    """
    Metric for how close encoder and decoder.T are
    :param model: LinearAE model
    :return: ||W1 - W2^T||_F^2 / hidden_dim
    """
    encoder_weight = get_weight_tensor_from_seq(model.encoder)
    decoder_weight = get_weight_tensor_from_seq(model.decoder)

    transpose_metric = torch.norm(encoder_weight - decoder_weight.T) ** 2
    return transpose_metric.item() / float(model.hidden_dim)


def metric_alignment(model, gt_eigvectors):
    """
    Metric for alignment of decoder columns to ground truth eigenvectors
    :param model: Linear AE model
    :param gt_eigvectors: ground truth eigenvectors (input_dims,hidden_dims)
    :return: sum_i (1 - max_j (cos(eigvector_i, normalized_decoder column_j)))
    """
    decoder_weight = get_weight_tensor_from_seq(model.decoder)
    decoder_np = decoder_weight.detach().cpu().numpy()

    # normalize columns of gt_eigvectors
    norm_gt_eigvectors = gt_eigvectors / np.linalg.norm(gt_eigvectors, axis=0)
    # normalize columns of decoder
    norm_decoder = decoder_np / (np.linalg.norm(decoder_np, axis=0) + 1e-8)

    total_angles = 0.0
    for eig_i in range(gt_eigvectors.shape[1]):
        eigvector = norm_gt_eigvectors[:, eig_i]
        total_angles += 1. - np.max(np.abs(norm_decoder.T @ eigvector)) ** 2

    return total_angles / float(model.hidden_dim)


def metric_subspace(model, gt_eigvectors, gt_eigs):
    decoder_weight = get_weight_tensor_from_seq(model.decoder)
    decoder_np = decoder_weight.detach().cpu().numpy()

    # k - tr(UU^T WW^T), where W is left singular vector matrix of decoder
    u, s, vh = np.linalg.svd(decoder_np, full_matrices=False)
    return 1 - np.trace(gt_eigvectors @ gt_eigvectors.T @ u @ u.T) / float(model.hidden_dim)


def metric_loss(model, data_loader):
    """
    Measures the full batch loss
    :param model: a linear (variational) AE model
    :param data_loader: full batch data loader. Should be different from the training data loader, if in minibatch mode
    """
    loss = None
    for x in data_loader:
        loss = model(x.to(device)).item()
    return loss


def metric_recon_loss(model, data_loader):
    recon_loss = None
    for x in data_loader:
        recon_loss = model.get_reconstruction_loss(x.to(device)).item()
    return recon_loss



