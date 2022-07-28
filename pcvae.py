import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.nn.utils import weight_norm

from itertools import combinations
import numpy as np
import os
from scipy.sparse import csr_matrix
import secrets

class LAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(LAE, self).__init__()
        self.x_dim = x_dim
        print(x_dim)
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim) # V
        self.fc2 = nn.Linear(h_dim, z_dim) # d_i
        
        # decoder part
        self.fc3 = nn.Linear(z_dim, x_dim) # u
        
    def encoder(self, x):
        h = self.fc1(x)
        z = self.fc2(h)
        return z
    
    def decoder(self, z):
        xhat = self.fc3(z)
        return xhat
    
    def forward(self, x):
        z = self.encoder(x.view(-1, self.x_dim))
        xhat = self.decoder(z)
        return xhat, None, None

class LAE2(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(LAE, self).__init__()
        self.x_dim = x_dim
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim) # V
        self.fc2 = nn.Linear(h_dim, z_dim) # d_i
        
        # decoder part
        self.fc3 = nn.Linear(z_dim, x_dim) # u
        
    def encoder(self, x):
        h = self.fc1(x)
        z = self.fc2(h)
        return z
    
    def decoder(self, z):
        xhat = self.fc3(z)
        return xhat
    
    def forward(self, x):
        z = self.encoder(x.view(-1, self.x_dim))
        xhat = self.decoder(z)
        return xhat, None, None

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, self.x_dim))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

def get_weights(X, gauss_coef=1.0, neighbors=None):
    ''' 
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

def sparse_D(n,p):
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
    return sparse_mx_to_torch_sparse_tensor(D)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    '''
    Convert a scipy sparse matrix to a torch sparse tensor.
    '''
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def convclust_penalty(recons, weights, wasserstein=True, q=2):
    '''
    Computes the differences between all rows of the output.
    :return: (Tensor)
    '''
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n,p = recons.shape        
    D = sparse_D(n,p).to(device)
    if wasserstein:
        recons = torch.sort(recons)[0]
    diffs = torch.norm(D.matmul(recons), q, dim=1)
    return torch.norm(torch.mul(weights, diffs), 1)

def loss_function(recon_x, x, mu, log_var, weights, cc_target, cc_lambda=0.1, cc_metric='Eulerian',  loss='BCE'):
    '''
    Loss function for VAE with convex clustering penalty on the reconstructions.
    '''
    # compute binary cross-entropy loss on reconstruction and original data
    x_dim = x[1].flatten().shape[0]
    if cc_metric=='Lagrangian':
        if loss == 'BCE':
            recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, x_dim), reduction='sum')
            recon_loss += torch.norm(torch.sort(recon_x)[0]-torch.sort(x.view(-1, x_dim))[0], 2)
        else:
            recon_loss = F.mse_loss(recon_x, x.view(-1, x_dim), reduction='sum')
    else:
        if loss == 'BCE':
            recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, x_dim), reduction='sum')
        else:
            recon_loss = F.mse_loss(recon_x, x.view(-1, x_dim), reduction='sum')
            
    # compute KL divergence loss in latent space
    if mu is not None:
        z_reg_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    else:
        z_reg_loss = 0.0
    # compute convex clustering loss
    if cc_metric == 'Eulerian':
        convclust_loss = convclust_penalty(cc_target, weights, wasserstein=False)
    elif cc_metric == 'Lagrangian':
        convclust_loss = convclust_penalty(cc_target, weights, wasserstein=True)
    elif cc_metric == 'Mixed':
        convclust_loss = convclust_penalty(cc_target, weights, wasserstein=False)
        convclust_loss += convclust_penalty(cc_target, weights, wasserstein=True)
        convclust_loss /= 2.0
    else: 
        convclust_loss = 0.0
    return recon_loss + z_reg_loss + cc_lambda*convclust_loss

#-------------- Training and testing functions -------------#

def l2_ball_proj(x,dim=0):
    if dim == 0:
        return torch.divide(x,torch.norm(x,dim=dim))
    elif dim == 1:
        return torch.divide(x.T,torch.norm(x,dim=dim).T).T
    else:
        return torch.divide(x,torch.norm(x,dim=None))
    
def train_vae(vae, optimizer, train_loader, epoch, cc_lambda=0.1, cc_target='recon', cc_metric='Lagrangian', loss='BCE'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae.train()
    train_loss = 0
    for batch_idx, data  in enumerate(train_loader):
        data = data[0]
        data = data.to(device) #cuda()
        weights = torch.from_numpy(get_weights(data.detach().cpu().numpy().reshape(data.shape[0], \
                                                                             np.prod(data.shape[1::])))).to(device) 
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        if cc_target == 'recon':
            loss = loss_function(recon_batch, data, mu, log_var, weights, recon_batch, cc_lambda, cc_metric, loss=loss)
        else:
            V = vae.fc1.weight.data
            loss = loss_function(recon_batch, data, mu, log_var, weights, V, cc_lambda, cc_metric, loss=loss)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        with torch.no_grad():
            vae.fc1.weight.data = l2_ball_proj(vae.fc1.weight.data, dim=1)
            vae.fc31.weight.data = l2_ball_proj(vae.fc31.weight.data, dim=0)
            vae.fc32.weight.data = l2_ball_proj(vae.fc32.weight.data, dim=0)
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Convclust lambda: {}  Average loss: {:.4f}'.format(epoch, cc_lambda,\
                                                                       train_loss / len(train_loader.dataset)))

def test_vae(vae, test_loader, loss='BCE'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device) #cuda()
            weights = torch.from_numpy(get_weights(data.detach().cpu().numpy().reshape(data.shape[0], \
                                                                                 np.prod(data.shape[1::])))).to(device) 
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var, weights, loss=loss).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return recon, data

def train_path_vae(vae, optimizer, train_loader, test_loader, lambda_path, cc_metric='Lagrangian', 
                   epochs_per_lambda=2, savepath='./models/', bs=100, loss='BCE'):
    # unique hash path for this model
    savepath += secrets.token_hex(nbytes=16) + '/' 
    
    # Create savepath                                                                                              
    path_exists = os.path.exists(savepath)
    if not path_exists:
        os.makedirs(savepath)
        print('Created folder', savepath, 'for saving model path.')
                  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, lambd in enumerate(lambda_path): 
        epoch = 1
        for epoch in range(1, epochs_per_lambda+1):
            train_vae(vae, optimizer, train_loader, epoch, cc_lambda=lambd, cc_metric=cc_metric, loss=loss)

        # save model
        torch.save(vae, savepath+'lambda_'+str(lambd)+'.pt')
        
        test_recon, test_data = test_vae(vae, test_loader, loss=loss)
        with torch.no_grad():
            torch.manual_seed(7)
            z = torch.randn(64, bs).to(device) #cuda()
            sample = vae.decoder(z).to(device) #cuda()
            save_image(sample.view(64, 1, 28, 28), savepath+'sample_' +str(lambd)+ '.png')
            save_image(test_recon[0:64,:].view(64, 1, 28, 28), savepath+'reconstruction_' +str(lambd)+ '.png')
            
    return savepath

if __name__ == '__main__':
    bs = 100
    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    # build model
    vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=256)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae.to(device) #cuda()

    # Choose optimizer and set learning parameters
    optimizer = optim.AdamW(vae.parameters(), lr=0.001, weight_decay=0.001)
    optimizer.zero_grad()
    
    for epoch in range(1, 51):
        train_vae(vae, optimizer, train_loader, epoch)
        test_recon, test_data = test_vae(vae, test_loader)

    # Check whether the path for exists or not
    # if not, create one.
    import os
    path = './samples/'
    path_exists = os.path.exists(path)
    if not path_exists:
        os.makedirs(path)
        print("Created folder for saving samples:",path)

    # Save out some samples
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        torch.manual_seed(7)
        z = torch.randn(64, 256).to(device) #cuda()
        sample = vae.decoder(z).to(device) #cuda()
        save_image(sample.view(64, 1, 28, 28), './samples/sample_' + '.png')
    save_image(test_recon[0:64,:].view(64, 1, 28, 28), './samples/reconstruction_' + '.png')
    save_image(test_data[0:64,:].view(64, 1, 28, 28), './samples/true_data_' + '.png')
    
