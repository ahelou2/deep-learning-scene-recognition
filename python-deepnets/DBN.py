from numpy import *
import math
from rbm import *

DEBUG = 0

def normalize(data, mu=None, sigma=None):
    '''
	data normalization function
	data : 2D array, each row is one data
	mu   : 1D array, each element is the mean the corresponding column in data
	sigma: 1D array, each element is the standard deviation of the corresponding
	       column in data
	'''
    (m, n) = data.shape
    if mu is None or sigma is None:
        sigma = ones(n)
        mu = mean(data,0)
        dev = std(data,0)
        (x,) = nonzero(dev)
        sigma[x] = dev[x]

        mu_rep = tile(mu, (m, 1))
        sigma_rep = tile(sigma, (m, 1))
        return (data - mu_rep)/sigma_rep, mu, sigma
    else:
        mu_rep = tile(mu, (m, 1))
        sigma_rep = tile(sigma, (m, 1))
        return (data - mu_rep)/sigma_rep

def un_normalize(data, mu, sigma):
    '''
    un-normalize the normalized data. This is used for visualization purpose
   	data : 2D array, each row is one data
	mu   : 1D array, each element is the mean the corresponding column in data
	sigma: 1D array, each element is the standard deviation of the corresponding
	       column in data
    '''
    (m, n) = data.shape
    mu_rep = tile(mu, (m, 1))
    sigma_rep = tile(sigma, (m, 1))
    return multiply(data,sigma_rep) + mu_rep

def flattenFeatures(features):
    
    a = array(features[0])
    n = len(features)
    for i in range(1, n):
        a = append(a, features[i], axis=1)
    return a

class DBN(object):
    '''
    Deep belief net, trains using regular RBM
    '''
    def __init__(self, num_hid_layers, hid_layer_sizes, momentums, sparsities, traindata, trainlabel):
        self.n_layers = num_hid_layers
        self.data = array(traindata)
        self.label = array(trainlabel)

        num_vis = len(self.data[0])
		
        self.trainers = []
		
		#hidden layers
        for i in range(self.n_layers):
            #gaussian visible unit only for the first layer
            rbm = RBM(num_vis, hid_layer_sizes[i], i>0)
            trainer = RBM.Trainer(rbm, momentums[i], sparsities[i])
            self.trainers.append(trainer)
            num_vis = hid_layer_sizes[i]
		
		#default batch sizes is the length of the data set
        self.batch_sizes = [len(self.data)]

    def reset(self):
        #hidden layers
        for i in range(self.n_layers):
            #gaussian visible unit only for the first layer
            rbm = RBM(num_vis, hid_layer_sizes[i], i>0)
            trainer = RBM.Trainer(rbm, momentums[i], sparsities[i])
            self.trainers.append(trainer)
            num_vis = hid_layer_sizes[i]
        
		#default batch sizes is the length of the data set
        self.batch_sizes = [len(self.data)]	
	
    def set_batches(self, batch_size):
        '''
		batch_size: an integer, size of each training batch
        '''
        m = len(self.data)
        num_batches = int(math.ceil(m / batch_size))
        self.num_batches = num_batches
        self.bsize = int(m / num_batches)
        self.batch_sizes = zeros(num_batches, 'i') 
        for i in range(num_batches):
            i1 = i*self.bsize
            i2 = min(m, i1+self.bsize)
            self.batch_sizes[i] = i2 - i1
        print 'There are '+str(self.num_batches)+' batches'

    def hidden_layer_train(self, idx, data, l2_reg, alpha, tau, epoch):
        '''
		training each hidden layer which is a RBM
		idx: index of the current layer
		data: data from the layer below this layer
		l2_reg: l2 reg of this layer
		alpha: learning rate
		tau: (??) is used to reduce learning rate
		epoch: number of iterations
		'''
        for e in range(epoch):
            for i in range(self.num_batches):
                #this is what i got from test_rbm.py
                lr = alpha * exp(-i/tau)
                #learning, the weights and biases already updated in this function
                self.trainers[idx].learn(data[i*self.bsize:i*self.bsize+self.batch_sizes[i]], lr, l2_reg)
            recons = self.trainers[idx].rbm.reconstruct(data , 2)
            error = sum(power(data - recons, 2))
            print 'Error of epoch '+str(e+1)+'/'+str(epoch)+': '+str(error)

    def unsupervised_train(self, l2_regs, alphas, taus, maxepochs):
        '''
        training the deep belief net
        l2_regs: list of l2 regs
        alphas: list of all learning rates
        taus: list of all taus
        maxepochs: list of all maxepoch
        the size of these lists must be equal to the number of hidden layers
        '''	
        #randomly permute data before training
        idx = random.permutation(len(self.data))
        self.data = self.data[idx]
        self.label = self.label[idx]

        data = array(self.data)
        for i in range(self.n_layers):
            if DEBUG == 1:
                print 'Training layer '+str(i+1)+' with '+str(maxepochs[i])+' epochs'
            self.hidden_layer_train(i, data, l2_regs[i], alphas[i], taus[i], maxepochs[i])
            data = self.trainers[i].rbm.hidden_expectation(data)            
	
    def getFeatures(self, data=[]):
        '''
		returns the features of the data in all layers
		by default, this function returns features of training data
        '''
        features = []
        if data == []:
           data = self.data
        
        d = array(data)
        for i in range(self.n_layers):
            d = self.trainers[i].rbm.hidden_expectation(d)
            features.append(d)	
	    return features
		
    def getReconsError(self, data, l):
        '''
        returns the reconstructed data upto hidden layer l
        and the error with original data
        by default, the function returns the reconstructed training data
        using all hidden layers
        '''
        d = array(data)
		#go up to layer l
        for i in range(l):
            d = self.trainers[i].rbm.hidden_expectation(d)
		#then go down to layer 0
        for i in range(l-1,-1,-1):
            d = self.trainers[i].rbm.visible_expectation(d)
        dims = len(d.shape)
        err = sum(power(data - d,2))
        return (d, err)
   
    def getAllLayersReconsError(self, data=[]):
        recons = []   
        errors = []
        if data == []:
            data = self.data
        for i in range(self.n_layers):
            (recon, error) = self.getReconsError(data, i)
            recons.append(recon)
            errors.append(error)
        return recons, errors

class ConvDBN(DBN):
    '''
    this is a subclass of DBN, and trains using Convolutional RBM
    '''
    def __init__(self, num_hid_layers, num_filters, filter_shapes, pool_shapes, momentums, sparsities, traindata, trainlabel):
        self.n_layers = num_hid_layers
        self.data = array(traindata)
        self.label = array(trainlabel)

        self.trainers = []
        for i in range(self.n_layers):
            #gaussian visible unit only for the first layer
            rbm = Convolutional(num_filters[i], filter_shapes[i], pool_shapes[i], i>0)
            trainer = Convolutional.Trainer(rbm, momentums[i], sparsities[i])
            self.trainers.append(trainer)
		
		#default batch sizes is the length of the data set
        self.batch_sizes = [len(self.data)]
        
    def reset(self):
        self.trainers = []
        for i in range(self.n_layers):
            #gaussian visible unit only for the first layer
            rbm = Convolutional(num_filters[i], filter_shapes[i], pool_shapes[i], i>0)
            trainer = Convolutional.Trainer(rbm, momentums[i], sparsities[i])
            self.trainers.append(trainer)
		
		#default batch sizes is the length of the data set
        self.batch_sizes = [len(self.data)]

    def unsupervised_train(self, l2_regs, alphas, taus, maxepochs):
        '''
        training the deep belief net
        l2_regs: list of l2 regs
        alphas: list of all learning rates
        taus: list of all taus
        maxepochs: list of all maxepoch
        the size of these lists must be equal to the number of hidden layers
        '''	
        #randomly permute data before training
        idx = random.permutation(len(self.data))
        self.data = self.data[idx]
        self.label = self.label[idx]

        data = array(self.data)
        for i in range(self.n_layers):
            if DEBUG == 1:
                print 'Training layer '+str(i+1)+' with '+str(maxepochs[i])+' epochs'
            self.hidden_layer_train(i, data, l2_regs[i], alphas[i], taus[i], maxepochs[i])
            data = self.trainers[i].rbm.pooled_expectation(data)            

    def getFeatures(self, data=[]):
        '''
		returns the features of the data in all layers
		by default, this function returns features of training data
        '''
        features = []
        if data == []:
           data = self.data
        
        d = array(data)
        for i in range(self.n_layers):
            d = self.trainers[i].rbm.pooled_expectation(d)
            features.append(d)	
	    return features

    def getReconsError(self, data, l):
        '''
        returns the reconstructed data upto hidden layer l
        and the error with original data
        by default, the function returns the reconstructed training data
        using all hidden layers
        '''
        d = array(data)
		#go up to layer l
        for i in range(l):
            d = self.trainers[i].rbm.pooled_expectation(d)
		#then go down to layer 0
        for i in range(l-1,-1,-1):
            d = self.trainers[i].rbm.visible_expectation(d)
        dims = len(d.shape)
        err = sum(power(data - d,2))
        return (d, err)
