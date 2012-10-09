import sys
from DBN import *
from scipy.io import *

l2_reg_default = 0.01
alpha_default = 0.05
tau_default = 1
maxepoch_default = 50

def writeToFile(filename, data, label):
    '''
    write data features and labels to a file
    this is used to run svm-multiclass
    '''
    f = open(filename, 'w')
    assert len(data) == len(label)
    (n,m) = data.shape
    for i in range(n):
        st = str(int(label[i]))
        for j in range(m):
            st = st+' '+str(j)+':'+str(data[i,j])
        st = st + '\n'
        f.write(st)
    f.close()

def trainingOption(dbn, num_hid_layers, traindata, trainlabel, testdata, testlabel, training_option_file, prefix):
    g = open(training_option_file,'r')
    m = int(g.readline())
    prefix = prefix + training_option_file
    for j in range(m):
       s = g.readline()
       opts = s.split(';')
       batch_size = int(opts[0])
       #l2_regs 
       l2_regs = [l2_reg_default] * num_hid_layers
       s = opts[1].split(',')
       for i in range(num_hid_layers):
           l2_regs[i] = float(s[i])
       #alphas (learning rate)
       alphas = [alpha_default] * num_hid_layers
       s = opts[2].split(',')
       for i in range(num_hid_layers):
           alphas[i] = float(s[i])
       #taus (penalty??)
       taus = [tau_default] * num_hid_layers
       s = opts[3].split(',')
       for i in range(num_hid_layers):
           taus[i] = float(s[i])
       #maxepochs
       maxepochs = [maxepoch_default]* num_hid_layers
       s = opts[4].split(',')
       for i in range(num_hid_layers):
           maxepochs[i] = float(s[i])
       
       #set batch size and training
       dbn.set_batches(batch_size)
       dbn.unsupervised_train(l2_regs, alphas, taus, maxepochs)
       
       #obtaining the results and write to mat file
       train_features = dbn.getFeatures()
       test_features = dbn.getFeatures(testdata)
       train_recons, train_errors = dbn.getAllLayersReconsError()
       test_recons, test_errors = dbn.getAllLayersReconsError(testdata)
       writeTofile(prefix+'_'+str(j)+'_traindata.txt', flattenFeatures(train_features), trainlabel)
       writeTofile(prefix+'_'+str(j)+'_testdata.txt', flattenFeatures(test_features), testlabel)
       mdict={'train_features': train_features, 'test_features': test_features, 'train_recons': train_recons, 'train_errors': train_errors, 'test_recons': test_recons, 'test_errors': test_errors}
       savemat(prefix+'_'+str(j)+'_experiment.mat', mdict)
       
       # reset dbn for next training options set
       dbn.reset()
    g.close()

def createDBN(traindata, trainlabel, testdata, testlabel, parameter_file):
    f = open(parameter_file, 'r')
    n = int(f.readline())
    for k in range(n):
        s = f.readline()
        args = s.split(';')
        num_hid_layer = int(args[0])
        
        #hidden layer sizes
        s = args[1].split(',')
        hid_layer_sizes = []
        for i in range(num_hid_layers):
            hid_layer_sizes.append(int(s[i]))

        #momentums
        s = args[2].split(',')
        momentums = []
        for i in range(num_hid_layers):
            momentums.append(float(s[i]))

        #sparsities
        s = args[3].split(',')
        sparsities = [None] * num_hid_layer
        for i in range(num_hid_layers):
            sparsities[i] = float(s[i])

        dbn = DBN(num_hid_layer, hid_layer_sizes, momentums, sparsities, traindata, trainlabel)
        trainingOption(dbn, num_hid_layers, traindata, trainlabel, testdata, testlabel, args[4], 'regRBM_'+parameter_file+'_')        
    f.close()
    
def createConvDBN(traindata, trainlabel, testdata, testlabel, parameter_file, train_option_file):
    f = open(parameter_file, 'r')
    n = int(f.readline())
    for k in range(n):
        s = f.readline()
        args = s.split(';')
        num_hid_layer = int(args[0])
        
        #num filters
        s = args[1].split(',')
        num_filters = []
        for i in range(num_hid_layers):
            num_filters.append(int(s[i]))
            
        #filter shapes
        s = args[2].split(',')
        filter_shapes = []
        for i in range(0,2*num_hid_layers,2):
            filter_shapes.append( (int(s[i]),int(s[i+1])) )

        #pool shapes
        s = args[3].split(',')
        pool_shapes = []
        for i in range(0,2*num_hid_layers,2):
            pool_shapes.append( (int(s[i]),int(s[i+1])) )

        #momentums
        s = args[4].split(',')
        momentums = []
        for i in range(num_hid_layers):
            momentums.append(float(s[i]))
            
        #sparsities
        s = args[5].split(',')
        sparsities = [None] * num_hid_layer
        for i in range(num_hid_layers):
            sparsities[i] = float(s[i])

        dbn = ConvDBN(num_hid_layers, num_filters, filter_shapes, pool_shapes, momentums, sparsities, traindata, trainlabel)
        trainingOption(dbn, num_hid_layers, traindata, trainlabel, testdata, testlabel, args[6], 'convRBM_'+parameter_file+'_')        
    f.close()
    
if __name__ == '__main__':
    ''' arguments:
      (1) .mat data file contains everything (train data, train label, test data, test label)
      (2) name of train data variable in data file
      (3) name of train label variable in data file
      (4) name of test data variable in data file
      (5) name of test label variable in data file
      (6) Type of Deep belief network (0 for regular, 1 for convolutional)
      (7) DBN parameter files: 
            first line: number parameter sets
            next line: <num hidden layers>;<hidden layer sizes>;<momentums>;<sparsities>;<train_option_file> (if regular DBM)
                or     <num hidden layers>;<num filters>;<filter shapes>;<pool shapes>;<momentums>;<sparsities>;<train_option_file> (if regular ConvDBM)
          Train option files:
            first line: number of option sets
            next lines: <batch size>;<l2_regs>,<alphas>;<taus>;<maxepochs>
    '''
    traindata = []
    trainlabel = []
    testdata = []
    testlabel = []
    mdict = {sys.argv[2]:traindata, sys.argv[3]: trainlabel, sys.argv[4]:testdata,sys.argv[5]:testlabel }
    d = loadmat(sys.argv[1], mdict)
    traindata = d[sys.argv[2]]
    trainlabel = d[sys.argv[3]]
    testdata = d[sys.argv[4]]
    testlabel = d[sys.argv[5]]
    d = None
    traindata, mu, sigma = normalize(traindata)
    testdata= normalize(testdata, mu, sigma)

    dbn_type = int(sys.argv[6])
    parameter_file = sys.argv[7]
    if dbn_type == 0 :
        createDBN(traindata, trainlabel, testdata, testlabel, parameter_file)
    else:
        createConvDBN(traindata, trainlabel, testdata, testlabel, parameter_file)
