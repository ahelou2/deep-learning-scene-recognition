from training import *
import sys

if __name__ == '__main__':
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
    dbn = DBN(3, [500, 500, 1000], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], traindata, trainlabel)
    dbn.set_batches(25)
    dbn.unsupervised_train([0.0001,0.0001,0.0001], [0.005, 0.05, 0.05], [0.0002, 0.0002, 0.0002], [50, 50, 50])
    train_features = dbn.getFeatures()
    test_features = dbn.getFeatures(testdata)
    train_recons, train_errors = dbn.getAllLayersReconsError()
    test_recons, test_errors = dbn.getAllLayersReconsError(testdata)
    writeToFile('traindata.txt', flattenFeatures(train_features), trainlabel)
    writeToFile('testdata.txt', flattenFeatures(test_features), testlabel)
    mdict={'train_features': train_features, 'test_features': test_features, 'train_recons': train_recons, 'train_errors': train_errors, 'test_recons': test_recons, 'test_errors': test_errors}
    savemat('experiment.mat', mdict)
       
