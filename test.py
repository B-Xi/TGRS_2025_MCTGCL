import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
from operator import truediv
import MCTGCL.mctgcl as mctgcl
import time
from tqdm import tqdm

data_path = os.path.join(os.getcwd(),'')
X = sio.loadmat(os.path.join(data_path, './data/PaviaU.mat'))['paviaU']
y = sio.loadmat(os.path.join(data_path, './data/PaviaU_gt.mat'))['paviaU_gt']
test_ratio = 0.995
patch_size = 13
num_classes = 9



num_tokens = (patch_size - 4) ** 2
batch_size = 32

# PCA bands number
pca_components = 30

# PCA
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

# padding operation
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX


# extract patch
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    
    return X_train, X_test, y_train, y_test


def create_data_loader(X, y, patch_size):
    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA transformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)

    print('\n... ... create train & test data ... ...')
    X_train, X_test, y_train, y_test = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('X_train shape: ', X_train.shape)
    print('X_test shape: ', X_test.shape)

    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    X_train = X_train.reshape(-1, patch_size, patch_size, pca_components, 1)
    X_test = X_test.reshape(-1, patch_size, patch_size, pca_components, 1)
    
    X = X.transpose(0, 4, 3, 1, 2)
    X_train = X_train.transpose(0, 4, 3, 1, 2)
    X_test = X_test.transpose(0, 4, 3, 1, 2)

    print('after transpose: X shape: ', X.shape)
    print('after transpose: Xtrain shape: ', X_train.shape)
    print('after transpose: Xtest  shape: ', X_test.shape)

    X = TestDS(X, y_all)
    trainset = TrainDS(X_train, y_train)
    testset = TestDS(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    all_data_loader = torch.utils.data.DataLoader(
        dataset=X,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader, all_data_loader, y


class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return self.len   
    
class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len

def test(device, net, test_loader):
    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0
    for item in tqdm(test_loader, desc="Test Process"):
        data, label = item
        data = data.to(device)
        pred = net(data)
        pred_class = np.argmax(pred.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = pred_class
            y_test = label
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, pred_class))
            y_test = np.concatenate((y_test, label))

    return y_pred_test, y_test


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test):
    target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                        'Self-Blocking Bricks','Shadows']
        
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100


def save_reports():
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)

    print('OA: {:.2f} (%)'.format(oa))
    print('AA: {:.2f} (%)'.format(aa))
    print('Kappa: {:.2f} (%)'.format(kappa))
    print('{}'.format(classification))

if __name__ == '__main__':
    train_loader, test_loader, all_data_loader, y_all= create_data_loader(X, y, patch_size) 

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)
    net = mctgcl.massformer(num_classes=num_classes, num_tokens=num_tokens).to(device)
    net.load_state_dict(torch.load('./params/Pavia.pth'))

    # test
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(device, net, test_loader)
    toc2 = time.perf_counter()
    test_time = toc2 - tic2
    
    save_reports()
