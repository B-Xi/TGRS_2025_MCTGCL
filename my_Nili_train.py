import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import get_cls_map
import time
import modelStatsRecord
from sklearn import metrics
from mctgcl  import mctgcl
from GCN_model import *
from sklearn.neighbors import kneighbors_graph
import  supervised_contrastive_loss
def loadData():
    # 读入数据
    data = sio.loadmat('/media/xd/hdd/ZY/reference/DCFSL-2021-main/datasets/NiLiFossae/NiliFossae.mat')['NiliFossae']
    labels = sio.loadmat('/media/xd/hdd/ZY/reference/DCFSL-2021-main/datasets/NiLiFossae/NiliFossae_gt_final.mat')['NiliFossae_gt']

    return data, labels

# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):

    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
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
    train_indices = np.zeros_like(y)
    test_indices = np.zeros_like(y)+1
    for i in range(0, 9):
        indices = np.argwhere(y == i)
        #print(len(indices))
        np.random.shuffle(indices)
        selected_indices = indices[:testRatio]
        train_indices[selected_indices]=1
        test_indices[selected_indices]=0
    train_indices=np.argwhere(train_indices==1)
    test_indices=np.argwhere(test_indices==1)
    X_train=X[train_indices,:,:,:]
    X_test=X[test_indices,:,:,:]
    y_train=y[train_indices]
    y_test=y[test_indices]
    y_train=y_train.squeeze(1)
    y_test=y_test.squeeze(1)
    #print(train_indices)

    '''
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)
    '''
    return X_train, X_test, y_train, y_test

BATCH_SIZE_TRAIN = 64

def create_data_loader():
    X, y = loadData()
    test_ratio = 10
    patch_size = 13
    pca_components = 30

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    X = TestDS(X, y_all)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=0,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=False,
                                                num_workers=0,
                                              )
    data_labeled_loader=torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=test_ratio*9,
                                               shuffle=False,
                                               num_workers=0,
                                               )
    return train_loader, test_loader, all_data_loader, data_labeled_loader,y

""" Training dataset"""

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

""" Testing dataset"""

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

def aff_to_adj(last_layer_data_src):
    last_layer_data_src = F.normalize(last_layer_data_src, dim=-1)
    features1 = last_layer_data_src.cpu().detach().numpy()

    adj_nei = kneighbors_graph(features1, 10, mode='distance')
    adj_nei = adj_nei.A
    sigam=1
    for i in range(adj_nei.shape[0]):
        for j in range(adj_nei.shape[1]):
            if adj_nei[i][j] != 0:
                adj_nei[i][j] = np.exp(-adj_nei[i][j]/(sigam*sigam))
    adj_d = np.sum(adj_nei,axis=1, keepdims=True)
    adj_d = np.diag(np.squeeze(adj_d**(-0.5)))
    adj_w = np.matmul(adj_nei,adj_d)
    adj_w = np.matmul(adj_d,adj_w)
    adj_nei = adj_w+np.eye(adj_w.shape[0])
    adj_nei = torch.from_numpy(adj_nei).cuda(1).to(torch.float32)
    return adj_nei

temperature = 1
a=0.5
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
def train(train_loader, data_labeled_loader,epochs):

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 网络放到GPU上
    net = mctgcl(num_classes=9, num_tokens=121).to(device)

    src_gcn_module = GCN_M(nfeat=128,
                            nhid=128,
                            nclass=1,
                            dropout=0.3).to(device)

    src_optim = optim.Adam(src_gcn_module.parameters(), lr=0.001)

    
    print(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 开始训练
    total_loss = 0
    for epoch in range(epochs):
        net.train()
        data_labeled=next(iter(data_labeled_loader))
        data_all,target_all=data_labeled[0].to(device),data_labeled[1]
   
        data_all_aug=torch.flip(data_all.clone().permute(0,1,2,4,3), dims=[3])

        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            outputs,_ = net(data)
            # GCN_model
            #计算每个类的均值
            outputs_all,features_all=net(data_all)
            src_adj_nei = aff_to_adj(features_all).to(device)
            outputs_src = src_gcn_module(features_all,  src_adj_nei)
            outputs_src = F.normalize(outputs_src, dim=-1)

            outputs_all_aug,features_all_aug=net(data_all_aug)
            tar_adj_nei = aff_to_adj(features_all_aug).to(device)
            outputs_tar = src_gcn_module(features_all_aug, tar_adj_nei)
            outputs_tar = F.normalize(outputs_tar, dim=-1)

            contrastiveLoss = supervised_contrastive_loss.SupConLoss(temperature)
            # 遍历查找每个数值 i 的位置
            out_class_list = []
            out_class2_list = []
            for i in range(9):  # i 从 0 到 8
                indices = torch.nonzero(target_all == i).squeeze()  # 使用 torch.nonzero() 找到所有等于 i 的索引
                out_class=outputs_tar[indices,:].mean(dim=0)
                out_class2=outputs_src[indices,:].mean(dim=0)
                out_class_list.append(out_class)
                out_class2_list.append(out_class2)
            out_class_tensor = torch.stack(out_class_list, dim=0)
            out_class2_tensor = torch.stack(out_class2_list, dim=0)
            features_class=torch.cat((out_class_tensor,out_class2_tensor),dim=0)
            features_class=features_class.unsqueeze(1)
            labels=torch.arange(9)
            labels_class=torch.cat((labels,labels),dim=0)
            f_contrastive_loss=contrastiveLoss(features_class,labels_class)
            #print(f_contrastive_loss)
            # 计算损失函数
            loss = criterion(outputs, target)+a*f_contrastive_loss
            # 优化器梯度归零
            optimizer.zero_grad()
            src_optim.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            src_optim.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f] [current c_loss: %.4f]  [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         f_contrastive_loss.item(),
                                                                         loss.item()))

    print('Finished Training')
    print(get_parameter_number(net),get_parameter_number(src_gcn_module))
    return net, device

def test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    features=0
    features = np.zeros([1,9])
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs,_ = net(inputs)
        features=np.append(features, outputs.detach().cpu().numpy(),axis=0)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))
    features=features[1:,:]
    sio.savemat('feature/test_features_all_Nili.mat', {'test_features_all': features})
    sio.savemat('feature/labels_Nili.mat', {'labels': y_test})
    print(features.shape,y_test.shape)
    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    target_names = ['1', '2', '3', '4'
        , '5', '6','7'
        , '8', '9']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

if __name__ == '__main__':
    nDataSet = 1
    CLASS_NUM=9
    acc = np.zeros([nDataSet, 1])
    A = np.zeros([nDataSet, CLASS_NUM])
    P = np.zeros([nDataSet, CLASS_NUM])
    k = np.zeros([nDataSet, 1])
    training_time = np.zeros([nDataSet, 1])
    test_time = np.zeros([nDataSet, 1])
    best_predict_all = []

    seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]
    for iDataSet  in range(nDataSet):
        torch.manual_seed(seeds[iDataSet])
        torch.cuda.manual_seed_all(seeds[iDataSet])
        import random
        random.seed(seeds[iDataSet])
        np.random.seed(seeds[iDataSet])
        train_loader, test_loader, all_data_loader,data_labeled_loader,y_all= create_data_loader()
        tic1 = time.perf_counter()
        net, device = train(train_loader,data_labeled_loader, epochs=100)
        # 只保存模型参数
        torch.save(net.state_dict(), 'cls_params/MCTGCL_params.pth')
        toc1 = time.perf_counter()
        tic2 = time.perf_counter()
        y_pred_test, y_test = test(device, net, test_loader)
        toc2 = time.perf_counter()
        # 评价指标
        classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
        classification = str(classification)
        Training_Time = toc1 - tic1
        Test_time = toc2 - tic2
        print(each_acc.shape)
        acc[iDataSet] = oa/100
        C = metrics.confusion_matrix(y_test, y_pred_test)
        A[iDataSet, :] = each_acc/100
        P[iDataSet, :] = each_acc/100
    
        k[iDataSet] = metrics.cohen_kappa_score(y_test, y_pred_test)
        training_time[iDataSet] = Training_Time
        test_time[iDataSet] = Test_time
        
        get_cls_map.get_cls_map(net, device, all_data_loader, y_all,oa)




    ELEMENT_ACC_RES_SS4 = np.transpose(A)
    AA_RES_SS4 = np.mean(ELEMENT_ACC_RES_SS4,0)
    OA_RES_SS4 = np.transpose(acc)
    KAPPA_RES_SS4 = np.transpose(k)
    ELEMENT_PRE_RES_SS4 = np.transpose(P)
    AP_RES_SS4= np.mean(ELEMENT_PRE_RES_SS4,0)
    TRAINING_TIME_RES_SS4 = np.transpose(training_time)
    TESTING_TIME_RES_SS4 = np.transpose(test_time)
    classes_num = CLASS_NUM
    ITER = nDataSet

    modelStatsRecord.outputRecord(ELEMENT_ACC_RES_SS4, AA_RES_SS4, OA_RES_SS4, KAPPA_RES_SS4,
                              ELEMENT_PRE_RES_SS4, AP_RES_SS4,
                              TRAINING_TIME_RES_SS4, TESTING_TIME_RES_SS4,
                              classes_num, ITER,
                              './records/p_{}_10_Nili_result_train_iter_times_{}shot_CRU_Chikusei_iter_10_true_knn_{}_{}.txt'.format(13,10,temperature,a))
