import os
import numpy as np
import random as rd
from keras.utils.np_utils import to_categorical

base_dir = os.path.dirname(os.path.realpath(__file__))
# data_dir = os.path.join(base_dir,'data')

def train_data(str):
    datas= []
    with open(os.path.join(base_dir,str)) as file:
        file.readline()
        for line_id,line in enumerate(file):
            label, feat= line.split(',')
            feat = np.fromstring(feat,dtype=int,sep=' ')
            feat = np.reshape(feat,(48,48,1))
            datas.append((feat,int(label),line_id))
    feats,labels,line_ids = zip(*datas)
    feats = np.asarray(feats)
    labels = to_categorical(np.asarray(labels,dtype=np.int32))
    return feats,labels,line_ids

def test_data(str):
    datas= []
    with open(os.path.join(base_dir,str)) as file:
        file.readline()
        for line_id,line in enumerate(file):
            _, feat= line.split(',')
            feat = np.fromstring(feat,dtype=int,sep=' ')
            feat = np.reshape(feat,(48,48,1))
            datas.append(feat)
    feats = datas
    feats = np.asarray(feats)
    return feats

def train_split_data(str,seed, val_size):
    x_val= []
    y_val= []
    x_train= []
    y_train= []
    datas= []
    rd.seed(seed)
    valid_set= rd.sample(range(28709), val_size)
    train_set= list(set(valid_set)^set(range(28709)))
    with open(os.path.join(base_dir,str)) as file:
        file.readline()
        for line_id,line in enumerate(file):
            label, feat= line.split(',')
            feat = np.fromstring(feat,dtype=int,sep=' ')
            feat = np.reshape(feat,(48,48,1))
            datas.append((feat,int(label),line_id))
    feats,labels,line_ids = zip(*datas)
    labels = to_categorical(np.asarray(labels,dtype=np.int32))
    feats = np.asarray(feats)

    for i in valid_set:
        y_val.append(labels[i])
        x_val.append(feats[i])
    for i in train_set:
        y_train.append(labels[i])
        x_train.append(feats[i])
    x_val= np.asarray(x_val)
    x_train= np.asarray(x_train)

    return x_train, y_train, x_val, y_val







# def read_dataset(mode='trainsplit',isFeat=True):
#     """
#     Return:
#         # features: (int. list) list
#         # labels: int32 2D array
#         data_ids: int. list
#     """
#     # num_data = 0
#     datas = []

#     with open(os.path.join(data_dir,'{}.csv'.format(mode))) as file:
#         file.readline()
#         for line_id,line in enumerate(file, 1):
#             # print(line)
#             if isFeat:
#                 label, feat=line.split(',')
#             else:
#                 _,feat = line.split(',')
#             feat = np.fromstring(feat,dtype=int,sep=' ')
#             # print(feat)
#             feat = np.reshape(feat,(48,48,1))

#             if isFeat:
#                 datas.append((feat,int(label),line_id))
#             else:
#                 datas.append(feat)

#     # random.shuffle(datas)  # shuffle outside
#     if isFeat:
#         feats,labels,line_ids = zip(*datas)
#     else:
#         feats = datas
#     feats = np.asarray(feats)
#     if isFeat:
#         # print(labels[0])
#         # labels = to_categorical(np.asarray(labels,dtype=np.int32))
#         # print(labels[0])
#         return feats,labels,line_ids
#     else:
#         return feats


# def dump_history(store_path,logs):
#     with open(os.path.join(store_path,'train_loss'),'a') as f:
#         for loss in logs.tr_losses:
#             f.write('{}\n'.format(loss))
#     with open(os.path.join(store_path,'train_accuracy'),'a') as f:
#         for acc in logs.tr_accs:
#             f.write('{}\n'.format(acc))
#     with open(os.path.join(store_path,'valid_loss'),'a') as f:
#         for loss in logs.val_losses:
#             f.write('{}\n'.format(loss))
#     with open(os.path.join(store_path,'valid_accuracy'),'a') as f:
#         for acc in logs.val_accs:
#             f.write('{}\n'.format(acc))