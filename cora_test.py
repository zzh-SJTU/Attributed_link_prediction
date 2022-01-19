# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:06:41 2022

@author: zzh
"""
import os
import networkx as nx
import pandas as pd
import numpy as np
import random
import csv
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm, trange
import pandas as pd
data_dir = os.path.expanduser("D:\link_predict\data\cora")
edgelist = pd.read_csv(os.path.join(data_dir, "cora.cites"), sep='\t', header=None, names=["target", "source"])
edgelist["label"] = "cites"
Gnx = nx.from_pandas_edgelist(edgelist, edge_attr="label")
nx.set_node_attributes(Gnx, "paper", "label")
feature_names = ["w_{}".format(ii) for ii in range(1433)]
column_names =  feature_names + ["subject"]
node_data = pd.read_csv(os.path.join(data_dir, "cora.content"), sep='\t', header=None, names=column_names)
map_subject = {
      'Case_Based':1,
		'Genetic_Algorithms':2,
		'Neural_Networks':3,
		'Probabilistic_Methods':4,
		'Reinforcement_Learning':5,
		'Rule_Learning':6,
		'Theory':7
        }
frac_of_test  = 0.5
list_edge = list(Gnx.edges)
edge_to_test_po = random.sample(Gnx.edges,int(frac_of_test*len(Gnx.edges)))
for edge in edge_to_test_po:
    Gnx.remove_edge(edge[0],edge[1])
edge_to_test_neg = []
counter = 0
while(1):
    node = random.sample(Gnx.nodes,2)
    edge = (node[0],node[1])
    if edge not in Gnx.edges:
        counter+=1
        edge_to_test_neg.append(node)
    if counter == len(edge_to_test_po):
        break
node_cnt = 0
node_dict = {}
for node in Gnx.nodes:
    node_dict[node] = node_cnt
    node_cnt += 1

dic_attri = {}
for node in Gnx.nodes:
    attribute = node_data.loc[node]
    attribue_vector  = torch.zeros(len(attribute))
    for i,attri in enumerate(attribute):
        if attri in map_subject.keys():
            attribue_vector[i] = map_subject[attri]
        else:
            attribue_vector[i] = attri
    dic_attri[node_dict[node]] = attribue_vector
print('Finish building graph')
def get_metrics(prediction, label):
    assert len(prediction) == len(label), (len(prediction), len(label))
    fpr, tpr, thresholds = metrics.roc_curve(label, prediction, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc
def softmax(x):
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum()
class node_embedding_net(nn.Module):
    def __init__(self,node_num,embedding_num,len_attribute):
        super(node_embedding_net, self).__init__()
        self.em = nn.Embedding(node_num,embedding_num)
        self.ln1 = nn.Linear(len_attribute,embedding_num)
        self.ln2 = nn.Linear(embedding_num,256)
        self.lamb1 = 0
        self.relu = nn.ReLU()
    def forward(self,targ,attribute):
        emb_node= self.em(targ)
        emb_attribute = self.ln1(attribute)
        emb_attribute = emb_attribute.reshape((1,-1))
       # emb = torch.cat((emb_node,self.lamb1*emb_attribute),dim = 1)
        emb = emb_node+self.lamb1*emb_attribute
        #emb = self.ln2(emb)
#        emb = self.relu(emb)
        return emb
class Custom_dataset(Dataset):
    def __init__(self, train_data_list):
        self.data_list = train_data_list
        self.counter=0
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx][0], self.data_list[idx][1]
    def __iter__(self):
        return iter([self.data_list[0][0], self.data_list[0][1]])
    def __next__(self):
        self.counter=self.counter+1
        return [self.data_list[self.counter][0], self.data_list[self.counter][1]]
def one_randwalk(start_point,graph,negative_num,posi_len):
    traj =[node_dict[start_point]]
    node = start_point
    for i in range(posi_len):
        list_nei = list(graph.neighbors(node))
        node = random.choice(list_nei)
        while len(list(graph.neighbors(node)))==0:
            node = random.choice(list_nei)
        traj.append(node_dict[node])
    counter = 0
    while(1):
        nega_number = random.sample(graph.nodes, 1)
        temp_edge = (start_point,nega_number[0])
        if temp_edge not in graph.edges:
            counter+=1
            traj.append(node_dict[nega_number[0]])
        if counter == negative_num:
            break
    return traj
def index2onehot(index,graph):
    array = torch.zeros(len(graph.nodes),dtype=torch.float64)
    array[index] = 1
    return array
def onehot2index(array):
    index = torch.argmax(array)
    return index    
max_len = 40
edge = []
trajecrories = []
EPOCH = 5
learning_rate = 0.01
embedding_num = 128
negative_num = 40
one_hot_dic ={}
temp = 0
counter = 0
for node in Gnx.nodes:
    if len(list(Gnx.neighbors(node)))==0 :
        continue
    for i in range(1):
        one_walk = one_randwalk(node,Gnx,negative_num,max_len)
        trajecrories.append(one_walk)
train_data = []
for traj in trajecrories:
    train_data.append((traj[0],traj[1:]))

data_set = Custom_dataset(train_data)
train_loader = DataLoader(data_set,batch_size=1,shuffle=True)
net = node_embedding_net(len(Gnx.nodes),embedding_num,len(attribute))
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
simlar = nn.CosineSimilarity()
loss_fn = nn.CrossEntropyLoss()
logsigmoid = nn.LogSigmoid()
sigmoid = nn.Sigmoid()
ex_scedule = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.9)

for i in range(EPOCH):
    with tqdm(total=len(train_loader)) as pbar:
        pbar.set_description('Processing:')
        for i,data in enumerate(train_loader):
            start,context = data
            list_context_attribute = []
            for word in context:     
                list_context_attribute.append(dic_attri[word.item()])
            context_attributes = torch.stack(list_context_attribute)
            loss = 0
            #print(start, context)
            context = torch.tensor(context)
            start = torch.tensor(start)
            start_attribute = torch.tensor(dic_attri[start.item()])
            embe_start = net(start,start_attribute)
            list_walk = []
            for i in range(max_len):
                list_walk.append(net(torch.tensor([context[i]]),context_attributes[i]))
            walk = torch.stack(list_walk)
            list_negative = []
            for i in range(max_len+1,len(context)):
                list_negative.append(net(torch.tensor([context[i]]),context_attributes[i]))
            embe_negative = torch.stack(list_negative)
            loss -=torch.sum(logsigmoid(simlar(embe_start,walk)))
            loss -=torch.sum(logsigmoid(simlar(-embe_start,embe_negative)))
            if i% 10000 == 1:
                print(loss.item())
            loss.backward()
            optimizer.step()
            pbar.update(1)
    ex_scedule.step()
list_label = []
list_predict = []
'''
for node1 in nodes[0:500]:
    for node2 in nodes:
        if node1 == node2:
            continue
        if node2 in edge[node1]:
            list_label.append(1)
        else:
            list_label.append(0)
        node1 = torch.tensor(node1).cuda()
        node2 = torch.tensor(node2).cuda()
        embedding_node1 = net(node1)
        embedding_node2 = net(node2)
        embedding_node1 = embedding_node1.reshape((1,len(embedding_node1)))
        embedding_node2 = embedding_node2.reshape((1,len(embedding_node2)))
        prob = sigmoid(simlar(embedding_node1,embedding_node2))
        prob = prob.cpu()
        list_predict.append(prob.detach().numpy())
'''
for i,j in edge_to_test_po:
    list_label.append(1)
    node1 = torch.tensor([node_dict[i]])
    attribute1 = torch.tensor(dic_attri[node1.item()])
    node2 = torch.tensor([node_dict[j]])
    attribute2 = torch.tensor(dic_attri[node2.item()])
    embedding_node1 = net(node1,attribute1)
    embedding_node2 = net(node2,attribute2)
    prob = sigmoid(simlar(embedding_node1,embedding_node2))
    prob = prob.cpu()
    list_predict.append(float(prob.detach()))
for i,j in edge_to_test_neg:
    list_label.append(0)
    node1 = torch.tensor([node_dict[i]])
    attribute1 = torch.tensor(dic_attri[node1.item()])
    node2 = torch.tensor([node_dict[j]])
    attribute2 = torch.tensor(dic_attri[node2.item()])
    embedding_node1 = net(node1,attribute1)
    embedding_node2 = net(node2,attribute1)
    prob = sigmoid(simlar(embedding_node1,embedding_node2))
    prob = prob.cpu()
    list_predict.append(float(prob.detach()))
        
print('start to compute auc')
auc = get_metrics(list_predict,list_label)
print(auc)
