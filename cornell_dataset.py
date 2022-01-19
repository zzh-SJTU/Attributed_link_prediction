# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:06:41 2022

@author: wzl
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
data_dir = os.path.expanduser("data/wisconsin")
edgelist = pd.read_csv(os.path.join(data_dir,"out1_graph_edges.txt"), sep='\t', header=None, names=["target", "source"])
Gnx = nx.from_pandas_edgelist(edgelist)
node_data = pd.read_csv(os.path.join(data_dir,"out1_node_feature_label.txt"), sep='\t', header=None)
frac_of_test  = 0.5
Gnx.remove_node("node_id")
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

node_cnt = len(Gnx.nodes)


def get_metrics(prediction, label):
    assert len(prediction) == len(label), (len(prediction), len(label))
    fpr, tpr, thresholds = metrics.roc_curve(label, prediction, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc
def softmax(x):
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum()
class node_embedding_net(nn.Module):
    def __init__(self,node_num,embedding_num):
        super(node_embedding_net, self).__init__()
        self.em = nn.Embedding(node_num,embedding_num)
    def forward(self,targ):
        emb = self.em(targ)
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

def get_MSS(Gnx):
    
    MSS = np.zeros((len(Gnx.nodes),len(Gnx.nodes)))
    nodes = list(Gnx.nodes)
    list_dis = []
    for i in range(node_cnt):
        node = str(i)
        list_nei = list(Gnx.neighbors(node))
        set_nei = set(list_nei)
        set_nei = set_nei | set([node])
        for nei in list_nei:
            list_nei_nei = list(Gnx.neighbors(nei))
            set_nei_nei = set(list_nei_nei)
            set_nei = set_nei | set_nei_nei
        list_dis.append(set_nei)
    
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            temp_edge = (str(i), str(j))
            if temp_edge not in Gnx.edges:
                continue
            
            MSS[i, j] = 2*len(list_dis[i] & list_dis[j] ) / (len(list_dis[i]) + len(list_dis[j]))
            
    return MSS

MSS = get_MSS(Gnx)
def get_MAS(Gnx, node_data):
    MAS = np.zeros((len(Gnx.nodes),len(Gnx.nodes)))
    nodes = list(Gnx.nodes)
    tmp_list = node_data.iloc[1][1].split(",")
    att_len = len(tmp_list)
    print(att_len)
    att_list = []
    
    for i in range(len(nodes)):
        att_list.append(node_data.iloc[i+1][1].split(","))
        att_list[i].append(node_data.iloc[i+1][2])
    IDF_TF = np.zeros(att_len)
    for idx in tqdm(range(att_len-1)):
        cnt = 0
        for i in range(len(nodes)):
            if att_list[i][idx]=='1':
                cnt += 1
        if cnt > 0:
            IDF_TF[idx] = np.log((len(nodes)/cnt))
    for i in tqdm(range(len(nodes))):
        for j in range(len(nodes)):
            same = 0
            tot = 0
            temp_edge = (str(i), str(j))
            if temp_edge not in Gnx.edges:
                continue
            for idx in range(att_len):
                if att_list[i][idx] == '1' or att_list[j][idx]== '1':
                    tot += IDF_TF[idx]
                    if att_list[i][idx] == att_list[j][idx]:
                        same += IDF_TF[idx]
            tot += 10
            if att_list[i][att_len] == att_list[j][att_len]:
                same += 10
            MAS[i][j] = same / tot
    return MAS
MAS = get_MAS(Gnx, node_data)
alpha = 0.9
W = MSS * alpha + MAS * (1-alpha)
#W = np.ones((len(Gnx.nodes),len(Gnx.nodes)))

def one_randwalk(start_point,graph,negative_num,posi_len, W):
    traj =[int(start_point)]
    node = start_point
    for i in range(posi_len):
        list_nei = list(graph.neighbors(node))
        p = []
        for nei in list_nei:
            nei = int(nei)
            p.append(W[int(node)][int(nei)])
        p /= sum(p)
        node = np.random.choice(a=list_nei,p=p)
        while len(list(graph.neighbors(node)))==0:
            node = random.choice(list_nei)
        traj.append(int(node))
    counter = 0
    while(1):
        nega_number = random.sample(graph.nodes, 1)
        temp_edge = (start_point,nega_number[0])
        if temp_edge not in graph.edges:
            counter+=1
            traj.append(int(nega_number[0]))
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
for max_len in [6, 8,10,12,14,16]:
    for learning_rate in [0.01, 0.001, 0.0001, 0.0005, 0.002, 0.003,0.02]:
        edge = []
        trajecrories = []
        EPOCH = 10
        embedding_num = 1024
        negative_num = 1
        one_hot_dic ={}
        temp = 0
        counter = 0
        for node in Gnx.nodes:
            if len(list(Gnx.neighbors(node)))==0 :
                continue
            for i in range(40):
                one_walk = one_randwalk(node,Gnx,negative_num,max_len, W)
                trajecrories.append(one_walk)
        train_data = []
        for traj in trajecrories:
            train_data.append((traj[0],traj[1:]))
        
        data_set = Custom_dataset(train_data)
        train_loader = DataLoader(data_set,batch_size=1,shuffle=True)
        #print(len(Gnx.nodes))
        net = node_embedding_net(len(Gnx.nodes),embedding_num).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        simlar = nn.CosineSimilarity()
        loss_fn = nn.CrossEntropyLoss()
        logsigmoid = nn.LogSigmoid()
        sigmoid = nn.Sigmoid()
        ex_scedule = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.95)
        
        for i in range(EPOCH):
            with tqdm(total=len(train_loader)) as pbar:
                pbar.set_description('Processing:')
                for i,data in enumerate(train_loader):
                    start,context = data
                    loss = 0
                    #print(start, context)
                    #print(context)
                    context = torch.tensor(context).cuda()
                    start = torch.tensor(start).cuda()
                    
                    embe_start = net(start)
                    walk = net(context[0:max_len])
                    embe_negative = net(context[max_len+1:])
                    loss -=torch.sum(logsigmoid(simlar(embe_start,walk)))
                    loss -=torch.sum(logsigmoid(simlar(-embe_start,embe_negative)))
                    #if i% 10000 == 1:
                    #    print(loss.item())
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
            node1 = torch.tensor(int(i)).cuda()
            node2 = torch.tensor(int(j)).cuda()
            embedding_node1 = net(node1)
            embedding_node2 = net(node2)
            embedding_node1 = embedding_node1.reshape((1,len(embedding_node1)))
            embedding_node2 = embedding_node2.reshape((1,len(embedding_node2)))
            prob = sigmoid(simlar(embedding_node1,embedding_node2))
            prob = prob.cpu()
            list_predict.append(float(prob.detach()))
        for i,j in edge_to_test_neg:
            list_label.append(0)
            node1 = torch.tensor(int(i)).cuda()
            node2 = torch.tensor(int(j)).cuda()
            embedding_node1 = net(node1)
            embedding_node2 = net(node2)
            embedding_node1 = embedding_node1.reshape((1,len(embedding_node1)))
            embedding_node2 = embedding_node2.reshape((1,len(embedding_node2)))
            prob = sigmoid(simlar(embedding_node1,embedding_node2))
            prob = prob.cpu()
            list_predict.append(float(prob.detach()))
                
        print('start to compute auc')
        auc = get_metrics(list_predict,list_label)
        print(auc)
        
        
        Phase = 3
        for phase in range(Phase):
            print("phase:",phase)
            M = np.zeros((len(Gnx.nodes),len(Gnx.nodes)))
            for edge in Gnx.edges:
                (i, j) = edge
                node1 = torch.tensor(int(i)).cuda()
                node2 = torch.tensor(int(j)).cuda()
                embedding_node1 = net(node1)
                embedding_node2 = net(node2)
                embedding_node1 = embedding_node1.reshape((1,len(embedding_node1)))
                embedding_node2 = embedding_node2.reshape((1,len(embedding_node2)))
                prob = sigmoid(simlar(embedding_node1,embedding_node2))
                M[int(i)][int(j)] = prob
            M = 0.3*M + 0.7*W
            trajecrories = []
            for node in Gnx.nodes:
                if len(list(Gnx.neighbors(node)))==0 :
                    continue
                for i in range(5):
                    one_walk = one_randwalk(node,Gnx,negative_num,max_len, M)
                    trajecrories.append(one_walk)
            train_data = []
            for traj in trajecrories:
                train_data.append((traj[0],traj[1:]))
        
            data_set = Custom_dataset(train_data)
            train_loader = DataLoader(data_set,batch_size=1,shuffle=True)
        
            for i in range(EPOCH):
                with tqdm(total=len(train_loader)) as pbar:
                    pbar.set_description('Processing:')
                    for i,data in enumerate(train_loader):
                        start,context = data
                        loss = 0
                        #print(start, context)
                        context = torch.tensor(context).cuda()
                        start = torch.tensor(start).cuda()
                        
                        embe_start = net(start)
                        walk = net(context[0:max_len])
                        embe_negative = net(context[max_len+1:])
                        loss -=torch.sum(logsigmoid(simlar(embe_start,walk)))
                        loss -=torch.sum(logsigmoid(simlar(-embe_start,embe_negative)))
                        #if i% 10000 == 1:
                        #    print(loss.item())
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
                node1 = torch.tensor(int(i)).cuda()
                node2 = torch.tensor(int(j)).cuda()
                embedding_node1 = net(node1)
                embedding_node2 = net(node2)
                embedding_node1 = embedding_node1.reshape((1,len(embedding_node1)))
                embedding_node2 = embedding_node2.reshape((1,len(embedding_node2)))
                prob = sigmoid(simlar(embedding_node1,embedding_node2))
                prob = prob.cpu()
                list_predict.append(float(prob.detach()))
            for i,j in edge_to_test_neg:
                list_label.append(0)
                node1 = torch.tensor(int(i)).cuda()
                node2 = torch.tensor(int(j)).cuda()
                embedding_node1 = net(node1)
                embedding_node2 = net(node2)
                embedding_node1 = embedding_node1.reshape((1,len(embedding_node1)))
                embedding_node2 = embedding_node2.reshape((1,len(embedding_node2)))
                prob = sigmoid(simlar(embedding_node1,embedding_node2))
                prob = prob.cpu()
                list_predict.append(float(prob.detach()))
                    
            print('start to compute auc')
            auc = get_metrics(list_predict,list_label)
            print(auc)
        
        
        
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
            node1 = torch.tensor(int(i)).cuda()
            node2 = torch.tensor(int(j)).cuda()
            embedding_node1 = net(node1)
            embedding_node2 = net(node2)
            embedding_node1 = embedding_node1.reshape((1,len(embedding_node1)))
            embedding_node2 = embedding_node2.reshape((1,len(embedding_node2)))
            prob = sigmoid(simlar(embedding_node1,embedding_node2))
            prob = prob.cpu()
            list_predict.append(float(prob.detach()))
        for i,j in edge_to_test_neg:
            list_label.append(0)
            node1 = torch.tensor(int(i)).cuda()
            node2 = torch.tensor(int(j)).cuda()
            embedding_node1 = net(node1)
            embedding_node2 = net(node2)
            embedding_node1 = embedding_node1.reshape((1,len(embedding_node1)))
            embedding_node2 = embedding_node2.reshape((1,len(embedding_node2)))
            prob = sigmoid(simlar(embedding_node1,embedding_node2))
            prob = prob.cpu()
            list_predict.append(float(prob.detach()))
                
        print('start to compute auc')
        auc = get_metrics(list_predict,list_label)
        print(learning_rate)
        print(max_len)
        print(auc)
        
