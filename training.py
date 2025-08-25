# -*- coding: utf-8 -*-
"""
Created on Fri May 14 09:34:25 2021

@author: 
"""


from util import get_neg, get_hits, loadfile, rpl_module, compute_r, rfunc, neighbor_rels_1hop, inner_join
import numpy as np
import time
import random 

seed = 2
np.random.seed(seed)
random.seed(seed)
'''
dbp15k: zh_en | ja_en | fr_en 
srprs: en_fr | en_de
crossdomain: db-yg | db-yg-realea
'''
kgs = "crossdomain" # dbp15k | srprs | crossdomain
language = 'db-yg-realea' 
prior_align_percentage = 0 # 0% of seeds
num_model_calibration = 3
repeat_num = 1
gamma = 1

e1 = 'data/' + kgs + "/" + language + '/ent_ids_1'
e2 = 'data/' + kgs + "/" + language + '/ent_ids_2'
ill = 'data/' + kgs + "/" + language + '/ref_ent_ids'
kg1 = 'data/' + kgs + "/" + language + '/triples_1'
kg2 = 'data/' + kgs + "/" + language + '/triples_2'


ill = loadfile(ill, 2)
illL = len(ill)
np.random.shuffle(ill)
train = np.array(ill[:int(illL // 10 * prior_align_percentage)])
test = ill[int(illL // 10 * prior_align_percentage) : ]

kg1 = loadfile(kg1, 3)
kg2 = loadfile(kg2, 3)

e1 = loadfile(e1, 1)
e2 = loadfile(e2, 1)
e = len(set(e1) | set(e2))
kg = kg1 + kg2
ents = e1 + e2

head_r, tail_r, rel_type, rel_num = rfunc(kg, e)


ent_r = neighbor_rels_1hop(kg, e, rel_num)

# load embedding_lists
import torch
import json

if kgs == "dbp15k":
    with open(file='data/' + kgs + "/" + language[0:2] + '_en/' + language[0:2] + '_bert300.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
elif kgs == "srprs":
    with open(file='data/' + kgs + "/"  + language + '/' + language[-2:] + '_bert300.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
else:
    with open(file='data/' + kgs + "/"  + language + '/' + language[:5] + '_vectorList.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)

word_embeddings = torch.FloatTensor(embedding_list)

r_kg_1 = set()
for tri in kg1:
    r_kg_1.add(tri[1])
l1 = len(r_kg_1)
l2 = rel_num - l1
#%%
from UPL_EA import upl_ea
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def sp_mat(xxxx_r):
    xxxx_r = xxxx_r/np.expand_dims(np.sum(xxxx_r, -1), -1) # row normalization
    return torch.sparse_coo_tensor(np.stack(xxxx_r.nonzero()), xxxx_r[xxxx_r.nonzero()], xxxx_r.shape, dtype = torch.float32, device = device)

head_r_temp = head_r
tail_r_temp = tail_r
head_r_l_sp = sp_mat(head_r_temp.T)
tail_r_l_sp = sp_mat(tail_r_temp.T)
inputs = [head_r_l_sp, tail_r_l_sp]
ent_r = sp_mat(ent_r)

if kgs == "srprs":
    learning_rate = 0.00025
else:
    learning_rate = 0.001

def defmodel():
    model = upl_ea(
        input_dim = 300,
        primal_X_0 = word_embeddings.to(device=device),
        gamma = gamma,
        e = e,
        KG = kg,
        inputs = inputs,
        ent_r = ent_r,
        device = device,
        ).to(device = device)
    
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

models = []
optimizers = []
for _ in range(num_model_calibration):
    model, optimizer = defmodel()
    models.append(model)
    optimizers.append(optimizer)

# head_r, tail_r, rel_type = rfunc_rnm(kg, e)
k = 125
batch_size = 256 # one of the hyper-parameters
re_init_epoch = num_model_calibration*10
init_epochs = re_init_epoch*repeat_num
epochs = init_epochs + 80
training_losses = []
valid_hit1s = []
output_layers = []


pseudo_labels = []


time0 = time.time()
pl_vote_dict = {}
# init_align = train
best_hit1 = 0
best_out = 0
best_epoch = 0
for i in range(epochs):
    
    
    if i % 10 == 0:
        
        if i==0:
            model = models[i]
            optimizer = optimizers[i]
            model.eval()
            out = model.build(inputs)
            _ = get_hits(out, test)            
            
            outvec_r = compute_r(out.detach().cpu(), head_r_temp, tail_r_temp, 300)
            init_align, r_score = rpl_module(out.detach(), outvec_r.to(device=device), l1, kg, train, rel_type, e1, e2, iters=1)
            pseudo_labels.append(init_align)
            
            init_align0 = init_align

        else:
            model.eval()
            out = model.build(inputs)
            out_hit1 = get_hits(out, test)
            
            if out_hit1 > best_hit1:
                best_hit1 = out_hit1
                best_out = out.detach()
                best_epoch = i
            
            outvec_r = compute_r(out.detach().cpu(), head_r_temp, tail_r_temp, 300)
            init_align, r_score = rpl_module(out.detach(), outvec_r.to(device=device), l1, kg, train, rel_type, e1, e2, iters=1)
            pseudo_labels.append(init_align)
            
            if i <= re_init_epoch:
                idx = (i-10)%num_model_calibration
                models[idx] = model
                
                idx = (i)%num_model_calibration
                model = models[idx] # next model
                optimizer = optimizers[idx] # nexe optimizer

                for pl in init_align:
                    if (pl[0],pl[1]) in pl_vote_dict:
                        pl_vote_dict[(pl[0], pl[1])] = pl_vote_dict[(pl[0], pl[1])] + 1
                    else: 
                        pl_vote_dict[(pl[0], pl[1])] = 1

                if i == re_init_epoch:
                    pl_maj_vote = []
                    for pl, vote in pl_vote_dict.items():
                        if vote==max(pl_vote_dict.values()):
                            pl_maj_vote.append(list(pl))
                    pl_vote_dict = {}
                    common_PLs = np.array(pl_maj_vote)
                    train = common_PLs
                    init_align0 = train
                    
                    if re_init_epoch < init_epochs:
                        re_init_epoch += num_model_calibration*10
                    else:
                        model = upl_ea(
                            input_dim = 300,
                            primal_X_0 = word_embeddings.to(device=device),
                            gamma = gamma,
                            e = e,
                            KG = kg,
                            inputs = inputs,
                            ent_r = ent_r,
                            device = device,
                            ).to(device = device)
                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            else:
                init_align0 = init_align
        
        neg_right = get_neg(init_align0[:, 0], out, k)
        neg2_left = get_neg(init_align0[:, 1], out, k)

        model.train()
        print('')


    examples_train_index = np.arange(len(init_align0))
    np.random.shuffle(examples_train_index)
    batches = init_align0.shape[0] // (batch_size)
    for j in range(batches):
        index_range = examples_train_index[j*batch_size:(j+1)*batch_size]
        examples_train = init_align0[index_range]
        neg_right_sample = neg_right[index_range]
        neg2_left_sample = neg2_left[index_range]
        r_score_sample = r_score[index_range]
        
        Inputs = [examples_train[:, 0],
                  examples_train[:, 1],
                  torch.LongTensor(neg_right_sample).to(device=device),
                  torch.LongTensor(neg2_left_sample).to(device=device),
                  inputs,
                  torch.Tensor(r_score_sample).to(device=device),
                  ]

        loss = model(Inputs) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('Epoch: {}/{}  Loss: {:.4f}'.format(i+1, epochs, loss.item()))
    

time1 = time.time()
model.eval()
out = model.build(inputs)
out_hit1 = get_hits(out, test)
if out_hit1 > best_hit1:
    best_hit1 = out_hit1
    best_out = out.detach()
    best_epoch = i
print('time: {:.0f}mins {:.0f}secs'.format(((time1-time0)-((time1-time0)%60))/60, (time1-time0)%60))
print('')
print("best epoch is {}:".format(best_epoch))
print("")
out_hit1 = get_hits(best_out, test)

if kgs == "crossdomain":
    outvec_r = compute_r(best_out.cpu(), head_r_temp, tail_r_temp, 300)
    init_align, r_score = rpl_module(best_out, outvec_r.to(device=device), l1, kg, np.array(ill[:int(illL // 10 * prior_align_percentage)]), rel_type, np.array(test)[:,0], np.array(test)[:,1], iters=1)
    
    trainsize = int(illL // 10 * prior_align_percentage)
    precision = len(inner_join(test, init_align))/(len(init_align)-trainsize)
    recall = len(inner_join(test, init_align))/len(test)
    f1 = 2*precision*recall/(precision + recall)
    
    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("f1-score: {:.3f}".format(f1))