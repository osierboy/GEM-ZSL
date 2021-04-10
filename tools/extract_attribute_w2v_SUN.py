#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:27:57 2019

@author: war-machince
"""

import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd)
#%%
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import pdb
import pandas as pd
import numpy as np
import gensim.downloader as api
import scipy.io as sio
import pickle
#%%
dataset = 'SUN'
#%%
print('Loading pretrain w2v modeling')
model_name = 'word2vec-google-news-300'#best modeling
model = api.load(model_name)
dim_w2v = 300
print('Done loading modeling')
#%%
replace_word = [('rockstone','rock stone'),('dirtsoil','dirt soil'),('man-made','man-made'),('sunsunny','sun sunny'),
                ('electricindoor','electric indoor'),('semi-enclosed','semi enclosed'),('far-away','faraway')]
#%%
file_path = 'datasets/attribute/{}/attributes.mat'.format(dataset)
matcontent = sio.loadmat(file_path)
des = matcontent['attributes'].flatten()
#%%
df = pd.DataFrame()
#%% filter
new_des = [''.join(i.item().split('/')) for i in des]
#%% replace out of dictionary words
for pair in replace_word:
    for idx,s in enumerate(new_des):
        new_des[idx]=s.replace(pair[0],pair[1])
print('Done replace OOD words')
#%%
df['new_des']=new_des
df.to_csv('datasets/attribute/{}/new_des.csv'.format(dataset))
print('Done preprocessing attribute des')
#%%
all_w2v = []
for s in new_des:
    print(s)
    words = s.split(' ')
    if words[-1] == '':     #remove empty element
        words = words[:-1]
    w2v = np.zeros(dim_w2v)
    for w in words:
        try:
            w2v += model[w]
        except Exception as e:
            print(e)
    all_w2v.append(w2v[np.newaxis,:])
#%%
all_w2v=np.concatenate(all_w2v,axis=0)
pdb.set_trace()
#%%
with open('./data/w2v/{}_attribute.pkl'.format(dataset),'wb') as f:
    pickle.dump(all_w2v,f)