#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:02:44 2019

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
print('Loading pretrain w2v modeling')
model_name = 'word2vec-google-news-300'#best modeling
model = api.load(model_name)
dim_w2v = 300
print('Done loading modeling')
#%%
replace_word = [('newworld','new world'),('oldworld','old world'),('nestspot','nest spot'),('toughskin','tough skin'),
                ('longleg','long leg'),('chewteeth','chew teeth'),('meatteeth','meat teeth'),('strainteeth','strain teeth'),
                ('quadrapedal','quadrupedal')]
dataset = 'AWA2'
#%%
path = 'datasets/Attribute/attribute/{}/predicates.txt'.format(dataset)
df=pd.read_csv(path,sep='\t',header = None, names = ['idx','des'])
des = df['des'].values
#%% filter
#new_des = [' '.join(i.split('_')) for i in des]
#new_des = [' '.join(i.split('-')) for i in new_des]
#new_des = [' '.join(i.split('::')) for i in new_des]
#new_des = [i.split('(')[0] for i in new_des]
#new_des = [i[4:] for i in new_des]
#%% replace out of dictionary words
for pair in replace_word:
    for idx,s in enumerate(des):
        des[idx]=s.replace(pair[0],pair[1])
print('Done replace OOD words')
#%%
df['new_des']=des
df.to_csv('datasets/Attribute/attribute/{}/new_des.csv'.format(dataset))
#print('Done preprocessing attribute des')
#%%
counter_err = 0
all_w2v = []
for s in des:
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
            counter_err += 1
    all_w2v.append(w2v[np.newaxis,:])
print('counter_err ',counter_err)
#%%
all_w2v=np.concatenate(all_w2v,axis=0)
pdb.set_trace()
#%%
with open('datasets/Attribute/w2v/{}_attribute.pkl'.format(dataset),'wb') as f:
    pickle.dump(all_w2v,f)    
