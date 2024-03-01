import json
import random
import time
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict, Counter
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import torch
import pickle


def preprocess(data_name, coder='BERT'):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []
  u_l = []
  i_l = []
  genre_all = []
  ts_min = 0
  m_t = 0.8
  t_m = 1
  C = 150 # Count
  d_t = 600  # duration 180

  # embedding_dict = {}
  if coder=='Glove':
    # Or, we can try 'glove.6B.100d.txt'
    with open("./data/glove.6B.50d.txt", "r", encoding="utf8") as f:
      for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embedding_dict[word] = vector
  elif coder=='BERT':
    # The pre-processed BERT data
    # Or, we can re-produced the data with the BERT.pkl
    embedding_dict = np.load("./data/BERT.npy", allow_pickle= True).item()

  with open(data_name, encoding='UTF-8') as f:
    s = next(f)
    for idx, line in enumerate(f):
        # coding:UTF-8
        e = line.strip().split(',')
        du_t = int(float(e[3]))
        if du_t >= d_t:
          timestamp = time.strptime(e[2], "%Y-%m-%d %H:%M:%S")# timestamp
          ts_max = float(time.mktime(timestamp))
          if ts_min == 0:
            ts_min = ts_max
  #  0:user_id 1:movie_id 2:datetime 3:duration 4:genres
  user,ind_u, count_u= np.unique(u_l, return_index = True, return_counts = True)
  user = user[ind_u.argsort()]
  count_u = count_u[ind_u.argsort()]

  users = []

  if coder == 'One':
    with open(data_name, encoding='UTF-8') as f:
      s = next(f)
      for idx, line in enumerate(f):
        # coding:UTF-8
        e = line.strip().split(',')
        du_t = int(e[3])
        if du_t >= d_t:
          timestamp = time.strptime(e[2], "%Y-%m-%d %H:%M:%S")  # timestamp
          if time.mktime(timestamp) - ts_min >= (ts_max - ts_min) * m_t:
            u_l.append(e[0])
            i_l.append(e[1])
            g = [re.sub(r'[^\w\-]', "", items) for items in e[5:]]
            genre_all.extend(g)
    value_g = np.array(genre_all)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(value_g)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    G_dictionary = dict(zip(genre_all, onehot_encoded)) # one hot
  else:
    G_dictionary = embedding_dict

  # item,ind_i,count_i = np.unique(i_l, return_index = True, return_counts = True)
  # item = item[ind_i.argsort()]
  # max_c_i = max(count_i)
  # max_c = max(count_u)
  # average_c = np.mean(count_u)
  # var_c = np.var(count_u)
  # count_c_n = []
  # for i in range(len(count_i)):
  #   x = np.random.normal(average_c, var_c/10)
  #   c = count_i[i]#item perspective for cyx
  #   if c >= C:
  #     users.append(user[i])
  #     count_c_n.append(c)
  # mean_c = np.mean(count_c_n)
  # var_c_n = np.var(count_c_n)
  # dict_c = Counter(count_u)
  # dict_c_n = Counter(count_c_n)
  # all_sum = np.sum(count_c_n)
  dict_u = dict(zip(users, range(len(users))))
  dict_i = {}
  dict_g = {}
  #  0:user_id 1:movie_id 2:datetime 3:duration 4:genres
  with open(data_name, encoding='UTF-8') as f:
    s = next(f)
    flag_i = 0
    flag = 0
    max_len = 0
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      du_t = int(e[3])
      if du_t >= d_t and e[0] in dict_u:  # item
        timestamp = time.strptime(e[2], "%Y-%m-%d %H:%M:%S")
        min_ts = ts_min
        ts = float(time.mktime(timestamp)-min_ts)
        if ts >= ((ts_max-min_ts)*t_m):
          break
        if ts >= ((ts_max-min_ts)*m_t):
          u = dict_u[e[0]]
          if e[1] not in dict_i:
            flag_i += 1
            dict_i[e[1]] = flag_i
          i = dict_i[e[1]]
          label = float(0)  # int(e[3]), state label, 1 whenever the user state changes, 0 otherwise.
          feat = np.ones(16)  # comma-separated array of features 16
          u_list.append(u)
          i_list.append(i)
          ts_list.append(ts)
          label_list.append(label)
          idx_list.append(flag)
          feat_l.append(feat)
          flag += 1
          genre = [re.sub(r'[^\w\-]', "", items) for items in e[4:]]
          # genre_l.append([embedding_dict[words] for words in genre])
          if i not in dict_g.keys():
            # genes = np.array([embedding_dict[words] for words in genre])
            genes = np.array([G_dictionary[words.lower()] for words in genre])#G_dictionary
            if 8-len(genre) > 0:
              zeros = np.array([np.zeros(768) for _ in range(8-len(genre))])
              dict_g[i] = np.vstack([zeros, genes])
            else:
              dict_g[i] = np.vstack([genes])
            max_len = max(max_len, len(genre))
  dict_g = dict(sorted(dict_g.items(), key=lambda x:x[0]))
  genre_l = list(dict_g.values())
  rand = np.zeros((len(np.unique(u_list))+len(np.unique(i_list)) + 2 - len(genre_l), 8, 768))
  genre_np = np.array(genre_l)
  genre_np = np.vstack([rand, genre_np])



  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l), genre_np


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:

    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    # upper_u = len(df.u) + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_name, bipartite=True, coder='BERT'):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = './data/{}.csv'.format(data_name)
  OUT_DF = './data/ml_{}.csv'.format(data_name)
  OUT_FEAT = './data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)
  OUT_NODE_GENE = './data/ml_{}_gene.npy'.format(data_name)

  df, feat, genre = preprocess(PATH, coder=coder)
  new_df = reindex(df, bipartite)

  empty = np.zeros(feat.shape[1])[np.newaxis, :] # [,0...]
  feat = np.vstack([empty, feat])


  max_idx = max(new_df.u.max(), new_df.i.max())
  rand_feat = np.random.rand(max_idx + 1, 172)


  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, rand_feat)
  np.save(OUT_NODE_GENE, genre)

parser = argparse.ArgumentParser('Interface for STGN data preprocessing (Netflix)')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

parser.add_argument('--coder', type=str, help='Semantic encoder', defaul='BERT')

args = parser.parse_args()

# python netflix_process.py --bipartite --coder BERT
run('netflix', bipartite=args.bipartite, coder=args.coder)