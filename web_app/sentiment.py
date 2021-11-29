#!/usr/bin/env python3
#function to compute sentiment score

import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords


from tensorflow import keras
import json
from nltk.tokenize import RegexpTokenizer

import warnings
warnings.filterwarnings("ignore")
#function to compute sentiment score
#function to compute sentiment score
def sentiment_score(LSTM_model, txt_data, index_wrd):

    list_txt = []
    pad_seq = len(txt_data)
    pad_len = np.zeros((56,pad_seq))

    dict_txt = [isinstance(item, (str, bytes)) for item in txt_data['title']]
    txt_data = txt_data.loc[dict_txt]

    for i, j in txt_data.iterrows():
        

        txt_col = txt_data['title'][i]
        
        # split the sentence into its words and remove any punctuations.
        tokenzz = RegexpTokenizer(r'\w+')
        list_txt_data = tokenzz.tokenize(txt_col)
        if len(list_txt_data)>56:
          list_txt_data = list_txt_data[:56]
          points = np.array(['1','2','3','4','5','6','7','8','9','10'], dtype = "int")
        
        # get index for the live stage
        ind_data = np.array([index_wrd[word.lower()] if word.lower() in index_wrd else 0 for word in list_txt_data])
        ind_data_ = np.array(ind_data)

        # padded with zeros of length 56 i.e maximum length
        pad_arr = np.zeros(56)
        pad_arr[:ind_data_.shape[0]] = ind_data_
        data_pad = pad_arr.astype(int)

        list_txt.append(data_pad)

    pad_len = np.asarray(list_txt)
    sent_score = LSTM_model.predict(pad_len, batch_size=64, verbose=0)
    sent_score1 = np.round(np.dot(sent_score, points)/10,decimals=2)
    sent_score2  = []
    for i in sent_score:

        three_top_idx = np.argsort(i)[-3:]
        three_top_scores = i[three_top_idx]
        three_top_wgts = three_top_scores/np.sum(three_top_scores)
        dot_prod = np.round(np.dot(three_top_idx, three_top_wgts)/10, decimals = 2)
        sent_score2.append(dot_prod)

    txt_data['Sentiment_Score'] = pd.DataFrame(sent_score2)

    return txt_data
    
