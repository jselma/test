import tensorflow 
import libreco

from libreco.data import random_split, DatasetPure
import time
import numpy as np
import tensorflow as tf
import pandas as pd
from libreco.data import split_by_ratio_chrono, DatasetPure
from libreco.algorithms import (
    SVD, SVDpp, NCF, ALS, UserCF, ItemCF, RNN4Rec, KnnEmbedding,
    KnnEmbeddingApproximate, BPR
)
from libreco.data import (
    split_by_num,
    split_by_ratio,
    split_by_num_chrono,
    split_by_ratio_chrono,
    random_split
)


############################# READ DATA #############################
data = pd.read_csv('dataLibrecoFormat.csv', sep=';')
data = data[['user','item','label']]

############################# SPLIT DATA TRAIN, EVAL, TEST #############################
# split whole data into three folds for training, evaluating and testing
train_data_01, eval_data_01, test_data_01 = random_split(data, multi_ratios=[0.7, 0.15, 0.15])

print("User train:",train_data_01['user'].unique().size,
      "\nUser validation:", eval_data_01['user'].unique().size,
      "\nUser test:", test_data_01['user'].unique().size,
      "\nAll users:", data['user'].unique().size,
      
      "\n\nItem train:",train_data_01['item'].unique().size,
      "\nItem validation:", eval_data_01['item'].unique().size,
      "\nItem test:", test_data_01['item'].unique().size,
      "\nAll items:", data['item'].unique().size,
      
     "\n\nTrain interactions:",train_data_01.shape[0],
     "\nValidation interactions:",eval_data_01.shape[0],
     "\nTest interactions:",test_data_01.shape[0],
     "\nAll interactions:",data.shape[0],
     
     "\nTrain percent:",train_data_01.shape[0]/data.shape[0],
     "\nValidation percent:",eval_data_01.shape[0]/data.shape[0],
     "\nTest percent:",test_data_01.shape[0]/data.shape[0],
     )



############################# DATA PREPARATION #############################
train_data, data_info = DatasetPure.build_trainset(train_data_01)
eval_data = DatasetPure.build_evalset(eval_data_01)

# do negative sampling, assume the data only contains positive feedback
train_data.build_negative_samples(data_info, item_gen_mode="random",
                                  num_neg=1, seed=20)
eval_data.build_negative_samples(data_info, item_gen_mode="random",
                                 num_neg=1, seed=22)


############################# TUNING HYPERPARAMETERS #############################
'''
regulSel = 10
itera = 60
alphaSel = 40

factors = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150]

for tempFactor in factors:   
  
    reset_state("ALS")
    als = ALS(task="ranking", data_info=data_info, embed_size=tempFactor, n_epochs=itera,
              reg=regulSel, alpha=alphaSel, seed=42)
    
    als.fit(train_data, verbose=2, use_cg=True, n_threads=4,
        eval_data=eval_data, metrics=["loss", "balanced_accuracy",
                                              "roc_auc", "pr_auc", "precision",
                                              "recall", "map", "ndcg"])
'''        
    
    




############################# TRAIN BEST HYPERPARAMETERS #############################
factorSel=350
regulSel=90
itera = 100
alphaSel = 0

reset_state("ALS")
als = ALS(task="ranking", data_info=data_info, embed_size=factorSel, n_epochs=itera,
          reg=regulSel, alpha=alphaSel, seed=42)

als.fit(train_data, verbose=2, use_cg=True, n_threads=4,
        eval_data=eval_data, metrics=["loss", "balanced_accuracy",
                                              "roc_auc", "pr_auc", "precision",
                                              "recall", "map", "ndcg"])



############################# RESULTS IN DATASET TEST #############################
test_data = DatasetPure.build_evalset(test_data_01)

als.fit(train_data, verbose=2, use_cg=True, n_threads=4,
        eval_data=test_data, metrics=["loss", "balanced_accuracy",
                                              "roc_auc", "pr_auc", "precision",
                                              "recall", "map", "ndcg"])



