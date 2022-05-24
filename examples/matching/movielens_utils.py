"""
    util function for movielens data.
"""

import collections
import numpy as np
import pandas as pd
from torch_rechub.utils.match import Annoy
from torch_rechub.basic.metric import topk_metrics
from collections import Counter


def match_evaluation(user_embedding, item_embedding, test_user, all_item, topk=10):
    print("evaluate embedding matching on test data")
    annoy = Annoy(n_trees=10)
    annoy.fit(item_embedding)

    #for each user of test dataset, get ann search topk result
    print("matching for topk")
    user_map, item_map = np.load("./data/ml-1m/saved/raw_id_maps.npy", allow_pickle=True)
    match_res = collections.defaultdict(dict)
    for user_id, user_emb in zip(test_user["user_id"], user_embedding):
        items_idx, items_scores = annoy.query(v=user_emb, n=topk)  #the index of topk match items
        match_res[user_map[user_id]] = all_item["movie_id"][items_idx]

    #get ground truth
    print("generate ground truth")
    user_col = "user_id"
    item_col = "movie_id"

    data = pd.DataFrame({"user_id": test_user["user_id"], "movie_id": test_user["movie_id"]})
    data[user_col] = data[user_col].map(user_map)
    data[item_col] = data[item_col].map(item_map)
    user_pos_item = data.groupby(user_col).agg(list).reset_index()
    ground_truth = dict(zip(user_pos_item[user_col], user_pos_item[item_col]))

    print("compute topk metrics")
    out = topk_metrics(y_true=ground_truth, y_pred=match_res, topKs=[topk])
    print(out)


def get_item_sample_weight(items):
    #It is a effective weight used in word2vec
    items_cnt = Counter(items)
    p_sample = {item: count**0.75 for item, count in items_cnt.items()}
    p_sum = sum([v for k, v in p_sample.items()])
    item_sample_weight = {k: v / p_sum for k, v in p_sample.items()}
    return item_sample_weight
