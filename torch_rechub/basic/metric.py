"""The metric module, it is used to provide some metrics for recommenders.
Available function:
- auc_score: compute AUC
- gauc_score: compute GAUC
- log_loss: compute LogLoss
- topk_metrics: compute topk metrics contains 'ndcg', 'mrr', 'recall', 'hit'
"""
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd


def auc_score(y_true, y_pred):

	return roc_auc_score(y_true, y_pred)


def get_user_pred(y_true, y_pred, users):
	"""divide the result into different group by user id

	Args:
		y_true: array, all true labels of the data
		y_pred: array, the predicted score
		users: array, user id 

	Return:
		user_pred: dict, key is user id and value is the labels and scores of each user
	"""
	user_pred = {}
	for i, u in enumerate(users):
		if u not in user_pred:
			user_pred[u] = [[y_true[i]], [y_pred[i]]]
		else:
			user_pred[u][0].append(y_true[i])
			user_pred[u][1].append(y_pred[i])

	return user_pred


def get_user_topk(y_true, y_pred, users, k):
	"""sort y_pred and find topk results
	this function is used to find topk predicted scores 
	and the corresponding index is applied to find the corresponding labels

	"""
	y_pred = np.argsort()
	for u in range(y_pred.shape[0]):
		idx = np.argsort(user_pred[u])[::-1][:k]
		user_pred[u][1] = np.array(user_pred[u][1])[idx]
		user_pred[u][0] = np.array(user_pred[u][0])[idx]
	return user_pred


def gauc_score(y_true, y_pred, users, weights=None):
	"""compute GAUC

	Args: 
		y_true: array, dim(N, ), all true labels of the data
		y_pred: array, dim(N, ), the predicted score
		users: array, dim(N, ), user id 
		weight: dict, it contains weights for each group. 
				if it is None, the weight is equal to the number
				of times the user is recommended
	Return:
		score: float, GAUC
	"""
	assert len(y_true) == len(y_pred) and len(y_true) == len(users)

	user_pred = get_user_topk(y_true, y_pred, users, len(users))
	score = 0
	num = 0
	for u in user_pred.keys():
		auc = auc_score(user_pred[u][0], user_pred[u][1])
		if weights is None:
			wg = len(user_pred[u][0])
		else:
			wg = weights[u]
		auc *= wg
		num += wg
		score += auc
	return score / num



# def ndcg_score(user_pred, k):
# 	"""compute NDCG
# 	Args:
# 		user_pred: dict, computed by get_user_topk()
# 	"""
# 	rank = np.arange(1, k+1, 1)
# 	idcgs = 1. / np.log2(rank + 1)
# 	idcg = sum(idcgs)
# 	score = 0
# 	for u in user_pred:
# 		dcgs = idcgs[np.where(user_pred[u][0] == 1)]
# 		dcg = sum(dcgs)
# 		score += dcg / idcg
# 	return score / len(user_pred.keys())


# def hit_score(user_pred):
# 	score = 0
# 	for u in user_pred:
# 		if 1 in user_pred[u][0]:
# 			score += 1.0
# 	return score / len(user_pred.keys())


# def mrr_score(user_pred):
# 	score = 0
# 	for u in user_pred:
# 		if 1 in user_pred[u][0]:
# 			score += 1.0 / (np.where(user_pred[u][0] == 1)[0][0] + 1)
# 	return score / len(user_pred.keys())


# def recall_score(user_pred):
# 	score = 0
# 	for u in user_pred:
# 		score += sum(user_pred[u][0]) * 1. / len(user_pred[u][0])
# 	return score / len(user_pred.keys())


def topk_metrics(y_true, y_pred, topKs=[3]):
	"""choice topk metrics and compute it
	the metrics contains 'ndcg', 'mrr', 'recall' and 'hit'

	Args:
		y_true: list, 2-dim, each row contains the items that the user interacted
		y_pred: list, 2-dim, each row contains the items recommended  
		topKs: list or tuple, if you want to get top5 and top10, topKs=(5, 10)

	Return:
		results: list, it contains five metrics, 'ndcg', 'recall', 'mrr', 'hit', 'precision'

	"""
	assert len(y_true) == len(y_pred)

	if not isinstance(topKs, (tuple, list)):
		raise ValueError('topKs wrong, it should be tuple or list')

	ndcg_result = []
	mrr_result = []
	hit_result = []
	precision_result = []
	recall_result = []
	for idx in range(len(topKs)):
		ndcgs = 0
		mrrs = 0
		hits = 0
		precisions = 0
		recalls = 0
		for i in range(len(y_true)):
			if len(y_true[i]) != 0:
				mrr_tmp = 0
				mrr_flag = True
				hit_tmp = 0
				dcg_tmp = 0
				idcg_tmp = 0
				hit = 0
				for j in range(topKs[idx]):
					if y_pred[i][j] in y_true[i]:
						hit += 1.
						if mrr_flag:
							mrr_flag = False
							mrr_tmp = 1. / (1 + j)
							hit_tmp = 1.
						dcg_tmp += 1. / (np.log2(j + 2))
					idcg_tmp += 1. / (np.log2(j + 2))
				hits += hit_tmp
				mrrs += mrr_tmp
				recalls += hit / len(y_true[i])
				precisions += hit / topKs[idx]
				if idcg_tmp != 0:
					ndcgs += dcg_tmp / idcg_tmp
		hit_result.append(round(hits / len(y_pred), 4))
		mrr_result.append(round(mrrs / len(y_pred), 4))
		recall_result.append(round(recalls / len(y_pred), 4))
		precision_result.append(round(precisions / len(y_pred), 4))
		ndcg_result.append(round(ndcgs / len(y_pred), 4))

	results = []
	for idx in range(len(topKs)):

		output = f'NDCG@{topKs[idx]}: {ndcg_result[idx]}'
		results.append(output)

		output = f'MRR@{topKs[idx]}: {mrr_result[idx]}'
		results.append(output)

		output = f'Recall@{topKs[idx]}: {recall_result[idx]}'
		results.append(output)
		output = f'Hit@{topKs[idx]}: {hit_result[idx]}'
		results.append(output)
		output = f'Precision@{topKs[idx]}: {precision_result[idx]}'
		results.append(output)
	return results




def log_loss(y_true, y_pred):
	score = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
	return -score.sum() / len(y_true)


# y_pred = np.array([  0.3, 0.2, 0.5, 0.9, 0.7, 0.31, 0.8, 0.1, 0.4, 0.6])
# y_true = np.array([   1,   0,   0,   1,   0,   0,    1,   0,   0,   1])
# users_id = np.array([ 2,   1,   0,   2,   1,   0,    0,   2,   1,   1])

# print('auc: ', auc_score(y_true, y_pred))
# print('gauc: ', gauc_score(y_true, y_pred, users_id))
# print('log_loss: ', log_loss(y_true, y_pred))

# for mt in ['ndcg', 'mrr', 'recall', 'hit','s']:
# 	tm = topk_metrics(y_true, y_pred, users_id, 3, metric_type=mt)
# 	print(f'{mt}: {tm}')
# y_pred = [[0, 1],[0, 1],[2, 3]]
# y_true = [[1, 2],[0, 1, 2],[2, 3]]
# out = topk_metrics(y_true, y_pred, topKs=(1,2))
# print(out)

