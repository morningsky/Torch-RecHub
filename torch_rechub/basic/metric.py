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
	user_pred = get_user_pred(y_true, y_pred, users)
	for u in user_pred:
		idx = np.argsort(user_pred[u][1])[::-1][:k]
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



def ndcg_score(user_pred, k):
	"""compute NDCG
	Args:
		user_pred: dict, computed by get_user_topk()
	"""
	rank = np.arange(1, k+1, 1)
	idcgs = 1. / np.log2(rank + 1)
	idcg = sum(idcgs)
	score = 0
	for u in user_pred:
		dcgs = idcgs[np.where(user_pred[u][0] == 1)]
		dcg = sum(dcgs)
		score += dcg / idcg
	return score / len(user_pred.keys())


def hit_score(user_pred):
	score = 0
	for u in user_pred:
		if 1 in user_pred[u][0]:
			score += 1.0
	return score / len(user_pred.keys())


def mrr_score(user_pred):
	score = 0
	for u in user_pred:
		if 1 in user_pred[u][0]:
			score += 1.0 / (np.where(user_pred[u][0] == 1)[0][0] + 1)
	return score / len(user_pred.keys())


def recall_score(user_pred):
	score = 0
	for u in user_pred:
		score += sum(user_pred[u][0]) * 1. / len(user_pred[u][0])
	return score / len(user_pred.keys())


def topk_metrics(y_true, y_pred, users, k, metric_type):
	"""choice topk metrics and compute it
	the metrics contains 'ndcg', 'mrr', 'recall' and 'hit'

	Args:
		y_true: array, dim(N, ), all true labels of the data
		y_pred: array, dim(N, ), the predicted score
		k: int, the number of topk
		metric_type: string, choice the metric, 
		it can be lowercase 'ndcg' or uppercase 'NDCG' or 'Ndcg' and so on

	Return:
		the score of topk metric

	"""
	assert len(y_true) == len(y_pred) and len(y_true) == len(users)

	user_pred = get_user_topk(y_true, y_pred, users, k)
	if metric_type.lower() == 'ndcg':
		return ndcg_score(user_pred, k)
	elif metric_type.lower() == 'mrr':
		return mrr_score(user_pred)
	elif metric_type.lower() == 'recall':
		return recall_score(user_pred)
	elif metric_type.lower() == 'hit':
		return hit_score(user_pred)
	else:
		raise ValueError('metric_type error, choice from \'ndcg\', \'mrr\', \'recall\', \'hit\'')	



def log_loss(y_true, y_pred):
	score = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
	return -score.sum() / len(y_true)


# y_pred = np.array([0.3, 0.2, 0.5, 0.9, 0.7, 0.31, 0.8, 0.1, 0.4, 0.6])
# y_true = np.array([1,   0,   0,   1,   0,   0,    1,   0,   0,   1])
# users_id = np.array([ 2,   1,   0,   2,   1,   0,    0,   2,   1,   1])

# print('auc: ', auc_score(y_true, y_pred))
# print('gauc: ', gauc_score(y_true, y_pred, users_id))
# print('log_loss: ', log_loss(y_true, y_pred))

# for mt in ['ndcg', 'mrr', 'recall', 'hit','s']:
# 	tm = topk_metrics(y_true, y_pred, users_id, 3, metric_type=mt)
# 	print(f'{mt}: {tm}')
