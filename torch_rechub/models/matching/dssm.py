"""
Date: create on 12/05/2022
References: 
    paper: (CIKM'2013) Learning Deep Structured Semantic Models for Web Search using Clickthrough Data
    url: https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch

from ...basic.layers import MLP, EmbeddingLayer


class DSSM(torch.nn.Module):
    """Deep Structured Semantic Model

    Args:
        user_features (list[Feature Class]): training by the user tower module.
        item_features (list[Feature Class]): training by the item tower module.
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
    """

    def __init__(self, user_features, item_features, user_params, item_params):
        super(DSSM, self).__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.user_dims = sum([fea.embed_dim for fea in user_features])
        self.item_dims = sum([fea.embed_dim for fea in item_features])

        self.embedding = EmbeddingLayer(user_features + item_features)
        self.user_tower = MLP(self.user_dims, **user_params)
        self.item_tower = MLP(self.item_dims, **item_params)

    def forward(self, x):
        input_user = self.embedding(x, self.user_features, squeeze_dim=True)  #[batch_size, num_features*deep_dims]
        input_item = self.embedding(x, self.fm_features, squeeze_dim=True)  #[batch_size, num_features*embed_dim]

        user_out = self.user_tower(input_user)
        item_out = self.item_tower(input_item)
        print(user_out.shape)
        y = torch.cosine_similarity(user_out, item_out, dim=1)
        return torch.sigmoid(y)