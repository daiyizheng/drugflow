# -*- encoding: utf-8 -*-
'''
Filename         :property_prediction.py
Description      :
Time             :2023/05/29 10:37:19
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
from __future__ import print_function, absolute_import, annotations
import logging

import torch
from torch.nn import functional as F
from torchdrug import core, tasks, metrics, layers

logger = logging.getLogger(__file__)

class PropertyPrediction(tasks.Task, core.Configurable):
    def __init__(self, 
                 model,#表征模型
                 task,# 任务
                 num_class, ## 类别
                 criterion="mse", ## loss
                 metric=("mae", "rmse"), ## 评价指标
                 num_mlp_layer=1, ## 映射层的个数
                 
                 ) -> None:
        super().__init__()
        self.model= model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer= num_mlp_layer
        self.num_class = num_class


    def preprocess(self, train_set, valid_set, test_set):
        ## 训练数据预处理


        ## 最后一层全链接层
        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims+[self.num_class])

    
    def forward(self, batch):
        """训练部分"""

        ## 记录总的损失
        metric = {}

        ## 训练部分
        pred = self.predict(batch=batch, all_loss=None, metric=metric)

        target = self.target(batch)
        labeled = ~torch.isnan(target) # 拿到标签值不是异常的索引, 这一步可以做特殊处理
        target[~labeled] = 0  # 

        ## loss 计算
        loss = F.cross_entropy(pred, target.long())
        metric['ce'] = loss
        return loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        output = self.model(graph=graph, input=graph.node_feature.float(), metric=metric, all_loss=all_loss)

        pred = self.mlp(output["graph_feature"])

        return pred

    def target(self, batch):
        target =  batch[self.task].float()
        return target

    def evaluate(self, pred, target):
        metric = {}
        labeled = ~torch.isnan(target)
        for _metric in self.metric:
            if _metric == "accuracy":
                score = metrics.accuracy(pred[labeled], target[labeled].long())
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric