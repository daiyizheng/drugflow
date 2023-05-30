# -*- encoding: utf-8 -*-
'''
Filename         :engine.py
Description      :
Time             :2023/05/29 15:14:47
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from __future__ import annotations, print_function, absolute_import
import logging

from torchdrug.core import engine
from tqdm import tqdm

logger = logging.getLogger(__file__)


class DeepEngine(engine.Engine):
    ...


class TraditionalEngine(object):
    def __init__(self, 
                 task, 
                 dataset) -> None:
        self.model = task
        self.dataset = dataset
    
    def predict(self, *args, **kwargs):
        results = []
        for data in tqdm(self.dataset):
            res = self.model.predict(data, **kwargs)    
            results.append(res)
        return results
     
    def evaluate(self, results):
        metrics = []
        for data in tqdm(results):
            self.model.evaluate(data)
            