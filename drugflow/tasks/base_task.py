# -*- encoding: utf-8 -*-
'''
Filename         :base_task.py
Description      :
Time             :2023/05/29 14:38:01
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from __future__ import annotations, print_function, absolute_import
import logging

from torchdrug import tasks

logger = logging.getLogger(__name__)

class DeepTask(tasks.Task):
    ...


class TraditionalTask(object):
    def preprocess(self, dataset):
        pass
    
    def predict(self, inputs, metric=None):
        raise NotImplementedError
    
    def evaluate(self, results):
        raise NotImplementedError