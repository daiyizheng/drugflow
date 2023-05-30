# -*- encoding: utf-8 -*-
'''
Filename         :pdbbind.py
Description      :
Time             :2023/05/30 15:04:06
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from __future__ import absolute_import, print_function, annotations
import logging

from drugflow.data.dockingdata import DockingData

logger = logging.getLogger(__name__)


class PdbBind(DockingData):
    def __init__(self, path) -> None:
        self.path = path
        super(PdbBind, self).__init__()
        self.load_data(path)
    

        