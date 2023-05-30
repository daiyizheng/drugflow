# -*- encoding: utf-8 -*-
'''
Filename         :dockingdata.py
Description      :
Time             :2023/05/30 15:09:37
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from __future__ import absolute_import, print_function, annotations
import logging, os

from tqdm import tqdm

logger = logging.getLogger(__name__)


class DockingData(object):
    def __init__(self) -> None:
        self.data = []
        self.index = -1
        
    def load_data(self, path):
        file_name_list = os.listdir(path)
        for name in tqdm(file_name_list):
            ligand_path = os.path.join(path, name, name + "_ligand.sdf")
            receptor_path = os.path.join(path, name,  name + "_protein_processed.pdb")
            config_path = os.path.join(path, name, "config.txt")
            
            if not os.path.exists(config_path):
                config_path = None
            
            self.data.append((ligand_path, receptor_path, config_path, None))
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.index += 1
        if self.index<len(self.data):
            return self.data[self.index]
        else:
            raise StopIteration
    