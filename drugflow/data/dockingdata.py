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
from typing import Text, Dict

import dataclasses
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class DockingExamples:
    name:Text
    ligand_path: Text
    receptor_path: Text
    autobox_ligand: Text = None
    process_ligand_path:Text = None
    process_receptor_path:Text = None
    config_path: Text = None
    output_path:Text = None
    middleware:Dict = dataclasses.field(default_factory=dict)
    metirics:Dict = dataclasses.field(default_factory=dict)
    success:int = 0

class DockingData(object):
    def __init__(self) -> None:
        self.data = []
        self.index = -1
        
    def load_data(self, 
                  path:Text, 
                  prefix={"ligand":"ligand", "receptor":"protein_processed", "config":"config.txt"}, 
                  format={"ligand":"sdf", "receptor":"pdb"}):
                      
        file_name_list = os.listdir(path)
        for name in tqdm(file_name_list):
            ligand_path = os.path.join(path, name, f"{name}_{prefix['ligand']}.{format['ligand']}")
            receptor_path = os.path.join(path, name,  f"{name}_{prefix['receptor']}.{format['receptor']}")
            config_path = os.path.join(path, name, prefix['config'])
            
            if not os.path.exists(config_path):
                config_path = None  

            self.data.append(DockingExamples(name=name,
                                             ligand_path=ligand_path, 
                                             receptor_path=receptor_path, 
                                             config_path=config_path))
            
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
    