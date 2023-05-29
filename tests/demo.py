# -*- encoding: utf-8 -*-
'''
Filename         :demo.py
Description      :
Time             :2023/05/29 11:06:58
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

import sys, os
sys.path.insert(0, "./")

from drugflow.data.dataset import CustomMoleculeDataset
from drugflow.data.molecule import CustomMolecule

class Mydata(CustomMoleculeDataset):
    target_fields = ["CT_TOX"]
    def __init__(self, path, verbose=1, **kwargs) -> None:
        super().__init__()
        if not os.path.exists(path):
            raise ValueError("暂时没有数据!!!")
        
        self.path = path

        self.load_csv(path, 
                      smiles_field="smiles", 
                      target_fields=self.target_fields,
                      verbose=verbose,
                      **kwargs)