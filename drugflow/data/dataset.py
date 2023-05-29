# -*- encoding: utf-8 -*-
'''
Filename         :dataset.py
Description      :
Time             :2023/05/29 10:16:26
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from __future__ import absolute_import, print_function,annotations
import warnings
from collections import defaultdict
import logging

from tqdm import tqdm
from rdkit import Chem
from torchdrug import data

logger = logging.getLogger(__name__)

class CustomMoleculeDataset(data.MoleculeDataset):
    def load_smiles(self, 
                    smiles_list, 
                    targets, 
                    molecular_feature=data.Molecule, 
                    transform=None, 
                    lazy=False, 
                    verbose=0, 
                    **kwargs):
        num_sample = len(smiles_list)
        if num_sample > 1000000:
            warnings.warn("Preprocessing molecules of a large dataset consumes a lot of CPU memory and time. "
                          "Use load_smiles(lazy=True) to construct molecules in the dataloader instead.")
        for field, target_list in targets.items():
            if len(target_list) != num_sample:
                raise ValueError("Number of target `%s` doesn't match with number of molecules. "
                                 "Expect %d but found %d" % (field, num_sample, len(target_list)))

        self.transform = transform # 数据处理
        self.lazy = lazy # 懒加载数据
        self.kwargs = kwargs
        self.smiles_list = []
        self.data = []
        self.targets = defaultdict(list)

        if verbose:
            smiles_list = tqdm(smiles_list, "Constructing molecules from SMILES")
        for i, smiles in enumerate(smiles_list):
            if not self.lazy or len(self.data) == 0:
                mol = Chem.MolFromSmiles(smiles)# 转为分子对象
                if not mol:
                    logger.debug("Can't construct molecule from SMILES `%s`. Ignore this sample." % smiles)
                    continue
                mol = molecular_feature.from_molecule(mol, **kwargs)
            else:
                mol = None
            self.data.append(mol)
            self.smiles_list.append(smiles)
            for field in targets:
                self.targets[field].append(targets[field][i])