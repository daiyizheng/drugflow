# -*- encoding: utf-8 -*-
'''
Filename         :docking.py
Description      :
Time             :2023/05/29 13:14:30
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from __future__ import annotations, print_function, absolute_import

import logging, os

import numpy as np
from biopandas.pdb import PandasPdb
from scipy.spatial.distance import cdist

from drugflow.utils.common import mol_conformers
from drugflow.utils.io import read_molecule, write_molecule
from drugflow.tasks.base_task import TraditionalTask 

logger = logging.getLogger(__name__)


class CommonDockingTask(TraditionalTask):
    def __init__(self, model, task, metric) -> None:
        super().__init__()
        self.model = model # 模型
        self.task = task # 任务名称
        self.metric = metric # 评价指标
    
    def ligand_preprocess(self, ligand_path, **kwargs):
        ## 配体位置随机化   
        mol = read_molecule(ligand_path)
        mol = mol_conformers(mol) # 随机生成构象
        file = os.path.basename(ligand_path)
        file_name, _ = os.path.splitext(file)
        ligand_path = os.path.join(os.path.dirname(ligand_path), file_name+"_3D_rdkit_process.pdb")
        write_molecule(mol, ligand_path) 
        return ligand_path
    
    def receptor_preprocess(self, 
                            receptor_path, 
                            **kwargs):
        return receptor_path
    
    def pocket_preprocess(self, 
                          ligand_path:str, 
                          receptor_path:str, 
                          pocket_cutoff:int=5.0,
                          autobox_add:int = 4
                          ):
        
        lig = read_molecule(ligand_path)
        rec = PandasPdb().read_pdb(receptor_path)
        rec_df = rec.get(s='c-alpha')
        rec_pos = rec_df[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
        lig_pos = lig.GetConformer().GetPositions()
        d = cdist(rec_pos, lig_pos)
        label = np.any(d < pocket_cutoff, axis=1)
        
        if np.any(label):
            center_pocket = rec_pos[label].mean(axis=0)
        else:
            logger.info("No pocket residue below minimum distance ", pocket_cutoff, "taking closest at", np.min(d))
            center_pocket = rec_pos[np.argmin(np.min(d, axis=1)[0])]
        
        radius_pocket = np.max(np.linalg.norm(lig_pos - center_pocket[None, :], axis=1))
        diameter_pocket = radius_pocket * 2
        center_x = center_pocket[0]
        size_x = diameter_pocket + autobox_add*2
        center_y = center_pocket[1]
        size_y = diameter_pocket + autobox_add*2
        center_z = center_pocket[2]
        size_z = diameter_pocket + autobox_add*2
        return center_x, size_x, center_y, size_y, center_z, size_z
    
    def predict(self, 
                data,
                autobox_add = 4, 
                autobox_ligand:str = None,
                use_pocket:bool = False,
                center_x:int=None, 
                size_x:int=None, 
                center_y:int=None, 
                size_y:int=None, 
                center_z:int=None, 
                size_z:int=None,
                pocket_cutoff:int = 5.0):
        
        ligand_path, receptor_path, config_path, autobox_ligand = data
        process_ligand_path = self.ligand_preprocess(ligand_path)
        process_receptor_path = self.receptor_preprocess(receptor_path)
        file = os.path.basename(ligand_path)
        file_name, _ = os.path.splitext(file)
        output_path = os.path.join(os.path.dirname(process_ligand_path), "out_"+file_name+"_"+self.task+".pdb")
        
        if use_pocket and size_x is None:
            center_x, size_x, center_y, size_y, center_z, size_z = self.pocket_preprocess(ligand_path, receptor_path, pocket_cutoff=pocket_cutoff, autobox_add=autobox_add)
        elif not use_pocket and size_x is None:
            autobox_ligand = process_receptor_path
        
        self.model.run(ligand_path = process_ligand_path, 
                       receptor_path = process_receptor_path,
                       output_path = output_path, 
                       config_path = config_path,
                       autobox_add = autobox_add,
                       autobox_ligand = autobox_ligand,
                       use_pocket = use_pocket,
                       center_x = center_x,
                       size_x = size_x,
                       center_y = center_y,
                       size_y = size_y,
                       center_z = center_z,
                       size_z = size_z)
        
        return ligand_path, receptor_path, output_path
    
    def evaluate(self, dataset):
        return 
    
