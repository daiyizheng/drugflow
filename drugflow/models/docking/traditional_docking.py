# -*- encoding: utf-8 -*-
'''
Filename         :traditional_docking.py
Description      :
Time             :2023/05/29 15:33:26
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from __future__ import annotations, print_function, absolute_import
import logging, os
import subprocess
from typing import Any

import numpy as np
from biopandas.pdb import PandasPdb
from scipy.spatial.distance import cdist

from drugflow.utils.common import timing, mol_conformers
from drugflow.utils.io import read_molecule, write_molecule

logger = logging.getLogger(__name__)


class BaseDocking(object):
    def __init__(self, binary_path:str) -> None:
        self.binary_path = binary_path 
  
    def run(self, *args, **kwargs):
        raise NotImplementedError
    

class GninaDocking(BaseDocking):
    def __init__(self,
                 binary_path:str, 
                 n_cpu:str = "16",
                 no_gpu:bool=True,) -> None:
        super(GninaDocking, self).__init__(binary_path=binary_path)
        self.n_cpu = n_cpu
        self.no_gpu = no_gpu
    
    def ligand_preprocess(self, ligand_path, **kwargs):
        ## 配体位置随机化   
        mol = read_molecule(ligand_path)
        mol = mol_conformers(mol) # 随机生成构象
        file = os.path.basename(ligand_path)
        file_name, _ = os.path.splitext(file)
        ligand_path = os.path.join(os.path.dirname(ligand_path), file_name+"_3D_rdkit_process.pdb")
        write_molecule(mol, ligand_path) 
        return ligand_path
    
    def receptor_preprocess(self, receptor_path, **kwargs):
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
        
    def run(self,
            receptor_path:str,
            ligand_path:str,
            output_path:str, 
            config_path:str = None,
            autobox_add = 4,
            autobox_ligand = None,
            use_pocket:bool = False,
            center_x:str = None,
            size_x:str = None,
            center_y:str = None,
            size_y:str = None,
            center_z:str = None,
            size_z:str = None,
            pocket_cutoff:int = 5.0
            ):
        
        random_ligand_path = self.ligand_preprocess(ligand_path)
        
        cmd = [
            self.binary_path,
            '--receptor', receptor_path,
            '--ligand', random_ligand_path,# 输出
            '--out', output_path, # 输出
            '--cpu', self.n_cpu, ## cpu数量     
        ]
        
        if config_path:
            cmd += ['--config', config_path]
            
        if autobox_ligand:
            autobox_ligand = receptor_path
            cmd += ['--autobox_ligand', autobox_ligand,
                    '--autobox_add', autobox_add]
            
        if use_pocket:
            ## 计算口袋位置
            if size_x is None:
                center_x, size_x, center_y, size_y, center_z, size_z = \
                    self.pocket_preprocess(ligand_path, 
                                           receptor_path, 
                                           pocket_cutoff=pocket_cutoff,
                                           autobox_add=autobox_add) 
            
            cmd += ['--szie_x', size_x,
                    '--size_y', size_y,
                    '--size_z', size_z,
                    '--center_x', center_x,
                    '--center_y', center_y,
                    '--center_z', center_z]  
                  
        cmd = cmd if self.no_gpu else cmd+["--no_gpu"]    
        file = os.path.basename(ligand_path)
        file_name, _ = os.path.splitext(file)
        log_path = os.path.join(os.path.dirname(output_path), file_name+".log")
        cmd += ['--log', log_path]
        
        cmd = [str(c) for c in cmd]
        logging.info(f'Launching subprocess {" ".join(cmd)}')
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with timing('Gnina docking'):
            stdout, stderr = process.communicate()
            retcode = process.wait()
        if retcode:
            logger.error('Gnina failed:\nfile name:%s\n\nstdout:\n%s\n\nstderr:\n%s\n' % (file_name, stdout.decode('utf-8'), stderr[:100_000].decode('utf-8')))
            return  0
        return 1
         

class SminaDocking(object):
    def __init__(self,
                 binary_path:str, 
                 config:str=None,
                 autobox_ligand:str=None,
                 autobox_add:int = 4,
                 exhaustiveness:int=8,
                 ligand_process=None
                 ) -> None:
        
        self.binary_path = binary_path
        self.config = config
        self.autobox_ligand = autobox_ligand
        self.autobox_add = autobox_add
        self.exhaustiveness = exhaustiveness
        self.ligand_process = ligand_process
        if config is None and autobox_ligand is None:
            raise ValueError("config and autobox_ligand need one parameter")

    def run(self,
            receptor_path: str, 
            ligand_path:str, 
            ligand_out:str)->Any:
          
            file = os.path.basename(ligand_path)
            file_mane, _ = os.path.splitext(file)
            out_path = os.path.join(ligand_out, file_mane+"_"+ self.__class__.__name__ +"_out.sdf") 
            cmd = [self.binary_path,
                   '--config', self.config, 
                   '--receptor', receptor_path,
                   '--ligand', ligand_path,# 输出
                   '--out', out_path, # 输出
                    ] 

            if self.autobox_ligand:
                cmd += ["--autobox_ligand", self.autobox_ligand, 
                        "--autobox_add", self.autobox_add,
                        "--exhaustiveness", self.exhaustiveness]
            logging.info('Launching subprocess "%s"', ' '.join(cmd))
            process = subprocess.Popen(
                 cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with timing('Smina query'):
                stdout, stderr = process.communicate()
                retcode = process.wait()
            if retcode:
                logger.error('Smina failed:\nfile name:%s\n\nstdout:\n%s\n\nstderr:\n%s\n' % (file_mane,
                             stdout.decode('utf-8'), stderr[:100_000].decode('utf-8')))
                return  None
            return out_path


# if __name__ == '__main__':
#     binary_path = "/DYZ/dyz1/custom_package/drugflow/drugflow/resources/gnina"
#     receptor_path = "/DYZ/dyz1/custom_package/drugflow/results/6qqw_protein_processed.pdb"
#     ligand_path = "/DYZ/dyz1/custom_package/drugflow/results/6qqw_rdkit_ligand.pdb"
#     out_path = "/DYZ/dyz1/custom_package/drugflow/results/6qqw_baseline_ligand.pdb"
#     config_path = "/DYZ/dyz1/custom_package/drugflow/results/config.txt"

#     dk = GninaDocking(binary_path=binary_path)
#     dk.run(receptor_path=receptor_path,
#            ligand_path=ligand_path,
#            output_path=out_path,
#            config_path=config_path,
#            use_pocket=True)
