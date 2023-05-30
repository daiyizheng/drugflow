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

from drugflow.utils.common import timing

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
            ):
        
        
        cmd = [
            self.binary_path,
            '--receptor', receptor_path,
            '--ligand', ligand_path,# 输出
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
            cmd += ['--size_x', size_x,
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
        with timing(f'Gnina docking, file_name: {file_name}'):
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


