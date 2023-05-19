# -*- encoding: utf-8 -*-
'''
Filename         :gnina.py
Description      :
Time             :2023/05/19 13:58:50
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
from __future__ import print_function, absolute_import, annotations

import logging 



logger = logging.getLogger(__file__)


class SminaDocking(Docking):
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

    def query(self, 
              receptor_path: str, 
              ligand_path:str, 
              ligand_out:str,
              smile_name: str=None, )->Any:

        with tmpdir_manager() as f:
            if self.ligand_process:
                ligand_path = self.ligand_process(item=ligand_path, f=f, smile_name=smile_name)
            
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

