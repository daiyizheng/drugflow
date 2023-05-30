# -*- encoding: utf-8 -*-
'''
Filename         :common.py
Description      :
Time             :2023/05/19 13:26:33
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
from __future__ import absolute_import, annotations, print_function
from typing import Optional
import tempfile, shutil, logging, time
import contextlib

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol


logger = logging.getLogger(__name__)
# logging
separator = ">" * 30
line = "-" * 30


@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None):
  """创建上下文管理器，创建和删除临时文件"""
  tmpdir = tempfile.mkdtemp(dir=base_dir)
  try:
    yield tmpdir
  finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

@contextlib.contextmanager
def timing(msg: str):#计算时间差
  logger.info('Started %s', msg)
  tic = time.time()
  yield
  toc = time.time()
  logger.info('Finished %s in %.3f seconds', msg, toc - tic)
  

def mol_conformers(mol:Mol):
  mol.RemoveAllConformers()
  ps = AllChem.ETKDGv2()
  id = AllChem.EmbedMolecule(mol, ps) # 生成3维几何构象
  if id == -1:
    logger.info('rdkit pos could not be generated without using random pos. using random pos now.')
    ps.useRandomCoords = True
    AllChem.EmbedMolecule(mol, ps)
    AllChem.MMFFOptimizeMolecule(mol, confId=0)
  return mol