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

logger = logging.getLogger(__file__)
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