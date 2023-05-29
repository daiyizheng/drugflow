# -*- encoding: utf-8 -*-
'''
Filename         :molecule.py
Description      :
Time             :2023/05/22 09:28:59
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
Desc:            :Code come from https://github.com/DeepGraphLearning/torchdrug/blob/master/torchdrug/data/molecule.py
'''
from __future__ import print_function, absolute_import, annotations

import torch
from rdkit import Chem
from torchdrug import data
from torchdrug.core import Registry as R

class CustomMolecule(data.Molecule):
    @classmethod
    def from_molecule(cls, 
                      mol, 
                      atom_feature="default", 
                      bond_feature="default", 
                      mol_feature=None,
                      with_hydrogen=False, 
                      kekulize=False):
        
        if mol is None:
            mol = cls.empty_mol

        if with_hydrogen: ## 原来pdb文件是不带H,是否加H
            mol = Chem.AddHs(mol)
        if kekulize:# 是否去除芳香性
            Chem.Kekulize(mol)

        atom_feature = cls._standarize_option(atom_feature)
        bond_feature = cls._standarize_option(bond_feature)
        mol_feature = cls._standarize_option(mol_feature)
        ## ------------------------原子特征-------------------------- ##
        atom_type = []
        formal_charge = []
        explicit_hs = []
        chiral_tag = []
        radical_electrons = []
        atom_map = []
        _atom_feature = []
        atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())] + [cls.dummy_atom]# 按照索引拿到原子对象
        for atom in atoms:
            atom_type.append(atom.GetAtomicNum()) # 原子类型的编号
            formal_charge.append(atom.GetFormalCharge()) # 原子电荷
            explicit_hs.append(atom.GetNumExplicitHs()) # 用于获取原子周围的显式氢原子数。
            chiral_tag.append(atom.GetChiralTag()) # 获取原子手性信息
            radical_electrons.append(atom.GetNumRadicalElectrons()) # 原子的原子基电子数
            atom_map.append(atom.GetAtomMapNum()) # map id 原子smarts形式冒号后面的数字,如[N:4], map id 就是4, 就是原子按顺序的序列编号
            feature = []
            for name in atom_feature:
                try:
                    func = R.get("features.atom.%s" % name)# 原子特征
                except :
                    func = R.get(name)
                feature += func(atom)
            _atom_feature.append(feature)
        atom_type = torch.tensor(atom_type)[:-1] # 去掉最后一个，上一步添加的 [cls.dummy_atom]
        atom_map = torch.tensor(atom_map)[:-1]
        formal_charge = torch.tensor(formal_charge)[:-1]
        explicit_hs = torch.tensor(explicit_hs)[:-1]
        chiral_tag = torch.tensor(chiral_tag)[:-1]
        radical_electrons = torch.tensor(radical_electrons)[:-1]
        if mol.GetNumConformers() > 0: # 分子异构体数量
            node_position = torch.tensor(mol.GetConformer().GetPositions())#
        else:
            node_position = None
        if len(atom_feature) > 0:
            _atom_feature = torch.tensor(_atom_feature)[:-1]
        else:
            _atom_feature = None
        ## ------------------------键特征-------------------------- ##
        edge_list = []
        bond_type = []
        bond_stereo = []
        stereo_atoms = []
        _bond_feature = []
        bonds = [mol.GetBondWithIdx(i) for i in range(mol.GetNumBonds())] + [cls.dummy_bond]# 获取所有键对象
        for bond in bonds:
            type = str(bond.GetBondType()) # 键的类型
            stereo = bond.GetStereo()
            if stereo:
                _atoms = [a for a in bond.GetStereoAtoms()]
            else:
                _atoms = [0, 0]
            if type not in cls.bond2id:
                continue
            type = cls.bond2id[type]
            h, t = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx() # 得到键的开始索引和结束索引
            edge_list += [[h, t, type], [t, h, type]] # 无向图
            # always explicitly store aromatic bonds, no matter kekulize or not 始终明确存储芳香键，无论是否 kekulize
            if bond.GetIsAromatic(): # 判断原子是否是芳香性原子
                type = cls.bond2id["AROMATIC"]
            bond_type += [type, type]
            bond_stereo += [stereo, stereo]
            stereo_atoms += [_atoms, _atoms]
            feature = []
            for name in bond_feature:
                try:
                    func = R.get("features.bond.%s" % name)
                except :
                    func = R.get(name)
                feature += func(bond)
            _bond_feature += [feature, feature]
        edge_list = edge_list[:-2]
        bond_type = torch.tensor(bond_type)[:-2]
        bond_stereo = torch.tensor(bond_stereo)[:-2]
        stereo_atoms = torch.tensor(stereo_atoms)[:-2]
        if len(bond_feature) > 0:
            _bond_feature = torch.tensor(_bond_feature)[:-2]
        else:
            _bond_feature = None

        _mol_feature = []
        for name in mol_feature:#分子特征
            try:
                func = R.get("features.molecule.%s" % name)
            except :
                func = R.get(name)
            _mol_feature += func(mol)
        if len(mol_feature) > 0:
            _mol_feature = torch.tensor(_mol_feature)
        else:
            _mol_feature = None

        num_relation = len(cls.bond2id) - 1 if kekulize else len(cls.bond2id)
        return cls(edge_list, # 边
                   atom_type, # 原子类型
                   bond_type, # 键类型
                   formal_charge=formal_charge, # 原子电荷
                   explicit_hs=explicit_hs, # 用于获取原子周围的显式氢原子数。
                   chiral_tag=chiral_tag, 
                   radical_electrons=radical_electrons, 
                   atom_map=atom_map, # 原子顺序编号
                   bond_stereo=bond_stereo, 
                   stereo_atoms=stereo_atoms, 
                   node_position=node_position, # 原子3维信息
                   atom_feature=_atom_feature, # 原子特征
                   bond_feature=_bond_feature, # 分子键特征
                   mol_feature=_mol_feature, # 分子特征
                   num_node=mol.GetNumAtoms(), # 分子数量 
                   num_relation=num_relation) # 分子键个数
