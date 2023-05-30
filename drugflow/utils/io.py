# -*- encoding: utf-8 -*-
'''
Filename         :io.py
Description      :
Time             :2023/05/19 13:26:44
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from __future__ import print_function, absolute_import, annotations
import logging
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem, MolToPDBFile

logger = logging.getLogger(__name__)


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        print(e)
        print("RDKit was unable to read the molecule.")
        return None

    return mol


def write_molecule(mol, output_path):
    if output_path.endswith('.pdb'):
        MolToPDBFile(mol, output_path)
    elif output_path.endswith('.mol2'):
        pass
    elif output_path.endswith('.sdf'):
        pass
    elif output_path.endswith('.pdbqt'):
        pass
    else:
        raise ValueError('Expect the format of the output_path to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(output_path))