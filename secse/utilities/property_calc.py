#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author: shien
@file: property_calc.py 
@time: 2024/5/7/16:50
"""
import argparse
import os
import sys
import time
import rdkit.Chem as Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcNumHBD, CalcNumHBA, CalcNumRotatableBonds
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import QED
from rdkit.Chem import RDConfig
import json

sys.path.append(os.getenv("SECSE"))
from utilities.ring_tool import RingSystems
from utilities.substructure_filter import StructureFilter
from utilities.wash_mol import wash_mol, neutralize, charge_mol, get_keen_rotatable_bound_num, get_rigid_body_num

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


class Calculator:
    def __init__(self):

        self.input_smiles = None
        self.mol = None
        self.pains_smarts = None
        self.MW = None
        self.logP = None
        self.chiral_center = None
        self.heteroatom_ratio = None
        self.rdkit_rotatable_bound_num = None
        self.keen_rotatable_bound_num = None
        self.rigid_body_num = None
        self.hbd = None
        self.hba = None
        self.tpsa = None
        self.lipinski_violation = None
        self.qed = None
        self.max_ring_size = None
        self.max_ring_system_size = None
        self.ring_system_count = None
        self.bridged_site_count = None
        self.spiro_site_count = None
        self.fused_site_count = None
        self.rdkit_sa_score = None

    def load_mol(self, input_smiles):
        self.clean()
        self.input_smiles = input_smiles
        self.mol = Chem.MolFromSmiles(self.input_smiles)

        # uncharged each atom
        if self.input_smiles.count("-") + self.input_smiles.count("+") > 0:
            self.mol, self.input_smiles = neutralize(self.input_smiles)

        if self.mol is None:
            self.input_smiles = wash_mol(self.input_smiles)
            self.mol = Chem.MolFromSmiles(self.input_smiles)
            if self.mol is None:
                raise Exception("{} is Not Valid".format(input_smiles))

    def clean(self):
        self.input_smiles = None
        self.mol = None

    def pp_calc(self):
        """
        property filter
        """
        violation_counter = 0

        mw = CalcExactMolWt(self.mol)
        self.MW = mw
        if mw > 500:
            violation_counter += 1

        mol_hbd = CalcNumHBD(self.mol)
        self.hbd = mol_hbd
        if mol_hbd > 5:
            violation_counter += 1

        mol_hba = CalcNumHBA(self.mol)
        self.hba = mol_hba
        if mol_hba > 10:
            violation_counter += 1

        logp = Descriptors.MolLogP(self.mol)
        self.logP = logp
        if logp > 5:
            violation_counter += 1

        self.lipinski_violation = violation_counter

        self.tpsa = Descriptors.TPSA(self.mol)

        self.rdkit_rotatable_bound_num = CalcNumRotatableBonds(self.mol)
        # rotatable bound customized @dalong
        self.keen_rotatable_bound_num = get_keen_rotatable_bound_num(self.mol)
        # rotatable bound customized @dalong
        self.rigid_body_num = get_rigid_body_num(self.mol)

    def load_pains(self):
        # read smarts for pains
        with open(os.path.join(os.getenv("SECSE"), 'growing/pains_smarts.json')) as f:
            data = json.load(f)
        pains_smarts = dict((k, Chem.MolFromSmarts(v)) for k, v in data.items())
        self.pains_smarts = pains_smarts

    def alert_calc(self):
        self.load_pains()
        for name in self.pains_smarts:
            sma = self.pains_smarts[name]
            if self.mol.HasSubstructMatch(sma):
                print("PAINS not pass for ", sma)
        print("PAINS pass PASS")

    def ring_system_calc(self):
        ring_sys = RingSystems(self.mol)
        self.max_ring_system_size = max(ring_sys.ring_systems_size() + [-1])
        self.bridged_site_count = max(ring_sys.bridged_site_count())
        self.spiro_site_count = max(ring_sys.spiro_site_count())
        self.fused_site_count = max(ring_sys.fused_site_count())
        self.ring_system_count = len(ring_sys.systems)

    def custom_calc(self):
        # add Chiral center filter, cycle size less than 7, remove 3 continues hetero-atom
        chiral_tags = Chem.FindMolChiralCenters(self.mol, includeUnassigned=True, useLegacyImplementation=True)
        # the maximum number of chiral center <= 3
        self.chiral_center = len(chiral_tags)

        chiral_atom_list = set([x[0] for x in chiral_tags])
        rings = self.mol.GetRingInfo().AtomRings()

        if rings:
            # the maximum of ring size <= 7
            mol_max_ring_size = max([len(x) for x in rings])
            self.max_ring_size = mol_max_ring_size

            if len(chiral_tags) == 3:
                # 3 CCs should not in the same ring
                for ring in rings:
                    if len(set(ring).intersection(chiral_atom_list)) >= 3:
                        print("chiral center in one ring >2")

    def heteroatom_calc(self):
        hetero_ratio = Chem.rdMolDescriptors.CalcNumHeteroatoms(self.mol) / self.mol.GetNumHeavyAtoms()
        self.heteroatom_ratio = hetero_ratio

    def QED_calc(self):
        self.qed = QED.qed(self.mol)

    def SA_calc(self):
        sa_score = sascorer.calculateScore(self.mol)
        self.rdkit_sa_score = sa_score

    def run(self):
        self.pp_calc()
        self.alert_calc()
        self.ring_system_calc()
        self.custom_calc()
        self.heteroatom_calc()
        self.QED_calc()
        self.SA_calc()

    def print_attributes(self):
        attributes = vars(self)
        for attr_name, attr_value in attributes.items():
            if attr_name in ['pains_smarts', 'mol']:
                pass
            else:
                print(f"{attr_name}: {attr_value}")


if __name__ == '__main__':
    #args = parser.parse_args()
    time1 = time.time()
    calc_instance = Calculator()
    input_smiles = "O=C1CC(C2=CC(C3=CC=CN=C3)=CC(C4CC4)=C2)C(C)CC5=CC=CC=C51"
    calc_instance.load_mol(input_smiles)
    calc_instance.run()
    calc_instance.print_attributes()
    time2 = time.time()
    print(f"Time taken: {round((time2 - time1) / 60, 2)} min")
