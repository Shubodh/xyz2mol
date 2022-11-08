"""
GIVEN BO Bond order file as input:
Module for generating rdkit molobj/smiles/molecular graph from free atoms

"""

import copy
import itertools

from collections import defaultdict

import numpy as np
import networkx as nx

import sys

import helper_graph as helg #import BOadj_to_BOmat

global __ATOM_LIST__
__ATOM_LIST__ = \
    ['h',  'he',
     'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne',
     'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar',
     'k',  'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu',
     'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',
     'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag',
     'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe',
     'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy',
     'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt',
     'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn',
     'fr', 'ra', 'ac', 'th', 'pa', 'u',  'np', 'pu']


global atomic_valence
global atomic_valence_electrons

atomic_valence = defaultdict(list)
atomic_valence[1] = [1]
atomic_valence[5] = [3,4]
atomic_valence[6] = [4]
atomic_valence[7] = [3,4]
atomic_valence[8] = [2,1,3]
atomic_valence[9] = [1]
atomic_valence[14] = [4]
atomic_valence[15] = [5,3] #[5,4,3]
atomic_valence[16] = [6,3,2] #[6,4,2]
atomic_valence[17] = [1]
atomic_valence[32] = [4]
atomic_valence[35] = [1]
atomic_valence[53] = [1]

atomic_valence_electrons = {}
atomic_valence_electrons[1] = 1
atomic_valence_electrons[5] = 3
atomic_valence_electrons[6] = 4
atomic_valence_electrons[7] = 5
atomic_valence_electrons[8] = 6
atomic_valence_electrons[9] = 7
atomic_valence_electrons[14] = 4
atomic_valence_electrons[15] = 5
atomic_valence_electrons[16] = 6
atomic_valence_electrons[17] = 7
atomic_valence_electrons[32] = 4
atomic_valence_electrons[35] = 7
atomic_valence_electrons[53] = 7

def int_atom(atom):
    """
    convert str atom to integer atom
    """
    global __ATOM_LIST__
    #print(atom)
    atom = atom.lower()
    return __ATOM_LIST__.index(atom) + 1

def read_xyz_file(filename, look_for_charge=True):
    """
    """

    atomic_symbols = []
    xyz_coordinates = []
    charge = 0
    title = ""

    with open(filename, "r") as file:
        for line_number, line in enumerate(file):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                title = line
                if "charge=" in line:
                    charge = int(line.split("=")[1])
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x), float(y), float(z)])

    atoms = [int_atom(atom) for atom in atomic_symbols]

    return atoms, charge, xyz_coordinates

def BO2mol(mol, BO_matrix, atoms, atomic_valence_electrons,
           mol_charge, allow_charged_fragments=True,  use_atom_maps=False):
    """
    based on code written by Paolo Toscani

    From bond order, atoms, valence structure and total charge, generate an
    rdkit molecule.

    args:
        mol - rdkit molecule
        BO_matrix - bond order matrix of molecule
        atoms - list of integer atomic symbols
        atomic_valence_electrons -
        mol_charge - total charge of molecule

    optional:
        allow_charged_fragments - bool - allow charged fragments

    returns
        mol - updated rdkit molecule with bond connectivity

    """

    l = len(BO_matrix)
    l2 = len(atoms)
    BO_valences = list(BO_matrix.sum(axis=1))

    if (l != l2):
        raise RuntimeError('sizes of adjMat ({0:d}) and Atoms {1:d} differ'.format(l, l2))

    rwMol = Chem.RWMol(mol)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE
    }

    for i in range(l):
        for j in range(i + 1, l):
            bo = int(round(BO_matrix[i, j]))
            if (bo == 0):
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.AddBond(i, j, bt)

    mol = rwMol.GetMol()

    if allow_charged_fragments:
        mol = set_atomic_charges(
            mol,
            atoms,
            atomic_valence_electrons,
            BO_valences,
            BO_matrix,
            mol_charge,
            use_atom_maps)
    else:
        mol = set_atomic_radicals(mol, atoms, atomic_valence_electrons, BO_valences,
                                                            use_atom_maps)

    return mol



def BOfile_to_BOadj(BOfile): #
    BO = open(BOfile,"r").read().split('CSD_code = ')
    # print
    BO = [i.splitlines()[:-1] for i in BO[1:]]

    bond_orders = {}
    csd_codes = []
    for mol in BO:
        res = {}
        # if 'Fe' in mol[1]:
        if True:
            csd_codes.append(mol[0])
            for k in mol[1:]:
                k = k.split()
                p_idx = int(k[0])-1
                p_atom = k[1]
                i = 3
                while i < len(k)-1:
                    c_atom, c_idx, bo = k[i], int(k[i+1])-1, float(k[i+2])
                    # print(f'{c_atom}, {c_idx}, {bo}')
                    res[frozenset([c_idx, p_idx])] = bo
                    i += 3
            bond_orders[csd_codes[-1]] = res
    
    # print(list(bond_orders.items())[:1][0][1])
    # TODO: Check if it is loading all atoms or ignoring the last ones.
    # print(bond_orders)
    return bond_orders


def xyz_bo_2mol(atoms, coordinates, charge=0, allow_charged_fragments=True,
            use_graph=True, use_huckel=False, embed_chiral=True,
            use_atom_maps=False):
    """
    Generate a rdkit molobj from atoms, coordinates and a total_charge.

    args:
        atoms - list of atom types (int)
        coordinates - 3xN Cartesian coordinates
        charge - total charge of the system (default: 0)

    optional:
        allow_charged_fragments - alternatively radicals are made
        use_graph - use graph (networkx)
        use_huckel - Use Huckel method for atom connectivity prediction
        embed_chiral - embed chiral information to the molecule

    returns:
        mols - list of rdkit molobjects

    """

    # Get atom connectivity (AC) matrix, list of atomic numbers, molecular charge,
    # and mol object with no connectivity information
    AC, mol = xyz2AC(atoms, coordinates, charge, use_huckel=use_huckel)

    # Convert AC to bond order matrix and add connectivity and charge info to
    # mol object
    new_mols = AC2mol(mol, AC, atoms, charge,
                     allow_charged_fragments=allow_charged_fragments,
                     use_graph=use_graph,
                     use_atom_maps=use_atom_maps)

    # Check for stereocenters and chiral centers
    if embed_chiral:
        for new_mol in new_mols:
            chiral_stereo_check(new_mol)

    return new_mols


if __name__ == "__main__":
    # read atoms and coordinates. Try to find the charge
    data_path = '/home/shubodh/OneDrive/mll_projects/2022/xyz2mol/examples/'
    full_bo_file = 'tmQM_X.BO'
    fe_bo_file = 'tmQM_X_Fe_mol.BO'
    fe_xyz_file = 'tmQM_X1_Fe_mol.xyz'

    BOfile = data_path + fe_bo_file
    xyz_filename = data_path + fe_xyz_file

    atoms, charge, xyz_coordinates = read_xyz_file(xyz_filename)
    BOadj = BOfile_to_BOadj(BOfile)
    BOmat, adjMatrix = helg.BOadj_to_BOmat(BOadj)

    print(BOmat)
    # import argparse

    # parser = argparse.ArgumentParser(usage='%(prog)s [options] molecule.xyz')
    # parser.add_argument('structure', metavar='structure', type=str)

    # BOfile_to_BOmat(BOfile)

    # mols = xyz_bo_2mol(atoms, xyz_coordinates,
    #     charge=charge,
    #     use_graph=quick,
    #     allow_charged_fragments=charged_fragments,
    #     embed_chiral=embed_chiral,
    #     use_huckel=use_huckel)

# if __name__ == "__main__":

#     import argparse

#     parser = argparse.ArgumentParser(usage='%(prog)s [options] molecule.xyz')
#     parser.add_argument('structure', metavar='structure', type=str)
#     parser.add_argument('-s', '--sdf',
#         action="store_true",
#         help="Dump sdf file")
#     parser.add_argument('--ignore-chiral',
#         action="store_true",
#         help="Ignore chiral centers")
#     parser.add_argument('--no-charged-fragments',
#         action="store_true",
#         help="Allow radicals to be made")
#     parser.add_argument('--no-graph',
#         action="store_true",
#         help="Run xyz2mol without networkx dependencies")

#     # huckel uses extended Huckel bond orders to locate bonds (requires RDKit 2019.9.1 or later)
#     # otherwise van der Waals radii are used
#     parser.add_argument('--use-huckel',
#         action="store_true",
#         help="Use Huckel method for atom connectivity")
#     parser.add_argument('-o', '--output-format',
#         action="store",
#         type=str,
#         help="Output format [smiles,sdf] (default=sdf)")
#     parser.add_argument('-c', '--charge',
#         action="store",
#         metavar="int",
#         type=int,
#         help="Total charge of the system")

#     args = parser.parse_args()

#     # read xyz file
#     filename = args.structure

#     # allow for charged fragments, alternatively radicals are made
#     charged_fragments = not args.no_charged_fragments

#     # quick is faster for large systems but requires networkx
#     # if you don't want to install networkx set quick=False and
#     # uncomment 'import networkx as nx' at the top of the file
#     quick = not args.no_graph

#     # chiral comment
#     embed_chiral = not args.ignore_chiral

#     # read atoms and coordinates. Try to find the charge
#     atoms, charge, xyz_coordinates = read_xyz_file(filename)

#     # huckel uses extended Huckel bond orders to locate bonds (requires RDKit 2019.9.1 or later)
#     # otherwise van der Waals radii are used
#     use_huckel = args.use_huckel

#     # if explicit charge from args, set it
#     if args.charge is not None:
#         charge = int(args.charge)

#     # Get the molobjs
#     mols = xyz_bo_2mol(atoms, xyz_coordinates,
#         charge=charge,
#         use_graph=quick,
#         allow_charged_fragments=charged_fragments,
#         embed_chiral=embed_chiral,
#         use_huckel=use_huckel)

#     # Print output
#     for mol in mols:
#         if args.output_format == "sdf":
#             txt = Chem.MolToMolBlock(mol)
#             print(txt)

#         else:
#             # Canonical hack
#             isomeric_smiles = not args.ignore_chiral
#             smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles)
#             m = Chem.MolFromSmiles(smiles)
#             smiles = Chem.MolToSmiles(m, isomericSmiles=isomeric_smiles)
#             print(smiles)
