import numpy as np
from rdkit import Chem


def find_identical_atoms(rdkit_mol, atom_list):
    len_list = len(atom_list)

    atom_rank = list(Chem.CanonicalRankAtoms(rdkit_mol, breakTies=False))
    for idx, atom in enumerate(rdkit_mol.GetAtoms()):
        if atom.GetIdx() in atom_list[:len_list]:
            sym_atoms = [
                int(atom_idx)
                for atom_idx, ranking in enumerate(atom_rank)
                if ranking == atom_rank[idx] and atom_idx not in atom_list
            ]
            atom_list.extend(sym_atoms)
    return atom_list


def find_identical_atoms_with_scores(rdkit_mol, atom_list, scores_list):
    len_list = len(atom_list)

    atom_rank = list(Chem.CanonicalRankAtoms(rdkit_mol, breakTies=False))
    for idx, atom in enumerate(rdkit_mol.GetAtoms()):
        if atom.GetIdx() in atom_list[:len_list]:
            sym_atoms = [
                int(atom_idx)
                for atom_idx, ranking in enumerate(atom_rank)
                if ranking == atom_rank[idx] and atom_idx not in atom_list
            ]
            atom_list.extend(sym_atoms)
            scores_list.extend(
                [scores_list[atom_list[:len_list].index(atom.GetIdx())]] * len(sym_atoms)
            )
    return atom_list, scores_list


def remove_identical_atoms(rdkit_mol, atom_list):
    idx_list = []
    rank_kept = []
    atom_rank = list(Chem.CanonicalRankAtoms(rdkit_mol, breakTies=False))
    for idx, atom in enumerate(atom_list):
        if atom_rank[atom] not in rank_kept:
            rank_kept.append(atom_rank[atom])
            idx_list.append(idx)

    atom_list = np.array(atom_list)[idx_list].tolist()

    return atom_list
