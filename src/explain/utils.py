from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Contrib.IFG.ifg import identify_functional_groups
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from dgl import graph

import itertools

def create_dirs(save_dir, num_classes=None, smiles=None):
    Path(f'{save_dir}/').mkdir(exist_ok=True)
    if smiles is not None:
        Path(f'{save_dir}/{smiles}/').mkdir(exist_ok=True)
        if num_classes is not None:
            for i in range(num_classes):
                Path(f'{save_dir}/{smiles}/class_{i}').mkdir(exist_ok=True)


    return

# check if ring is aromatic
def is_ring_aromatic(mol, bond_ring):
    for id in bond_ring:
        if not mol.GetBondWithIdx(id).GetIsAromatic():
            return False
    return True

# combine linker atoms into subgroups
def combine_linkers_from_atom_list(mol, l_idxs):
    natms = len(l_idxs)
    
    # identify bonds between all linker atoms
    bonds = []
    for i in range(natms):
        for j in range(i, natms):
            bond = mol.GetBondBetweenAtoms(l_idxs[i], l_idxs[j])
            if bond:
                bonds.append(bond)

    if len(bonds) == 0:
        return [[l_idx] for l_idx in l_idxs]
    
    # combine linkers which have a bond between them
    combined_ls = []
    for bond in bonds:
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()

        combined_l_idxs = [idx for idxs in combined_ls for idx in idxs]

        if atom1_idx not in combined_l_idxs and atom2_idx not in combined_l_idxs:
            combined_ls.append([atom1_idx, atom2_idx])

        elif atom1_idx in combined_l_idxs and atom2_idx not in combined_l_idxs:
            for i, combined_l in enumerate(combined_ls):
                if atom1_idx in combined_l:
                    combined_l.append(atom2_idx)
                    combined_ls[i] = combined_l

        elif atom1_idx not in combined_l_idxs and atom2_idx in combined_l_idxs:
            for i, combined_l in enumerate(combined_ls):
                if atom2_idx in combined_l:
                    combined_l.append(atom1_idx)
                    combined_ls[i] = combined_l

    # check for possible "duplicate" linkers
    combined_l_idxs = [idx for idxs in combined_ls for idx in idxs]
    assert len(combined_l_idxs) == len(set(combined_l_idxs)), 'Duplicate linkers!'

    # get linkers which have no bonds
    combined_l_idxs = [idx for idxs in combined_ls for idx in idxs]
    for l_idx in l_idxs:
        if l_idx not in combined_l_idxs:
            combined_ls.append([l_idx])

    return combined_ls 

def identify_aromatic_rings(mol, rings, fgs):
    # loop over all fgs identify if fgs are in heterocycles
    fgs_in_heterocyclics = []
    fgs_not_in_heterocyclics = []
    for fg_idx, fg in enumerate(fgs):
        fg_in_heterocycle = False
        for atom_idx in fg:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetIsAromatic() and atom.IsInRing():
                fg_in_heterocycle = True

        if fg_in_heterocycle:
            fgs_in_heterocyclics.append(fg_idx)
        else:
            fgs_not_in_heterocyclics.append(fg_idx)
                
    # get rings that our fg atoms are in
    fg_rings = []
    for fg_idx in fgs_in_heterocyclics:
        atom_rings = []
        fg = fgs[fg_idx]
        for atom_idx in fg:
            for ring_idx, ring in enumerate(rings.AtomRings()):
                if atom_idx in ring:
                    atom_rings.append(ring_idx)
        fg_rings.append(atom_rings)

    # get heterocyclic aromatic rings
    hars = []
    for ring_idxs in fg_rings:
        for ring_idx in ring_idxs:
            if list(rings.AtomRings()[ring_idx]) in hars:
                continue
            hars.append(list(rings.AtomRings()[ring_idx]))

    # get remaining aromatic rings
    ars = []
    for ring_idx, ring in enumerate(rings.AtomRings()):
        if is_ring_aromatic(mol, rings.BondRings()[ring_idx]) and list(ring) not in hars:
            ars.append(list(ring))


    # remove fgs that are now in aromatic heterocycles
    fgs = [fgs[i] for i in fgs_not_in_heterocyclics]


    return fgs, hars, ars

def identify_linkers(mol, fgs, hars, ars):
    # get atom idxs for fgs, hars, and ars
    fgs_idxs = [atom_idx for fg in fgs for atom_idx in fg]
    hars_idxs = [atom_idx for har in hars for atom_idx in har]
    ars_idxs = [atom_idx for ar in ars for atom_idx in ar]
    non_l_idxs = fgs_idxs + hars_idxs + ars_idxs

    # get non link atom idxs
    l_idxs = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in non_l_idxs:
            l_idxs.append(atom.GetIdx())

    l_idxs = combine_linkers_from_atom_list(mol, l_idxs)

    return l_idxs

def identify_connections_between_subgroups(mol, nodes):
    # find all bonds between different nodes
    subgroup_bonds = []
    for bond in mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()

        for i, subgroup in enumerate(nodes):
            if atom1_idx in subgroup:
                atom1_subgroup_idx = i
            if atom2_idx in subgroup:
                atom2_subgroup_idx = i
        
        if atom1_subgroup_idx == atom2_subgroup_idx:
            continue

        subgroup_bonds.append([atom1_subgroup_idx, atom2_subgroup_idx])

    subgroup_bonds.sort()
    subgroup_bonds = list(subgroup_bond for subgroup_bond, _ in itertools.groupby(subgroup_bonds))

    return subgroup_bonds

def construct_graph_from_subgroups(mol, nodes, edges):
    src_list = []
    dst_list = []
    for edge in edges:
        u = edge[0]
        v = edge[1]
        src_list.extend([u, v])
        dst_list.extend([v, u])

    g = graph((src_list, dst_list), num_nodes=len(nodes))

    return g

def generate_chem_subgraphs(smiles):
    # convert smiles to mol
    mol = Chem.MolFromSmiles(smiles)

    # identify all rings in molecule
    rings = mol.GetRingInfo()

    # identify functional groups in molecule
    fgs = identify_functional_groups(mol)
    fgs = [list(fg.atomIds) for fg in fgs] 
    fgs.sort()
    fgs = list(fg for fg, _ in itertools.groupby(fgs))

    # identify fgs in aromatic rings
    fgs, hars, ars = identify_aromatic_rings(mol, rings, fgs)

    # identify remaining linkers in molecule
    ls = identify_linkers(mol, fgs, hars, ars)

    # combine all subgroups
    nodes = fgs + hars + ars + ls

    # get node connections
    edges = identify_connections_between_subgroups(mol, nodes)

    dgl_graph = construct_graph_from_subgroups(mol, nodes, edges)

    return dgl_graph, nodes

def explain_edges_to_atoms(smiles, edges):
    mol = Chem.MolFromSmiles(smiles)

    atoms = []
    for i, bond in enumerate(mol.GetBonds()):
        if edges[i*2] > 0.8:        
            atom1 = bond.GetBeginAtomIdx()
            atom2 = bond.GetEndAtomIdx()

            atoms.append(atom1)
            atoms.append(atom2)

    return atoms

def sme_fg_masks(smiles):
    # get all functional groups
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    fgs = identify_functional_groups(mol)
    fgs = [list(fg.atomIds) for fg in fgs] 
    fgs.sort()
    fgs = list(fg for fg, _ in itertools.groupby(fgs))

    # remove fgs in aromatic rings
    rings = mol.GetRingInfo()
    fgs, _, _ = identify_aromatic_rings(mol, rings, fgs)

    masks = []
    for fg in fgs:
        mask = np.ones((num_atoms), dtype=np.float32)
        mask[fg] = 0.0
        masks.append(mask)

    return fgs, masks

def sme_brics_masks(smiles):
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    res = list(BRICS.FindBRICSBonds(mol))  # [((1, 2), ('1', '5'))]

    # return brics_bond
    all_brics_bond = [set(res[i][0]) for i in range(len(res))]

    all_brics_substructure_subset = dict()
    # return atom in all_brics_bond
    all_brics_atom = []
    for brics_bond in all_brics_bond:
        all_brics_atom = list(set(all_brics_atom + list(brics_bond)))

    if len(all_brics_atom) > 0:
        # return all break atom (the break atoms did'n appear in the same substructure)
        all_break_atom = dict()
        for brics_atom in all_brics_atom:
            brics_break_atom = []
            for brics_bond in all_brics_bond:
                if brics_atom in brics_bond:
                    brics_break_atom += list(set(brics_bond))
            brics_break_atom = [x for x in brics_break_atom if x != brics_atom]
            all_break_atom[brics_atom] = brics_break_atom

        substrate_idx = dict()
        used_atom = []
        for initial_atom_idx, break_atoms_idx in all_break_atom.items():
            if initial_atom_idx not in used_atom:
                neighbor_idx = [initial_atom_idx]
                substrate_idx_i = neighbor_idx
                begin_atom_idx_list = [initial_atom_idx]
                while len(neighbor_idx) != 0:
                    for idx in begin_atom_idx_list:
                        initial_atom = mol.GetAtomWithIdx(idx)
                        neighbor_idx = neighbor_idx + [neighbor_atom.GetIdx() for neighbor_atom in
                                                       initial_atom.GetNeighbors()]
                        exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i
                        if idx in all_break_atom.keys():
                            exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i + all_break_atom[idx]
                        neighbor_idx = [x for x in neighbor_idx if x not in exlude_idx]
                        substrate_idx_i += neighbor_idx
                        begin_atom_idx_list += neighbor_idx
                    begin_atom_idx_list = [x for x in begin_atom_idx_list if x not in substrate_idx_i]
                substrate_idx[initial_atom_idx] = substrate_idx_i
                used_atom += substrate_idx_i
            else:
                pass
    else:
        substrate_idx = dict()
        substrate_idx[0] = [x for x in range(mol.GetNumAtoms())]

    subgroups = [group for group in substrate_idx.values()]    
    
    masks = []
    for subgroup in subgroups:
        mask = np.ones((num_atoms), dtype=np.float32)
        mask[subgroup] = 0.0
        masks.append(mask)

    return subgroups, masks

def find_murcko_link_bond(mol):
    core = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_index = mol.GetSubstructMatch(core)
    link_bond_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        link_score = 0
        if u in scaffold_index:
            link_score += 1
        if v in scaffold_index:
            link_score += 1
        if link_score == 1:
            link_bond_list.append([u, v])
    return link_bond_list

def sme_murcko_masks(smiles):
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()

    # return murcko_link_bond
    all_murcko_bond = find_murcko_link_bond(mol)

    # return atom in all_murcko_bond
    all_murcko_atom = []
    for murcko_bond in all_murcko_bond:
        all_murcko_atom = list(set(all_murcko_atom + murcko_bond))

    if len(all_murcko_atom) > 0:
        # return all break atom (the break atoms did'n appear in the same substructure)
        all_break_atom = dict()
        for murcko_atom in all_murcko_atom:
            murcko_break_atom = []
            for murcko_bond in all_murcko_bond:
                if murcko_atom in murcko_bond:
                    murcko_break_atom += list(set(murcko_bond))
            murcko_break_atom = [x for x in murcko_break_atom if x != murcko_atom]
            all_break_atom[murcko_atom] = murcko_break_atom

        substrate_idx = dict()
        used_atom = []
        for initial_atom_idx, break_atoms_idx in all_break_atom.items():
            if initial_atom_idx not in used_atom:
                neighbor_idx = [initial_atom_idx]
                substrate_idx_i = neighbor_idx
                begin_atom_idx_list = [initial_atom_idx]
                while len(neighbor_idx) != 0:
                    for idx in begin_atom_idx_list:
                        initial_atom = mol.GetAtomWithIdx(idx)
                        neighbor_idx = neighbor_idx + [neighbor_atom.GetIdx() for neighbor_atom in
                                                       initial_atom.GetNeighbors()]
                        exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i
                        if idx in all_break_atom.keys():
                            exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i + all_break_atom[idx]
                        neighbor_idx = [x for x in neighbor_idx if x not in exlude_idx]
                        substrate_idx_i += neighbor_idx
                        begin_atom_idx_list += neighbor_idx
                    begin_atom_idx_list = [x for x in begin_atom_idx_list if x not in substrate_idx_i]
                substrate_idx[initial_atom_idx] = substrate_idx_i
                used_atom += substrate_idx_i
            else:
                pass
    else:
        substrate_idx = dict()
        substrate_idx[0] = [x for x in range(mol.GetNumAtoms())]

    subgroups = [group for group in substrate_idx.values()]    
    
    masks = []
    for subgroup in subgroups:
        mask = np.ones((num_atoms), dtype=np.float32)
        mask[subgroup] = 0.0
        masks.append(mask)

    return subgroups, masks

def generate_pair_combinations(smiles, n_combinations=100):
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()

    (fg, _), (murcko, _), (brics, _) = sme_fg_masks(smiles), sme_murcko_masks(smiles), sme_murcko_masks(smiles)

    subgroups = fg + murcko + brics
    subgroup_connections = identify_connections_between_subgroups(mol, subgroups)

    pairs = list(itertools.combinations(range(len(subgroups)), 2))

    subgroup_pairs = []
    for pair in pairs:
        if list(pair) in subgroup_connections:
            subgroup_pairs.append(subgroups[pair[0]] + subgroups[pair[1]])

    masks = []
    for subgroup in subgroup_pairs:
        mask = np.ones((num_atoms), dtype=np.float32)
        mask[subgroup] = 0.0
        masks.append(mask)

    return subgroup_pairs, masks