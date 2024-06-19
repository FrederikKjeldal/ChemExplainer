from rdkit import Chem
from rdkit.Chem import Draw


def save_smiles_svg(smiles, filename, atom_label=True):
    mol = Chem.MolFromSmiles(smiles)

    if atom_label:
        for i, atom in enumerate(mol.GetAtoms()):
            label = i
            atom.SetProp("atomNote", f"{label}")

    mol = Draw.rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(300, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    with open(filename, "w") as f:
        f.write(svg)

    return


def save_explanation_svg(smiles, atoms, filename, atom_label=False):
    mol = Chem.MolFromSmiles(smiles)

    if atom_label:
        for i, atom in enumerate(mol.GetAtoms()):
            label = i
            atom.SetProp("atomNote", f"{label}")

    mol = Draw.rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(300, 300)
    drawer.DrawMolecule(mol, highlightAtoms=atoms)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    with open(filename, "w") as f:
        f.write(svg)

    return
