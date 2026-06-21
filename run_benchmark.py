import time
from boltz.data.parse.schema import compute_chiral_atom_constraints, compute_stereo_bond_constraints
from rdkit import Chem
from rdkit.Chem import AllChem

def benchmark_list_comp():
    mol = Chem.MolFromSmiles("CC(C)C1CCC(C)CC1")
    AllChem.EmbedMolecule(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    idx_map = {atom.GetIdx(): atom.GetIdx() for atom in mol.GetAtoms()}

    start = time.time()
    for _ in range(10000):
        # We replace the function logic locally to avoid imports and dependencies
        if all([atom.HasProp("_CIPRank") for atom in mol.GetAtoms()]):
            pass
    end = time.time()
    return end - start

def benchmark_generator():
    mol = Chem.MolFromSmiles("CC(C)C1CCC(C)CC1")
    AllChem.EmbedMolecule(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    idx_map = {atom.GetIdx(): atom.GetIdx() for atom in mol.GetAtoms()}

    start = time.time()
    for _ in range(10000):
        # We replace the function logic locally to avoid imports and dependencies
        if all(atom.HasProp("_CIPRank") for atom in mol.GetAtoms()):
            pass
    end = time.time()
    return end - start

print(f"List comp: {benchmark_list_comp():.4f}")
print(f"Generator: {benchmark_generator():.4f}")
