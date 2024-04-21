from rdkit import Chem
import torch


def smiles_to_onehot(smiles, max_atoms=300, add_hs=False):
    """
    turn SMILES into 1-hot encoded tensor
    """
    atom_index = {
        "H": 0,
        "C": 1,
        "O": 2,
        "N": 3,
        "P": 4,
        "S": 5,
        "Cl": 6,
        "F": 7,
    }
    molecule = Chem.MolFromSmiles(smiles)
    if add_hs:
        molecule = Chem.AddHs(molecule)
    out_tensor = torch.zeros(max_atoms, len(atom_index))

    for index, atom in enumerate(molecule.GetAtoms()):
        for symbol in atom_index:
            if atom.GetSymbol() == symbol:
                out_tensor[index, atom_index[symbol]] = 1
    return out_tensor

    
def smiles_to_edge(smiles, add_hs=False):
    """
    turn SMILES into edge intex tensor
    """
    edge_inex = [[],[]]
    molecule = Chem.MolFromSmiles(smiles)
    Chem.rdmolops.Kekulize(molecule)
    if add_hs:
        molecule = Chem.AddHs(molecule)
    for bond in molecule.GetBonds():
        first_atom_index = bond.GetBeginAtom().GetIdx()
        second_atom_index = bond.GetEndAtom().GetIdx()
        edge_inex[0].append(first_atom_index)
        edge_inex[1].append(second_atom_index)
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            edge_inex[0].append(first_atom_index)
            edge_inex[1].append(second_atom_index)
        elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
            for _ in range(2):
                edge_inex[0].append(first_atom_index)
                edge_inex[1].append(second_atom_index)
    return torch.tensor(edge_inex, dtype=torch.int64)

