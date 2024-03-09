# We implement the evaluation metric in this file.
from rdkit import Chem
from torch_geometric.data import Data
from core.evaluation.utils import (
    convert_atomcloud_to_mol_smiles,
    build_molecule,
    mol2smiles,
    build_xae_molecule,
    check_stability,
)
from typing import List, Dict, Tuple


class BasicMolGenMetric(object):
    def __init__(
        self, atom_decoder, dataset_smiles_set, type_one_hot=True, single_bond=False
    ):
        self.atom_decoder = atom_decoder
        self.dataset_smiles_set = dataset_smiles_set
        self.type_one_hot = type_one_hot
        self.single_bond = single_bond

    def compute_stability(self, generated2idx: List[Tuple[Data, int]]):
        n_samples = len(generated2idx)
        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        return_list = []
        for data, idx in generated2idx:
            positions = data.pos
            atom_type = data.x
            stability_results = check_stability(
                positions=positions,
                atom_type=atom_type,
                atom_decoder=self.atom_decoder,
                single_bond=self.single_bond,
            )

            molecule_stable += int(stability_results[0])
            nr_stable_bonds += int(stability_results[1])
            n_atoms += int(stability_results[2])
            if int(stability_results[0]) != 0:
                return_list.append((data, idx))

        # stability
        fraction_mol_stable = molecule_stable / float(n_samples)
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)
        stability_dict = {
            "mol_stable": fraction_mol_stable,
            "atm_stable": fraction_atm_stable,
        }
        return stability_dict, return_list

    def compute_validity(self, generated2idx: List[Tuple[Data, int]]):
        """generated: list of couples (positions, atom_types)"""
        valid = []
        return_list = []
        for graph, idx in generated2idx:
            mol, smiles = convert_atomcloud_to_mol_smiles(
                positions=graph.pos,
                atom_type=graph.x,
                atom_decoder=self.atom_decoder,
                type_one_hot=self.type_one_hot,
                single_bond=self.single_bond,
            )
            if smiles is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                if smiles is not None:
                    valid.append(smiles)
                    return_list.append((smiles, idx))

        return valid, len(valid) / (len(generated2idx) + 1e-12), return_list

    def compute_uniqueness(self, valid):
        """valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / (len(valid) + 1e-12)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_set:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / (len(unique) + 1e-12)

    def evaluate(self, generated: List[Data]):
        """generated: list of pairs (positions: n x 3, atom_types: n [int])
        the positions and atom types should already be masked."""
        generated2idx = [(graph, i) for i, graph in enumerate(generated)]
        stability_dict, return_generated2idx_list = self.compute_stability(
            generated2idx
        )
        valid, validity, _ = self.compute_validity(generated2idx)
        _, _, return_generated2idx_list = self.compute_validity(
            return_generated2idx_list
        )
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(
                f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%"
            )

            _, novelty = self.compute_novelty(unique)
            print(
                f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%"
            )
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None
        if len(return_generated2idx_list) > 0:
            _, stable_valid_uniqueness = self.compute_uniqueness(
                [g for g, i in return_generated2idx_list]
            )
            stable_valid_uniqueness = (
                stable_valid_uniqueness
                * len(return_generated2idx_list)
                / len(generated)
            )
        else:
            stable_valid_uniqueness = 0.0

        return {
            "validity": validity,
            "uniqueness": uniqueness,
            "stable_valid_uniqueness": stable_valid_uniqueness,
            "novelty": novelty,
            **stability_dict,
        }
