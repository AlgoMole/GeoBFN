import os
import os.path as osp
import sys
from typing import Any, Callable, List, Optional
import warnings
import re

import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import one_hot, scatter
from torch_geometric.data.makedirs import makedirs

atomic_nb = [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83]
atomic_nb_no_h = [5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83]

"""
    How does PyG determine whether the dataset has been created and processed with the same arguments?
    ANS: PyG would open two files in processed_dir to store the latest arguments of pre_transform and pre_filter.
        If any of the arguments are different from the stored values, the library would generate a warning and prompt the user 
        to first clear the entire preocessed__dir before creating a dataset with the new filter/transormation
        For more details, see the source code for torch_geometric.data.dataset._process() 
            @ https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset
"""
""" TODO:
    1. DONE! PyG would notify the user to delete {self.processed_dir} and reprocess from the raw data
        , only if the user call Geom() with a different pre_filter or pre_transform. 
        As such, under our current implementation, calling GeomGen() with a different 
        remove_h, max_kept_conformer, or max_mol_len will not incur any warining
        , and you will still receive the last pre-processed version.

        Solution: mimic the implementation of _process @https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset
        a. If the pre-processing parameters(e.g. remove_h) haven't been saved, save them as a file in {self.processed_dir}
        b. If they've been saved, compare the pre-processing parameters of the current call with those in the saved file.
        c. generate a warning if they are different.
    2. Shall we drop "conformerweights" which is a vector of varying size
    3. Determine the dtype of each data attribute. 
        Now, I set everything to torch.float32
    4. Consider how to simultaneously store h and no_h versions of the dataset
"""


class Geom(InMemoryDataset):
    r"""The Geom_Drug dataset from <> .
        This pyG implementation references the preprocessing implementation from the "Equivariant Diffusion for Molecule Generation in 3D" <https://arxiv.org/abs/2203.17003>

    Data:
        mol_id: The ID of the molecule to which the conformation belongs. The ID corresponds to order of molecule SMILES in the geom_drugs_smiles.txt locating in the processed directroy
        x: The atomtypes in one_hot and atomic number. Same as QM9(InMemoryDataset)
        pos: The xyz coordinates of the atom. Same as QM9(InMemoryDataset)
        geom_id: a unique number identifying every geometry
        set: the index of the MD set that this geometry came from during the CRET run
        degeneracy: how many degenerate rotamers there are for this conformer
        y: The values of [totalenergy, relativeenergy, relativeenergy] stacked into a vector.
        # NOTE: conformerweights is not included in the current version
    Args:
        root (str): Root directory where the dataset should be saved.
        max_kept_conformations: The maxumum number of conformations to keep per molecule.
            If the limit is exceeded, those conformations with smaller total energy are kept.
        remove_h (bool): remove hydrogen atoms from all conformations
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    **STATS:**


    """  # noqa: E501

    raw_url = "https://dataverse.harvard.edu/api/access/datafile/4360331"

    def __init__(
        self,
        root: str,
        max_kept_conformers: int,
        remove_h: bool,
        max_mol_len: int,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.max_kept_conformers = max_kept_conformers
        self.remove_h = remove_h
        self.max_mol_len = max_mol_len
        super().__init__(
            root, transform, pre_transform, pre_filter
        )  # download() and process() are called here by the parent constructor
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.index2charge = torch.tensor(
            [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83]
        )
        """
        InMemoryDataset.collate() collates the long list of data() into 
            a) the huge data object and
            b) slices that are used to recover a single instance from this object
        """

    @property
    def raw_file_names(self) -> List[str]:
        return ["drugs_crude.msgpack"]

    @property
    def processed_file_names(self) -> str:
        conformations = self.max_kept_conformers
        if not self.remove_h:
            return [
                "train.pt".format(conformations),
                "geom_drugs_withh_smiles.txt",
                "geom_drugs_withh_n_.npy",
            ]  # process() processes the raw file and save the processed data dictionary into this .pt file

    # def download(self):
    #     """
    #     Download and Unzip rawdata @ raw_url to root/raw
    #     """
    #     file_path = download_url(self.raw_url, self.raw_dir) # download | self.raw_dir() <-> osp.join(root, 'raw')
    #     extract_zip(file_path, self.raw_dir) # unzip to root/raw
    #     os.unlink(file_path) # remove the zip file @ root/raw/xxx.zip

    def process(self):
        import msgpack

        file_path = self.raw_paths[0]
        unpacker = msgpack.Unpacker(open(file_path, "rb"))

        data_list = []
        all_smiles = []
        all_number_atoms = []
        mol_id = 0
        atomic_nb_list = atomic_nb_no_h if self.remove_h else atomic_nb
        atomic_nb_list = torch.tensor(atomic_nb_list)
        for i, drugs_1k in enumerate(unpacker):
            print(f"Unpacking file {i}...")

            # cnt = 0 #DEBUG

            for smiles, all_info in drugs_1k.items():
                all_smiles.append(smiles)
                conformers = all_info["conformers"]
                # Get the energy of each conformer. Keep only the lowest values
                all_energies = []
                for conformer in conformers:
                    all_energies.append(conformer["totalenergy"])
                all_energies = np.array(all_energies)
                argsort = np.argsort(all_energies)
                lowest_energies = argsort[: self.max_kept_conformers]
                for id in lowest_energies:
                    conformer = conformers[id]
                    x_pos = np.array(conformer["xyz"]).astype(
                        float
                    )  # num_atoms x 4(which are atomtype,x,y,z)
                    # Remove Hydrogen
                    mask = x_pos[:, 0] != self.remove_h
                    x_pos = x_pos[mask]
                    # Prefilter based on molecule length/size
                    if len(x_pos) > self.max_mol_len:
                        break  # we can break to skip this molecule when we find one of its conformer is too big
                    # Record atom number of each conformer
                    num_atoms = x_pos.shape[0]
                    all_number_atoms.append(num_atoms)
                    # Get x
                    x = torch.from_numpy(x_pos[:, 0].astype(np.float32)).unsqueeze(-1)
                    one_hot = x == (atomic_nb_list.unsqueeze(0))
                    charge = x
                    # x = torch.cat((one_hot, x), dim=1) # num_atoms x (num_atom_type+1)
                    # Get pos
                    pos = torch.from_numpy(x_pos[:, -3:].astype(np.float32))
                    # Get geom_id, set, and degeneracy
                    geom_id = torch.tensor(conformer["geom_id"], dtype=torch.float32)
                    set = torch.tensor(conformer["set"])
                    degeneracy = torch.tensor(
                        conformer["degeneracy"], dtype=torch.float32
                    )
                    # Stack remaining conformer features into a vector
                    y = [
                        conformer[k]
                        for k in conformer.keys()
                        if k not in ["xyz", "geom_id", "set", "conformerweights"]
                    ]
                    y = torch.tensor(y, dtype=torch.float32)
                    data = Data(
                        mol_id=mol_id,
                        x=one_hot,
                        pos=pos,
                        geom_id=geom_id,
                        set=set,
                        degeneracy=degeneracy,
                        y=y,
                        charge=charge,
                    )
                    # Apply pre_transform(0) and pre_filter()
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    # Append data to data_list
                    data_list.append(data)
                mol_id += 1

            #     #DEBUG
            #     cnt += 1
            #     if(cnt == 2):
            #         break

            # break #DEBUG
        # Check whether the configuration results in a empty datalist
        assert (
            len(data_list) > 0
        ), "The current configuration(max_kept_conformers, remove_h, max_mol_len) results in every conformation get dropped/filter \n Please raise 'max_kept_conformers' or 'max_mol_len'"
        print("Dataset processed, now saving...")
        # Save the processed Geom confromers
        torch.save(self.collate(data_list), self.processed_paths[0])
        # Save SMILES
        with open(self.processed_paths[1], "w") as f:
            for s in all_smiles:
                f.write(s)
                f.write("\n")
        # Save number of atoms per conformation
        np.save(self.processed_paths[2], all_number_atoms)
        print("Dataset saved.")

    def _process(
        self,
    ):  # Overriding _process() of its grandparent class torch_geometric.data.Dataset
        f = osp.join(self.processed_dir, "config.pt")
        if osp.exists(f):
            last_config = torch.load(f)
            config_change = not all(
                [
                    self.max_kept_conformers == last_config["max_kept_conformers"],
                    self.remove_h == last_config["remove_h"],
                    self.max_mol_len == last_config["max_mol_len"],
                ]
            )
            if config_change:
                warnings.warn(
                    f"The configuration(max_kept_conformers, remove_h, max_mol_len) differs from the one used in "
                    f"the pre-processed version of this dataset. If you want to "
                    f"apply another configuration, make sure to "
                    f"delete '{self.processed_dir}' first so that"
                    f"the dataset get reprocessed from the raw form"
                )
        f = osp.join(self.processed_dir, "pre_transform.pt")
        if osp.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first"
            )

        f = osp.join(self.processed_dir, "pre_filter.pt")
        if osp.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-fitering technique, make sure to "
                "delete '{self.processed_dir}' first"
            )

        if files_exist(self.processed_paths):  # pragma: no cover
            return

        if self.log and "pytest" not in sys.modules:
            print("Processing...", file=sys.stderr)

        makedirs(self.processed_dir)
        self.process()

        path = osp.join(self.processed_dir, "config.pt")
        config = dict(
            max_kept_conformers=self.max_kept_conformers,
            remove_h=self.remove_h,
            max_mol_len=self.max_mol_len,
        )
        torch.save(config, path)
        path = osp.join(self.processed_dir, "pre_transform.pt")
        torch.save(_repr(self.pre_transform), path)
        path = osp.join(self.processed_dir, "pre_filter.pt")
        torch.save(_repr(self.pre_filter), path)

        if self.log and "pytest" not in sys.modules:
            print("Done!", file=sys.stderr)


def _repr(obj: Any) -> str:
    if obj is None:
        return "None"
    return re.sub("(<.*?)\\s.*(>)", r"\1\2", str(obj))


def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])
