from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch_geometric.data import Data
import numpy as np
import torch
import os
import tqdm
import pickle as pkl
from core.evaluation.utils import convert_atomcloud_to_mol_smiles, save_molist, dump2mol
from core.evaluation.metrics import BasicMolGenMetric
from core.evaluation.visualization import visualize, visualize_chain
import json
import matplotlib
import wandb
import copy

# this file contains the model which we used to visualize the

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def compute_or_retrieve_dataset_smiles(
    dataset, atom_decoder, save_path, single_bond=False
):
    # create parent directory if it does not exist
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if not os.path.exists(save_path):
        all_smiles = []
        with tqdm.tqdm(total=len(dataset)) as pbar:
            print("Computing all smiles")
            for data in dataset:
                mol, smiles = convert_atomcloud_to_mol_smiles(
                    data.pos,
                    data.x,
                    atom_decoder,
                    type_one_hot=True,
                    single_bond=single_bond,
                )
                if smiles is not None:
                    all_smiles.append(smiles)
                pbar.update(1)
        print(f"Saving {len(all_smiles)} smiles to {save_path}")
        with open(save_path, "wb") as f:
            pkl.dump(all_smiles, f)
    else:
        print("Loading all smiles")
        with open(save_path, "rb") as f:
            all_smiles = pkl.load(f)
    if isinstance(all_smiles[0], tuple):  # legacy wrong format
        smiles_set = set([s[1] for s in all_smiles])
    else:
        smiles_set = set([s for s in all_smiles])
    return smiles_set


class MolGenValidationCallback(Callback):
    def __init__(self, dataset, atom_type_one_hot=True, single_bond=False) -> None:
        super().__init__()
        self.dataset = dataset
        self.single_bond = single_bond
        self.type_one_hot = atom_type_one_hot
        self.outputs = []
        self.test_outputs = []

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        dataset_all_smiles = compute_or_retrieve_dataset_smiles(
            dataset=self.dataset,
            atom_decoder=pl_module.cfg.dataset.atom_decoder,
            save_path=os.path.join(
                pl_module.cfg.dataset.datadir, "processed", "all_smiles.pkl"
            ),
            single_bond=self.single_bond,
        )
        self.all_smiles = set(dataset_all_smiles)
        self.metric = BasicMolGenMetric(
            atom_decoder=pl_module.cfg.dataset.atom_decoder,
            dataset_smiles_set=self.all_smiles,
            type_one_hot=self.type_one_hot,
            single_bond=self.single_bond,
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        self.outputs.extend(outputs)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_start(trainer, pl_module)
        self.outputs = []

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        out_metrics = self.metric.evaluate(self.outputs)
        pl_module.log_dict(out_metrics)
        print(json.dumps(out_metrics, indent=4))


class MolVisualizationCallback(Callback):
    # here the call back, we save the molecules and also draw the figures also to the wandb.
    def __init__(self, atomic_nb, remove_h, atom_decoder, generated_mol_dir) -> None:
        super().__init__()
        self.outputs = []
        self.test_outputs = {"in_data": [], "out_data": []}
        self.chain_outputs = []
        self.atomic_nb = atomic_nb
        self.remove_h = remove_h
        self.atom_type_num = len(atomic_nb) - remove_h
        self.generated_mol_dir = generated_mol_dir
        self.atom_decoder = atom_decoder

    # TODO delete this function
    def charge_decode(self, charge):
        """
        charge: [n_nodes, 1]
        """
        anchor = torch.tensor(
            [
                (2 * k - 1) / max(self.atomic_nb) - 1
                for k in self.atomic_nb[self.remove_h :]
            ],
            dtype=torch.float32,
            device=charge.device,
        )
        atom_type = (charge - anchor).abs().argmin(dim=-1)
        one_hot = torch.zeros(
            [charge.shape[0], self.atom_type_num], dtype=torch.float32
        )
        one_hot[torch.arange(charge.shape[0]), atom_type] = 1
        return one_hot

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

        self.outputs.extend(outputs)
        if len(self.chain_outputs) == 0:
            _, _, _, edge_index, segment_ids = (
                batch.zx,  # [n_nodes, n_features]
                batch.zpos,  # [n_nodes, 3]
                batch.zcharges,
                batch.edge_index,  # [2, edge_num]
                batch.batch,  # [n_nodes]
            )
            # z_h = (
            #     torch.concat([z_h, z_charges], dim=-1)
            #     if self.cfg.dynamics.include_charges
            #     else z_h
            # )
            # tseqs = torch.linspace(0, 1, 250, dtype=torch.float32, device=z_x.device
            n_nodes = segment_ids.shape[0]
            # TODO don't call model in evaluation callbacks, make sure this stays in the evaluation_step
            theta_chain = pl_module.dynamics(
                n_nodes=n_nodes,
                edge_index=edge_index,
                sample_steps=pl_module.dynamics.sample_steps,
                segment_ids=segment_ids,
            )
            for i in range(len(theta_chain)):
                x, h = theta_chain[i]
                atom_type = self.charge_decode(h[:, :1])
                out_batch = copy.deepcopy(batch)

                out_batch.x, out_batch.pos = (atom_type, x)
                _slice_dict = {
                    "x": out_batch._slice_dict["zx"],
                    "pos": out_batch._slice_dict["zpos"],
                }
                _inc_dict = {
                    "x": out_batch._inc_dict["zx"],
                    "pos": out_batch._inc_dict["zpos"],
                }
                out_batch._inc_dict.update(_inc_dict)
                out_batch._slice_dict.update(_slice_dict)
                out_data_list = out_batch.to_data_list()
                self.chain_outputs.append(
                    out_data_list[0]
                )  # always append the first sampled dtat

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_start(trainer, pl_module)
        self.outputs = []
        self.chain_outputs = []

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        epoch = pl_module.current_epoch

        path = os.path.join(pl_module.cfg.accounting.generated_mol_dir, str(epoch))

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        chain_path = os.path.join(
            pl_module.cfg.accounting.generated_mol_dir, str(epoch), "chain"
        )

        if not os.path.exists(chain_path):
            os.makedirs(chain_path, exist_ok=True)

        if pl_module.cfg.visual.save_mols:
            # we save the figures here.
            save_molist(
                path=path,
                molecule_list=self.outputs,
                index2atom=pl_module.cfg.dataset.atom_decoder,
            )
            if pl_module.cfg.visual.visual_nums > 0:
                images = visualize(
                    path=path,
                    atom_decoder=pl_module.cfg.dataset.atom_decoder,
                    color_dic=pl_module.cfg.dataset.colors_dic,
                    radius_dic=pl_module.cfg.dataset.radius_dic,
                    max_num=pl_module.cfg.visual.visual_nums,
                )
                # table = [[],[]]
                table = []
                for p_ in images:
                    im = plt.imread(p_)
                    table.append(wandb.Image(im))
                    # if len(table[0]) < 5:
                    #     table[0].append(wandb.Image(im))
                    # else:
                    #     table[1].append(wandb.Image(im))
                # pl_module.logger.log_table(key="epoch {}".format(epoch),data=table,columns= ['1','2','3','4','5'])
                pl_module.logger.log_image(key="epoch {}".format(epoch), images=table)
                # wandb.log()
                # update to wandb
        if pl_module.cfg.visual.visual_chain:
            # we save the chains and visual the gif here.
            # print(len(self.chain_outputs),chain_path)
            save_molist(
                path=chain_path,
                molecule_list=self.chain_outputs,
                index2atom=pl_module.cfg.dataset.atom_decoder,
            )
            # if pl_module.cfg.visual.visual_nums > 0:
            gif_path = visualize_chain(
                path=chain_path,
                atom_decoder=pl_module.cfg.dataset.atom_decoder,
                color_dic=pl_module.cfg.dataset.colors_dic,
                radius_dic=pl_module.cfg.dataset.radius_dic,
                spheres_3d=False,
            )
            gifs = wandb.Video(gif_path)
            columns = ["Generation Path"]
            pl_module.logger.log_table(
                key="epoch_{}".format(epoch), data=[[gifs]], columns=columns
            )

            # table = [[],[]]

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for key in outputs:
            self.test_outputs[key].extend(outputs[key])

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.test_outputs = {"in_data": [], "out_data": []}
        # create dir if not exist for generated molecules
        if self.generated_mol_dir is not None:
            os.makedirs(self.generated_mol_dir, exist_ok=True)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for key in self.test_outputs:
            for idx, data_list in enumerate(self.test_outputs[key]):
                for step, data in enumerate(data_list):
                    dump2mol(
                        data,
                        os.path.join(
                            self.generated_mol_dir,
                            f"{key}-molid_{idx:03}-step_{step:03}.mol",
                        ),
                        index2atom=self.atom_decoder,
                        get_bond=True,
                    )
