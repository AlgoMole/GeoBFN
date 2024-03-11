from typing import Any, Optional
import pytorch_lightning as pl
import argparse
import copy
from pytorch_lightning.utilities.types import STEP_OUTPUT

from torch.optim.optimizer import Optimizer
from core.config.config import Config
from core.model.bfn.bfn_base import bfn4MolEGNN
from core.data.qm9_gen import QM9Gen
import torch
import os
import datetime, pytz
from core.losses import loss
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import wandb
from core.callbacks.basic import (
    Gradient_clip,
    DebugCallback,
    NormalizerCallback,
    RecoverCallback,
    EMACallback,
)
from core.evaluation.validation_callback import (
    MolGenValidationCallback,
    MolVisualizationCallback,
)
from absl import logging
from core.data.prefetch import PrefetchLoader
import core.utils.ctxmgr as ctxmgr


class BFN4MolGenTrain(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.dynamics = bfn4MolEGNN(
            self.cfg.dynamics.in_node_nf,
            self.cfg.dynamics.hidden_nf,
            n_layers=self.cfg.dynamics.n_layers,
            sigma1_coord=self.cfg.dynamics.sigma1_coord,
            sigma1_charges=self.cfg.dynamics.sigma1_charges,
            bins=self.cfg.dynamics.bins,
            beta1=self.cfg.dynamics.beta1,
            sample_steps=self.cfg.dynamics.sample_steps,
            no_diff_coord=self.cfg.dynamics.no_diff_coord,
            charge_discretised_loss=self.cfg.dynamics.charge_discretised_loss,
            charge_clamp=self.cfg.dynamics.charge_clamp,
            t_min=self.cfg.dynamics.t_min,
        )
        # [ time, h_t, pos_t, edge_index]
        self.train_losses = []
        self.save_hyperparameters(logger=False)
        # self.logger.log_hyperparams(self.cfg.todict())
        self.atomic_nb = self.cfg.dataset.atomic_nb
        self.remove_h = self.cfg.dataset.remove_h
        self.atom_type_num = len(self.atomic_nb) - self.remove_h

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        h, charges, x, edge_index, segment_ids = (
            batch.x,  # [n_nodes, n_features]
            batch.charges,  # [n_nodes, 1]
            batch.pos,  # [n_nodes, 3]
            batch.edge_index,  # [2, edge_num]
            batch.batch,  # [n_nodes]
        )
        num_molecules = batch.idx.shape[0]
        # print("train_step",batch.charges)
        h = charges
        # print("h",h,h[:,-1:])

        if self.cfg.optimization.difftime:
            t = torch.rand(
                [num_molecules, 1], dtype=x.dtype, device=x.device
            ).index_select(0, segment_ids)
        else:
            t = torch.rand([1, 1], dtype=x.dtype, device=x.device) * torch.ones(
                size=[segment_ids.shape[0], 1], dtype=x.dtype, device=x.device
            )  # [n_nodes, 1]
        posloss, charge_loss, _ = self.dynamics.loss_one_step(
            t, x=h, pos=x, edge_index=edge_index, segment_ids=segment_ids
        )
        # valid_loss = losses < 1000
        # if not valid_loss.all():
        #     logging.warning(f"valid loss {torch.sum(valid_loss)} < {losses.shape[0]}")
        #     losses = losses * valid_loss.to(losses.dtype)

        loss = torch.mean(posloss + charge_loss)

        self.log(
            "loss",
            loss,
            on_step=True,
            prog_bar=True,
            batch_size=self.cfg.optimization.batch_size,
        )

        self.train_losses.append(loss.clone().detach().cpu())
        return loss

    def test_step(self, batch, batch_idx):
        sample_steps = self.cfg.dynamics.sample_steps
        n_nodes = batch.pos.shape[0]
        h, charges, x, edge_index, segment_ids = (
            batch.x,  # [n_nodes, n_features]
            batch.charges,  # [n_nodes, 1]
            batch.pos,  # [n_nodes, 3]
            batch.edge_index,  # [2, edge_num]
            batch.batch,  # [n_nodes]
        )
        num_molecules = batch.idx.shape[0]
        # print("train_step",batch.charges)
        h = charges
        # print("h",h,h[:,-1:])
        in_data = []
        out_data = []
        for i in range(1, sample_steps + 1):
            t = torch.ones((n_nodes, 1)).to(self.device) * (i - 1) / sample_steps
            t = torch.clamp(t, min=self.cfg.dynamics.t_min)
            posloss, charge_loss, assets = self.dynamics.loss_one_step(
                t, x=h, pos=x, edge_index=edge_index, segment_ids=segment_ids
            )
            (mu_coord, mu_charge, coord_pred, k_hat, gamma_coord, gamma_charge) = assets
            in_atom_type = self.charge_decode(mu_charge)
            out_atom_type = self.charge_decode(k_hat)

            in_batch = copy.deepcopy(batch)
            out_batch = copy.deepcopy(batch)

            in_batch.x, in_batch.pos = in_atom_type, mu_coord
            out_batch.x, out_batch.pos = out_atom_type, coord_pred

            in_data_list = in_batch.to_data_list()
            out_data_list = out_batch.to_data_list()
            in_data.append(in_data_list)
            out_data.append(out_data_list)
        in_data = list(zip(*in_data))
        out_data = list(zip(*out_data))
        assert len(in_data) == len(out_data) == num_molecules
        return {"in_data": in_data, "out_data": out_data}

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

    def validation_step(self, batch, batch_idx):
        edge_index, segment_ids = (
            batch.edge_index,  # [2, edge_num]
            batch.batch,  # [n_nodes]
        )

        n_nodes = segment_ids.shape[0]
        theta_chain = self.dynamics(
            n_nodes=n_nodes,
            edge_index=edge_index,
            segment_ids=segment_ids,
        )

        x, h = theta_chain[-1]
        atom_type = self.charge_decode(h[:, :1])
        out_batch = copy.deepcopy(batch)

        out_batch.x, out_batch.pos = atom_type, x
        _slice_dict = {
            "x": out_batch._slice_dict["zx"],
            "pos": out_batch._slice_dict["zpos"],
        }
        _inc_dict = {"x": out_batch._inc_dict["zx"], "pos": out_batch._inc_dict["zpos"]}
        out_batch._inc_dict.update(_inc_dict)
        out_batch._slice_dict.update(_slice_dict)
        out_data_list = out_batch.to_data_list()

        return out_data_list

    def on_train_epoch_end(self) -> None:
        if len(self.train_losses) == 0:
            epoch_loss = 0
        else:
            epoch_loss = torch.stack([x for x in self.train_losses]).mean()
        print(f"epoch_loss: {epoch_loss}")
        self.log(
            "epoch_loss",
            epoch_loss,
            batch_size=self.cfg.optimization.batch_size,
        )
        self.train_losses = []

    def configure_optimizers(self):
        # optim = torch.optim.SGD(self.parameters(), lr=self.cfg.optimization.lr)
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optimization.lr,
            amsgrad=True,
            weight_decay=float(self.cfg.optimization.weight_decay),
        )
        return optim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="debug.yaml",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--logging_level", type=str, default="warning")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sigma1_coord", type=float, default=0.001)
    parser.add_argument("--sigma1_charges", type=float, default=0.15)
    parser.add_argument("--beta1", type=float, default=2.0)
    parser.add_argument("--sample_steps", type=int, default=1000)
    parser.add_argument("--eval_data_num", type=int, default=1000)
    parser.add_argument("--checkpoint_freq", type=int, default=20)
    parser.add_argument("--exp_version", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ckpt_pattern", type=str, default="last*.ckpt")

    _args = parser.parse_args()
    # _args, unknown = parser.parse_known_args()
    cfg = Config(**_args.__dict__)
    print(f"The config of this process is:\n{cfg}")
    logging_level = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "fatal": logging.FATAL,
    }
    logging.set_verbosity(logging_level[cfg.logging_level])
    # create dir if not exist
    os.makedirs(cfg.accounting.wandb_logdir, exist_ok=True)
    wandb_logger = WandbLogger(
        name=cfg.exp_name
        + f'_{datetime.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d-%H:%M:%S")}',
        project=cfg.project_name,
        offline=cfg.debug or cfg.no_wandb,
        save_dir=cfg.accounting.wandb_logdir,
        version=cfg.accounting.exp_version,
    )  # add wandb parameters
    wandb_logger.log_hyperparams(cfg.todict())
    cfg.save2yaml(cfg.accounting.dump_config_path)
    if cfg.dataset.name == "qm9":
        train_loader = QM9Gen(
            datadir=cfg.dataset.datadir,
            batch_size=cfg.optimization.batch_size,
            n_node_histogram=cfg.dataset.n_node_histogram,
            debug=cfg.debug,
            num_workers=cfg.dataset.num_workers,
            split="train" if not cfg.test else "test",
        )
        eval_loader = QM9Gen.initiate_evaluation_dataloader(
            data_num=cfg.evaluation.eval_data_num if not cfg.debug else 50,
            n_node_histogram=cfg.dataset.n_node_histogram,
            batch_size=cfg.evaluation.batch_size,
        )
    else:
        raise NotImplementedError

    model = BFN4MolGenTrain(config=cfg)
    # print(model)

    trainer = pl.Trainer(
        limit_test_batches=1,
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.optimization.epochs,
        check_val_every_n_epoch=cfg.accounting.checkpoint_freq,
        devices=1,
        # overfit_batches=10,
        logger=wandb_logger,
        num_sanity_val_steps=2,
        # overfit_batches=10,
        # gradient_clip_val=1.0,
        # devices=1,
        callbacks=[
            RecoverCallback(
                latest_ckpt=cfg.accounting.checkpoint_path,
                resume=cfg.optimization.resume or cfg.test,
                recover_trigger_loss=cfg.optimization.recover_trigger_loss,
                skip_count_limit=cfg.optimization.skip_count_limit,
            ),
            Gradient_clip(
                maximum_allowed_norm=cfg.optimization.maximum_allowed_norm,
            ),  # time consuming
            NormalizerCallback(normalizer_dict=cfg.dataset.normalizer_dict),
            MolGenValidationCallback(
                dataset=train_loader.ds,
                atom_type_one_hot=True,
                single_bond=cfg.evaluation.single_bond,
            ),
            ModelCheckpoint(
                dirpath=cfg.accounting.checkpoint_dir,
                filename="{epoch}-{mol_stable:2f}-{atm_stable:2f}-{validity:2f}",
                every_n_epochs=cfg.accounting.checkpoint_freq,
                save_last=True,
                save_top_k=20,
                mode="max",
                monitor="atm_stable",
            ),
            MolVisualizationCallback(
                atomic_nb=cfg.dataset.atomic_nb,
                remove_h=cfg.dataset.remove_h,
                atom_decoder=cfg.dataset.atom_decoder,
                generated_mol_dir=cfg.accounting.generated_mol_dir,
            ),
            EMACallback(decay=0.9999, ema_device="cuda"),
            # DebugCallback(),
        ],
    )
    # num_sanity_val_steps=2, overfit_batches=10, devices=1
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader)
    if not cfg.test:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader)
    else:
        trainer.validate(
            model,
            dataloaders=eval_loader,
        )
        # trainer.test(model, dataloaders=train_loader)
    wandb_logger.finalize("success")
    wandb_logger.experiment.finish()
    # trainer.test(model, datamodule=None)
