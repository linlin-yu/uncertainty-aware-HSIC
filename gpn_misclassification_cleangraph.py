import os, time
import torch
from torch import optim
from torch import nn
from datetime import datetime
import pandas as pd
import numpy as np

from torch_geometric.data.lightning import LightningNodeData

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.callbacks import EarlyStopping

from utils.utils import load_data
from utils.yaml import read_yaml_file
from utils.config import RunConfiguration, DataConfiguration
from utils.config import ModelConfiguration, TrainingConfiguration
from utils.metrics import get_metrics
from utils.loss import ModelLoss, UCELoss, ProjectedCELoss
from utils.gpn_model import GPNNet

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES']='4'
torch.set_float32_matmul_precision("medium")


# define the LightningModule
class GPN_MIS(pl.LightningModule):
    def __init__(self, config, data, num_classes, device):
        super().__init__()
        self.save_hyperparameters(config["model"])
        self.save_hyperparameters(config["training"])
        # self.save_hyperparameters()
        self.run_cfg = RunConfiguration(**config["run"])
        self.data_cfg = DataConfiguration(**config["data"])
        self.model_cfg = ModelConfiguration(**config["model"])
        self.train_cfg = TrainingConfiguration(**config["training"])

        assert self.model_cfg.model_name == "GPN"

        self.num_classes = num_classes
        self.data = data.to(device)

        self.gnn = self._build_model(device)

        if self.model_cfg.pretrain_mode == "encoder":
            self.warmup_criterion = ProjectedCELoss(self.model_cfg.reduction)
        else:
            self.warmup_criterion = UCELoss(self.model_cfg.reduction)
        self.criterion = ModelLoss(
            self.model_cfg, self.data, mode="mis", dataset=config["data"]["dataset"]
        )

        self.x = self.data.x
        if hasattr(data, "endmemberS"):
            self.endmemberS = self.data.endmemberS

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.warmup_epochs = (
            0 if self.train_cfg.warmup_epochs is None else self.train_cfg.warmup_epochs
        )

    def _build_model(self, device):
        in_dim = self.data.num_features
        out_dim = self.num_classes

        assert self.model_cfg.model_name == "GPN"
        model = GPNNet(
            in_dim,
            self.model_cfg.hidden_dim,
            self.model_cfg.latent_dim,
            out_dim,
            self.model_cfg.radial_layers,
            self.model_cfg.drop_prob,
            iteration_step=self.model_cfg.iteration_step,
            teleport=self.model_cfg.teleport,
            pretrain_mode=self.model_cfg.pretrain_mode,
        ).to(device)
        print(model)
        return model

    def forward(self, data):
        return self.gnn(data)

    def training_step(self, batch, batch_idx):

        # print('print', self.current_epoch)
        data = batch
        # set-up optimizer
        model_optimizer, flow_optimizer = self.optimizers()
        # ------------------------------------------------------------------------------------------------
        # warmup training
        if self.current_epoch < self.warmup_epochs:
            print(f"Warm Up-------------epoch:{self.current_epoch}")
            # forward run
            alpha = self(data)

            if self.model_cfg.pretrain_mode == "encoder":
                warmup_optimizer = model_optimizer
            else:
                warmup_optimizer = flow_optimizer

            warmup_loss = self.warmup_criterion(
                alpha[data.train_mask], data.y[data.train_mask]
            )
            loss = warmup_loss
            warmup_optimizer.zero_grad()
            self.manual_backward(loss)
            warmup_optimizer.step()

        else:
            # ------------------------------------------------------------------------------------------------

            # ------------------------------------------------------------------------------------------------
            # main training loop training
            # set-up optimizer

            # print(f'Training-------------epoch:{batch_idx}')
            alpha = self(data)
            loss_dict = self.criterion(alpha, data.y, mask=data.train_mask)
            loss = sum(loss_dict.values())

            # log loss and training evaluation
            self.log(
                "loss/total_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            train_metrics = ["id_oa", "id_f1", "id_ce", "id_kappa", "id_aa"]
            train_results = get_metrics(
                train_metrics, data, alpha=alpha, mode="train", task="mis"
            )
            for k, v in loss_dict.items():
                self.log(f"loss/{k}", v)
            for k, v in train_results.items():
                self.log(f"train_evaluation/training_{k}", v)

            model_optimizer.zero_grad()
            self.manual_backward(loss)
            model_optimizer.step()

        return loss

    def validation_step(self, batch, batch_idx):
        # print('validation')
        data = batch
        alpha = self.gnn(data)
        results = {}
        val_metrics = [
            "id_oa",
            "id_f1",
            "id_ce",
            "id_kappa",
            "id_aa",
            "vacuity_roc",
            "dissonance_roc",
            "aleatoric_roc",
            "entropy_roc",
            "vacuity_pr",
            "dissonance_pr",
            "aleatoric_pr",
            "entropy_pr",
        ]
        val_results = get_metrics(
            val_metrics, data, alpha=alpha, mode="validation", task="mis"
        )
        for k, v in val_results.items():
            # self.log(f'val_evaluation/val_{k}', v)
            results[f"val_{k}"] = v
        results[f"val_overall"] = val_results["aleatoric_roc"] + val_results["id_f1"]
        test_metrics = [
            "id_oa",
            "id_f1",
            "id_ce",
            "id_kappa",
            "id_aa",
            "vacuity_roc",
            "dissonance_roc",
            "aleatoric_roc",
            "entropy_roc",
            "vacuity_pr",
            "dissonance_pr",
            "aleatoric_pr",
            "entropy_pr",
        ]
        test_results = get_metrics(
            test_metrics, data, alpha=alpha, mode="test", task="mis"
        )
        for k, v in test_results.items():
            # self.log(f'val_evaluation/test_{k}', v)
            results[f"test_{k}"] = v
        # self.validation_step_outputs.append(results)
        for k, v in results.items():
            self.log(f"val_evaluation/{k}", v)
        return results

    def test_step(self, batch, batch_idx):
        data = batch
        alpha = self.gnn(data)
        test_metrics = [
            "id_oa",
            "id_f1",
            "id_ce",
            "id_kappa",
            "id_aa",
            "id_acc_classwise",
            "vacuity_roc",
            "dissonance_roc",
            "aleatoric_roc",
            "entropy_roc",
            "vacuity_pr",
            "dissonance_pr",
            "aleatoric_pr",
            "entropy_pr",
        ]
        test_results = get_metrics(
            test_metrics, data, alpha=alpha, mode="test", task="mis"
        )
        for k, v in test_results.items():
            self.log(f"test_evaluation/test_{k}", v)
        return test_results

    def configure_optimizers(self):
        flow_lr = self.train_cfg.lr
        flow_weight_decay = 0

        flow_params = list(self.gnn.flow.named_parameters())
        flow_param_names = [f"flow.{p[0]}" for p in flow_params]
        flow_param_weights = [p[1] for p in flow_params]

        all_params = list(self.gnn.named_parameters())
        params = [p[1] for p in all_params if p[0] not in flow_param_names]

        # all params except for flow
        flow_optimizer = optim.Adam(
            flow_param_weights, lr=flow_lr, weight_decay=flow_weight_decay
        )
        model_optimizer = optim.Adam(
            [
                {
                    "params": flow_param_weights,
                    "lr": flow_lr,
                    "weight_decay": flow_weight_decay,
                },
                {"params": params},
            ],
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )

        return model_optimizer, flow_optimizer


def train_gpn_mis(config: dict) -> dict:

    if (
        config["model"]["reconstruction_reg_weight"] == 0
        and config["model"]["tv_vacuity_reg_weight"] == 0
    ):
        submodel = ""
    elif (
        config["model"]["reconstruction_reg_weight"] == 0
        and config["model"]["tv_vacuity_reg_weight"] != 0
    ):
        submodel = "-TV"
    elif (
        config["model"]["reconstruction_reg_weight"] != 0
        and config["model"]["tv_vacuity_reg_weight"] == 0
    ):
        submodel = "-UR"
    else:
        submodel = "-UR-TV"

    pl.seed_everything(config["model"]["seed"])
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    data, num_classes = load_data(config["data"])

    datamodule = LightningNodeData(data, loader="full", batch_size=1)
    model = GPN_MIS(config=config, data=data, num_classes=num_classes, device="cuda")
    save_folder = os.path.join(
        f'{datetime.now().strftime("%Y_%m_%d")}',
        config["data"]["dataset"],
        f"runs-mis-GPN-{submodel}",
    )
    save_name = f'{config["model"]["model_name"]}_{datetime.now().strftime("%I_%M_%S_%p")}_{config["model"]["seed"]}'

    # params_version = f"{config['model']['reconstruction_reg_weight']}_{config['model']['tv_alpha_reg_weight']}_{config['model']['tv_vacuity_reg_weight']}"

    callbacks = []
    if config["training"]["stopping_metric"] == "val_CE":
        early_callback = EarlyStopping(
            monitor="val_evaluation/val_id_ce",
            mode="min",
            patience=config["training"]["stopping_patience"],
        )
        callbacks.append(early_callback)
    elif config["training"]["stopping_metric"] == "val_overall":
        early_callback = EarlyStopping(
            monitor="val_evaluation/val_overall",
            mode="max",
            patience=config["training"]["stopping_patience"],
        )
        callbacks.append(early_callback)
    else:
        early_callback = None
    if config["run"]["save_model"] == True:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join("saved_models", save_folder), filename=save_name
        )
        callbacks.append(checkpoint_callback)
    else:
        checkpoint_callback = None
    trainer = pl.Trainer(
        check_val_every_n_epoch=10,
        min_epochs=50,
        log_every_n_steps=1,
        max_epochs=config["training"]["epochs"],
        accelerator="gpu",
        devices=1,
        logger=TensorBoardLogger(
            save_dir=os.path.join("logs", save_folder), version=save_name
        ),
        fast_dev_run=False,
        enable_checkpointing=True,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule)
    result = trainer.test(model, datamodule)
    if config["run"]["save_model"] == True:
        test_metric = result[0].keys()
        test_values = result[0].values()
        df = pd.DataFrame.from_dict(
            {"Test metric": test_metric, "DataLoader 0": test_values}
        )
        df.to_csv(os.path.join("saved_models", save_folder, save_name + ".csv"))
    return result


if __name__ == "__main__":

    config = read_yaml_file(path="", directory="configs", file_name="mis_config_paviau")
    config["model"]["model_name"] = "GPN"
    train_gpn_mis(config)
