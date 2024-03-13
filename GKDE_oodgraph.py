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

from utils.utils import load_ood_data
from utils.yaml import read_yaml_file
from utils.config import RunConfiguration, DataConfiguration
from utils.config import ModelConfiguration, TrainingConfiguration
from utils.metrics import get_metrics
from utils.graph_layers import (
    GCNNet,
    GATNet,
    GCNNetExp,
    GCNNetExpPropagation,
    GCNNetPropagation,
    GCNNetReExp,
)
from utils.loss import ModelLoss
from utils.gpn_model import GPNNet


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES']='5,6,7'
torch.set_float32_matmul_precision("medium")


# define the LightningModule
class GKDE_OOD(pl.LightningModule):
    def __init__(self, config, data, num_classes, device):
        super().__init__()
        self.save_hyperparameters(config["data"])
        self.save_hyperparameters(config["model"])
        self.save_hyperparameters(config["training"])
        # self.save_hyperparameters()
        self.run_cfg = RunConfiguration(**config["run"])
        self.data_cfg = DataConfiguration(**config["data"])
        self.model_cfg = ModelConfiguration(**config["model"])
        self.train_cfg = TrainingConfiguration(**config["training"])

        self.num_classes = num_classes
        self.data = data.to(device)

        self.gnn = self._build_model(device)
        self.criterion = ModelLoss(
            self.model_cfg, self.data, mode="ood", dataset=config["data"]["dataset"]
        )

        self.x = self.data.x
        if hasattr(data, "endmemberS"):
            self.endmemberS = self.data.endmemberS
            # initialize endmemberS_00 and prepare information for updating it
            num_features = self.data.x.shape[-1]
            self.endmemberS_00 = torch.ones([1, num_features], device=device)

        self.validation_step_outputs = []

    def _build_model(self, device):
        in_dim = self.data.num_features
        out_dim = self.num_classes
        if self.model_cfg.model_name == "GCN":
            model = GCNNet(
                in_dim,
                self.model_cfg.hidden_dim,
                out_dim,
                self.model_cfg.seed,
                self.model_cfg.drop_prob,
                bias=True,
            ).to(device)
        elif self.model_cfg.model_name == "GCNExp":
            model = GCNNetExp(
                in_dim,
                self.model_cfg.hidden_dim,
                out_dim,
                self.model_cfg.seed,
                self.model_cfg.drop_prob,
                bias=True,
            ).to(device)
        elif self.model_cfg.model_name == "GCNReExp":
            model = GCNNetReExp(
                in_dim,
                self.model_cfg.hidden_dim,
                out_dim,
                self.model_cfg.seed,
                self.model_cfg.drop_prob,
                bias=True,
            ).to(device)
        elif self.model_cfg.model_name == "GAT":
            model = GATNet(
                in_dim,
                self.model_cfg.hidden_dim,
                out_dim,
                self.model_cfg.seed,
                self.model_cfg.drop_prob,
                bias=True,
            ).to(device)
        elif self.model_cfg.model_name == "GCNExpProp":
            model = GCNNetExpPropagation(
                in_dim,
                self.model_cfg.hidden_dim,
                out_dim,
                self.model_cfg.seed,
                self.model_cfg.drop_prob,
                bias=True,
                iteration_step=self.model_cfg.iteration_step,
                teleport=self.model_cfg.teleport,
            ).to(device)
        elif self.model_cfg.model_name == "GCNProp":
            model = GCNNetPropagation(
                in_dim,
                self.model_cfg.hidden_dim,
                out_dim,
                self.model_cfg.seed,
                self.model_cfg.drop_prob,
                bias=True,
                iteration_step=self.model_cfg.iteration_step,
                teleport=self.model_cfg.teleport,
            ).to(device)

        elif self.model_cfg.model_name == "GPN":
            model = GPNNet(
                in_dim,
                self.model_cfg.hidden_dim,
                self.model_cfg.latent_dim,
                out_dim,
                self.model_cfg.radial_layers,
                self.model_cfg.drop_prob,
                iteration_step=self.model_cfg.iteration_step,
                teleport=self.model_cfg.teleport,
            ).to(device)
        else:
            return 0
        # print(model)
        return model

    def _get_optimal_s(self, alpha):
        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        scale = alpha_0 + self.num_classes
        scaled_probability = alpha / scale
        scaled_vacuity = self.num_classes / scale
        m = self.x - torch.mm(scaled_probability, self.endmemberS)
        # print(m.size(), vacuity.size(), self.endmemberS_00.size())
        den = torch.mm(scaled_vacuity.T, m)
        num = torch.sum(torch.square(scaled_vacuity))
        self.endmemberS_00 = den / num

    def forward(self, data):
        return self.gnn(data)

    def training_step(self, batch, batch_idx):
        # print('training')
        data = batch
        alpha = self(data)
        if hasattr(data, "endmemberS"):
            loss_dict = self.criterion(
                alpha, data.y, self.endmemberS_00, mask=data.train_mask
            )
        else:
            loss_dict = self.criterion(alpha, data.y, mask=data.train_mask)
        if self.model_cfg.reconstruction_reg_weight != None:
            self._get_optimal_s(alpha.detach())
        loss = sum(loss_dict.values())
        # if loss > 1e5:
        #     import pdb
        #     pdb.set_trace()

        # self.log('loss/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "loss/total_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        train_metrics = ["id_oa", "id_f1", "id_ce", "id_kappa", "id_aa"]
        # def get_metrics(metrics:list, data:Data, logit:Tensor =None, alpha:Tensor = None, mode:str='test',  task:str = 'ood')
        train_results = get_metrics(
            train_metrics, data, alpha=alpha, mode="train", task="ood"
        )
        # Logging to TensorBoard by default

        for k, v in loss_dict.items():
            self.log(f"loss/{k}", v)
            # self.logger.experiment.log_metric(key = f'loss/{k}', value = v)
        for k, v in train_results.items():
            # self.logger.experiment.log_metric(key =f'train_evaluation/{k}', value=v)
            self.log(f"train_evaluation/train_{k}", v)
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
            val_metrics, data, alpha=alpha, mode="validation", task="ood"
        )
        for k, v in val_results.items():
            results[f"val_{k}"] = v
        results[f"val_overall"] = val_results["vacuity_roc"] + val_results["id_f1"]
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
            test_metrics, data, alpha=alpha, mode="test", task="ood"
        )
        for k, v in test_results.items():
            results[f"test_{k}"] = v
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
            test_metrics, data, alpha=alpha, mode="test", task="ood"
        )
        for k, v in test_results.items():
            self.log(f"test_evaluation/test_{k}", v)
        return test_results

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )
        return optimizer


def train_GKDE_ood(config: dict) -> dict:

    pl.seed_everything(config["model"]["seed"])
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    data, num_classes = load_ood_data(config["data"])
    if "alpha_teacher_weight" in config["model"]:
        sigma = 0.2
        data.alpha_prior = torch.load(
            f'teacher/alpha_teacher_{config["data"]["ood_left_out_classes"][0]}_{sigma}.pt'
        )
    if "probability_teacher_weight" in config["model"]:
        data.probability_prior = torch.load(
            f'teacher/probability_teacher_{config["data"]["dataset"]}_{config["data"]["ood_left_out_classes"][0]}.pt'
        )

    datamodule = LightningNodeData(data, loader="full", batch_size=1)
    model = GKDE_OOD(config=config, data=data, num_classes=num_classes, device="cuda")

    save_folder = os.path.join(
        f'{datetime.now().strftime("%Y_%m_%d")}',
        config["data"]["dataset"] + "_" + str(config["data"]["ood_left_out_classes"]),
        "runs-ood-GKDE",
    )
    save_name = (
        f'{config["model"]["model_name"]}_{datetime.now().strftime("%I_%M_%S_%p")}'
    )

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
            dirpath=os.path.join("saved_models_debug", save_folder),
            filename=save_name,
        )
        callbacks.append(checkpoint_callback)
    else:
        checkpoint_callback = None

    trainer = pl.Trainer(
        check_val_every_n_epoch=10,
        # gradient_clip_val=1,
        min_epochs=50,
        log_every_n_steps=1,
        max_epochs=config["training"]["epochs"],
        accelerator="gpu",
        devices=1,
        logger=TensorBoardLogger(
            save_dir=os.path.join("logs_debug", save_folder), version=save_name
        ),
        fast_dev_run=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
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
        df.to_csv(os.path.join("saved_models_debug", save_folder, save_name + ".csv"))
    return result


if __name__ == "__main__":
    config = read_yaml_file(path="", directory="configs", file_name="ood_config_paviau")
    train_GKDE_ood(config)
