import os
import torch
from torch import optim
from datetime import datetime
import pandas as pd

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
from utils.graph_layers import (
    GCNNet,
    GATNet,
    GCNNetExp,
    GCNNetReExp,
    GCNNetExpPropagation,
    GCNNetPropagation,
)
from utils.loss import ModelLoss, cross_entropy

torch.set_float32_matmul_precision("medium")


# define the LightningModule
class GKDE_MIS(pl.LightningModule):
    def __init__(self, config, data, num_classes, device):
        super().__init__()
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
            self.model_cfg, self.data, mode="mis", dataset=config["data"]["dataset"]
        )

        self.x = self.data.x
        if hasattr(data, "endmemberS"):
            self.endmemberS = self.data.endmemberS

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

        else:
            return 0
        print(model)
        return model

    def training_step(self, batch, batch_idx):
        # print('training')
        data = batch
        alpha = self.gnn(data)
        loss_dict = self.criterion(alpha, data.y, mask=data.train_mask)
        loss = sum(loss_dict.values())
        # if loss > 1e5:
        #     import pdb
        #     pdb.set_trace()
        train_metrics = ["id_oa", "id_f1", "id_ce", "id_kappa", "id_aa"]
        train_results = get_metrics(
            train_metrics, data, alpha=alpha, mode="train", task="mis"
        )

        # Logging to TensorBoard by default
        self.log("loss/total_loss", loss)
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
            val_metrics, data, alpha=alpha, mode="validation", task="mis"
        )
        # val_ce = cross_entropy(alpha[data.val_mask], data.y[data.val_mask])
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
            results[f"test_{k}"] = v
        # results['val_id_ce'] = val_ce
        for k, v in results.items():
            self.log(f"val_evaluation/{k}", v)
        return results

    # def validation_epoch_end(self, outputs):
    #     for k in outputs[0].keys():
    #         tmp = torch.stack([x[k] for x in outputs]).mean()
    #         self.log(f'val_evaluation/{k}', tmp)

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
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )
        return optimizer


def train_misclassification_cleangraph(config: dict) -> dict:

    pl.seed_everything(config["model"]["seed"])
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    data, num_classes = load_data(config["data"])
    if "alpha_teacher_weight" in config["model"]:
        sigma = 0.2
        data.alpha_prior = torch.load(
            f'teacher/alpha_teacher_{config["data"]["dataset"]}_{sigma}.pt'
        )
    if "probability_teacher_weight" in config["model"]:
        data.probability_prior = torch.load(
            f'teacher/probability_teacher_{config["data"]["dataset"]}.pt'
        )

    datamodule = LightningNodeData(data, loader="full", batch_size=1)
    model = GKDE_MIS(config=config, data=data, num_classes=num_classes, device="cuda")
    save_folder = os.path.join(
        f'{datetime.now().strftime("%Y_%m_%d")}',
        config["data"]["dataset"],
        "runs-mis-cleangraph",
    )
    save_name = f'{config["model"]["model_name"]}_{datetime.now().strftime("%I_%M_%S_%p")}_{config["model"]["seed"]}'

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
            dirpath=os.path.join("saved_models", save_folder),
            filename=save_name,
        )
        callbacks.append(checkpoint_callback)
    else:
        checkpoint_callback = None
    # gradient_clip_val=1,
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
        df.to_csv(os.path.join("saved_models", save_folder, save_name + ".csv"))
    return result


if __name__ == "__main__":
    config = read_yaml_file(path="", directory="configs", file_name="mis_config_paviau")
    train_misclassification_cleangraph(config)
