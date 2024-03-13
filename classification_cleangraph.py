import os, time
import torch
from torch import optim
from torch import nn
from torch_geometric.data.lightning import LightningNodeData
from datetime import datetime
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

from torchmetrics.functional import (
    f1_score,
    cohen_kappa,
    confusion_matrix,
    accuracy,
    auroc,
)
import torch.nn.functional as F

from utils.utils import load_ood_data, load_data
from utils.yaml import read_yaml_file
from utils.config import RunConfiguration, DataConfiguration
from utils.config import ModelConfiguration, TrainingConfiguration
from utils.metrics import get_metrics
from utils.graph_layers import GCNNet, GATNet, GCNNetClassification
from utils.loss import ModelLoss

import multiprocessing

torch.set_float32_matmul_precision("medium")
"""
Build a classification model with CE loss on clean graph
"""


class ClassificationCleanGraph(pl.LightningModule):
    def __init__(self, config, data, num_classes, device):
        super().__init__()
        self.save_hyperparameters(config["model"])
        self.save_hyperparameters(config["training"])

        self.run_cfg = RunConfiguration(**config["run"])
        self.data_cfg = DataConfiguration(**config["data"])
        self.model_cfg = ModelConfiguration(**config["model"])
        self.train_cfg = TrainingConfiguration(**config["training"])

        self.num_classes = num_classes
        self.data = data.to(device)

        self.gnn = self._build_model(device)
        self.criterion = nn.CrossEntropyLoss()

    def _build_model(self, device):
        in_dim = self.data.num_features
        out_dim = self.num_classes
        if self.model_cfg.model_name == "GCN":
            model = GCNNetClassification(
                in_dim,
                self.model_cfg.hidden_dim,
                out_dim,
                self.model_cfg.seed,
                self.model_cfg.drop_prob,
                bias=True,
            ).to(device)
        else:
            return 0
        print(model)
        return model

    def training_step(self, batch, batch_idx):
        # print('training')
        data = batch
        logits = self.gnn(data)
        loss = self.criterion(logits[data.train_mask], data.y[data.train_mask])
        train_metrics = [
            "id_oa",
            "id_f1",
            "id_ce",
            "id_kappa",
            "id_aa",
            "entropy_roc",
            "aleatoric_roc",
        ]
        train_results = get_metrics(
            train_metrics, data, logit=logits, mode="train", task="mis"
        )
        self.log("train_loss", loss)
        for k, v in train_results.items():
            self.log(f"train_evaluation/train_{k}", v)
        return loss

    def validation_step(self, batch, batch_idx):
        # print('validation')
        data = batch
        logits = self.gnn(data)
        results = {}
        val_metrics = [
            "id_oa",
            "id_f1",
            "id_ce",
            "id_kappa",
            "id_aa",
            "entropy_roc",
            "entropy_pr",
            "aleatoric_roc",
            "aleatoric_pr",
        ]
        val_results = get_metrics(
            val_metrics, data, logit=logits, mode="validation", task="mis"
        )
        for k, v in val_results.items():
            results[f"val_{k}"] = v
        test_metrics = [
            "id_oa",
            "id_f1",
            "id_ce",
            "id_kappa",
            "id_aa",
            "entropy_roc",
            "entropy_pr",
            "aleatoric_roc",
            "aleatoric_pr",
        ]
        test_results = get_metrics(
            test_metrics, data, logit=logits, mode="test", task="mis"
        )
        for k, v in test_results.items():
            results[f"test_{k}"] = v
        for k, v in results.items():
            self.log(f"val_evaluation/{k}", v)
        return results

    def test_step(self, batch, batch_idx):
        data = batch
        logits = self.gnn(data)
        test_metrics = {}
        test_metrics = [
            "id_oa",
            "id_f1",
            "id_ce",
            "id_kappa",
            "id_aa",
            "id_acc_classwise",
            "entropy_roc",
            "entropy_pr",
            "aleatoric_roc",
            "aleatoric_pr",
        ]
        test_results = get_metrics(
            test_metrics, data, logit=logits, mode="test", task="mis"
        )
        for k, v in test_results.items():
            self.log(f"test_evaluation/test_{k}", v)
        return test_metrics

    def configure_optimizers(self):
        print(self.train_cfg)
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )
        return optimizer


def train_classification_clean(config: dict) -> dict:

    seed = config["model"]["seed"]
    pl.seed_everything(config["model"]["seed"])
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    data, num_classes = load_data(config["data"])
    datamodule = LightningNodeData(data, loader="full", batch_size=1)
    model = ClassificationCleanGraph(
        config=config, data=data, num_classes=num_classes, device="cuda"
    )
    save_folder = os.path.join(
        f'{datetime.now().strftime("%Y_%m_%d")}',
        config["data"]["dataset"],
        "runs-classification-cleangraph",
    )
    save_name = f'{config["model"]["model_name"]}_{datetime.now().strftime("%I_%M_%S_%p")}_{seed}'

    callbacks = []
    if config["training"]["stopping_metric"] == "val_CE":
        early_callback = EarlyStopping(
            monitor="val_evaluation/val_id_ce",
            mode="min",
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
    trainer = pl.Trainer(
        check_val_every_n_epoch=10,
        min_epochs=50,
        log_every_n_steps=1,
        max_epochs=config["training"]["epochs"],
        accelerator="gpu",
        devices=1,
        logger=TensorBoardLogger(
            save_dir=os.path.join("logs", save_folder), name=save_name
        ),
        fast_dev_run=False,
        enable_progress_bar=False,
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


def tune_classification_clean(config):
    from ray import tune
    from ray.tune.integration.pytorch_lightning import TuneReportCallback

    pl.seed_everything(42)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    data, num_classes = load_data(config["data"])
    datamodule = LightningNodeData(data, loader="full", batch_size=1)
    model = ClassificationCleanGraph(
        config=config, data=data, num_classes=num_classes, device="cuda"
    )

    # Create the Tune Reporting Callback
    report_metrics = [
        "val_evaluation/val_id_oa",
        "val_evaluation/val_id_f1",
        "val_evaluation/val_id_ce",
        "val_evaluation/val_id_kappa",
        "val_evaluation/val_id_aa",
        "val_evaluation/val_entropy_roc",
        "val_evaluation/val_aleatoric_roc",
        "val_evaluation/test_id_oa",
        "val_evaluation/test_id_f1",
        "val_evaluation/test_id_ce",
        "val_evaluation/test_id_kappa",
        "val_evaluation/test_id_aa",
        "val_evaluation/test_entropy_roc",
        "val_evaluation/test_aleatoric_roc",
    ]

    callbacks = [TuneReportCallback(metrics=report_metrics, on="validation_end")]
    trainer = pl.Trainer(
        check_val_every_n_epoch=10,
        log_every_n_steps=1,
        min_epochs=50,
        max_epochs=config["training"]["epochs"],
        accelerator="gpu",
        devices=1,
        logger=TensorBoardLogger(
            save_dir="runs-tune-classification-clean",
            name=f"classification-clean-{time.time()}",
        ),
        callbacks=callbacks,
        enable_progress_bar=False,
        fast_dev_run=False,
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


def process_seed_main(seed, config):
    config["model"]["seed"] = seed
    result_summary = train_classification_clean(config)
    return result_summary


if __name__ == "__main__":
    config = read_yaml_file(
        path="", directory="configs", file_name="classification_config_paviau"
    )

    # Change the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method("spawn", force=True)

    seed_list = [0, 111, 222, 333, 444, 555, 666, 777, 888, 999]
    # Create a multiprocessing pool with multiple CPU cores
    cpu_counts = 4
    # cpu_counts = multiprocessing.cpu_count()
    gpu_indices = [4, 5, 6, 7]

    with multiprocessing.Pool(processes=cpu_counts) as pool:

        # Iterate over each seed and map them to the pool for parallel processing
        results = []
        for i, seed in enumerate(seed_list):
            result = pool.apply_async(process_seed_main, (seed, config))
            results.append(result)

        # Wait for all tasks to complete
        for result in results:
            result.get()
