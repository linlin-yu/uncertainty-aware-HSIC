import os
from classification_cleangraph import ClassificationCleanGraph
from torch_geometric.data.lightning import LightningNodeData
from utils.utils import load_data, load_ood_data
from utils.yaml import read_yaml_file
import pytorch_lightning as pl
from classification_cleangraph import ClassificationCleanGraph
import torch.nn.functional as F
import torch
from pytorch_lightning.callbacks import EarlyStopping


def generate_probability_teacher_clean(config):
    data, num_classes = load_data(config["data"])
    datamodule = LightningNodeData(data, loader="full", batch_size=1)
    model = ClassificationCleanGraph(
        config=config, data=data, num_classes=num_classes, device="cuda"
    )
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

    trainer = pl.Trainer(
        check_val_every_n_epoch=10,
        min_epochs=50,
        log_every_n_steps=1,
        max_epochs=config["training"]["epochs"],
        accelerator="gpu",
        devices=1,
        fast_dev_run=False,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule)
    result = trainer.test(model, datamodule)
    print(result)

    # disable randomness, dropout, etc...
    model.eval()

    # predict with the model
    data = data.to("cuda")
    logits = model.gnn(data).detach().cpu()
    # alpha_0 = alpha.sum(dim=-1)
    data = data.cpu()

    pred = F.softmax(logits, dim=1)
    # save the probability teacher
    os.makedirs("teacher", exist_ok=True)
    torch.save(pred, f"teacher/probability_teacher_{config['data']['dataset']}.pt")


def generate_probability_teacher_ood(i, config):
    config["data"]["ood_left_out_classes"] = [
        i,
    ]
    data, num_classes = load_ood_data(config["data"])
    datamodule = LightningNodeData(data, loader="full", batch_size=1)
    model = ClassificationCleanGraph(
        config=config, data=data, num_classes=num_classes, device="cuda"
    )
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

    trainer = pl.Trainer(
        check_val_every_n_epoch=10,
        min_epochs=50,
        log_every_n_steps=1,
        max_epochs=config["training"]["epochs"],
        accelerator="gpu",
        devices=1,
        fast_dev_run=False,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule)
    result = trainer.test(model, datamodule)
    print(result)

    # disable randomness, dropout, etc...
    model.eval()

    # predict with the model
    data = data.cpu()
    logits = model.gnn(data).detach().cpu()
    # alpha_0 = alpha.sum(dim=-1)

    pred = F.softmax(logits, dim=1)
    # save the probability teacher
    os.makedirs("teacher", exist_ok=True)
    torch.save(pred, f"teacher/probability_teacher_{config['data']['dataset']}_{i}.pt")


# load settings
dataset = "paviaU"

## get probability teacher for OOD graphs
config = read_yaml_file(path="", directory="configs", file_name=f"ood_config_{dataset}")
i = 3
assert config["model"]["model_name"] == "GCN"
generate_probability_teacher_ood(i, config)

# ## get probability teacher for clean graphs
# config = read_yaml_file(path='', directory='configs', file_name=f'classification_config_{dataset}')
# assert config['model']['model_name'] == 'GCN'
# generate_probability_teacher_clean(config)
