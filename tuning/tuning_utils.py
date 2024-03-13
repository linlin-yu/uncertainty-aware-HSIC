from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import torch
import pytorch_lightning as pl
from torch_geometric.data.lightning import LightningNodeData

from utils.utils import load_data, load_ood_data
from GKDE_misclassification_cleangraph import GKDE_MIS
from GKDE_oodgraph import GKDE_OOD
from gpn_misclassification_cleangraph import GPN_MIS
from gpn_oodgraph import GPN_OOD


def tune_misclassification_cleangraph(config):

    pl.seed_everything(42)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    sigma = 0.2
    data, num_classes = load_data(config['data'])
    data.alpha_prior = torch.load(f'teacher/alpha_teacher_{config["data"]["dataset"]}_{sigma}.pt')
    datamodule = LightningNodeData(data, loader='full', batch_size = 1)
    model = GKDE_MIS(config=config, data= data, num_classes = num_classes, device='cuda')
    
    # Create the Tune Reporting Callback
    report_metrics = ['train_evaluation/training_id_oa', 'train_evaluation/training_id_f1',
                'val_evaluation/val_id_oa', 'val_evaluation/val_id_f1', 'val_evaluation/val_id_ce', 
                'val_evaluation/test_id_oa', 'val_evaluation/test_id_f1', 
                'val_evaluation/val_dissonance_roc', 'val_evaluation/val_vacuity_roc', 
                'val_evaluation/val_entropy_roc', 'val_evaluation/val_aleatoric_roc', 
                'val_evaluation/val_dissonance_pr', 'val_evaluation/val_vacuity_pr', 
                'val_evaluation/val_entropy_pr', 'val_evaluation/val_aleatoric_pr', 
                'val_evaluation/val_overall', 
                'val_evaluation/test_dissonance_roc', 'val_evaluation/test_vacuity_roc', 
                'val_evaluation/test_entropy_roc', 'val_evaluation/test_aleatoric_roc', 
                'val_evaluation/test_dissonance_pr', 'val_evaluation/test_vacuity_pr', 
                'val_evaluation/test_entropy_pr', 'val_evaluation/test_aleatoric_pr', ]
    
    save_folder = os.path.join(f'{datetime.now().strftime("%Y_%m_%d")}', 
                               config['data']['dataset'],  
                               'runs-mis-cleangraph')
    save_name = f'{config["model"]["model_name"]}_{datetime.now().strftime("%I_%M_%S_%p")}'
    
    callbacks = []
    if config['training']['stopping_metric'] == 'val_CE':
        early_callback =  EarlyStopping(monitor="val_evaluation/val_id_ce", mode="min", patience=config['training']['stopping_patience'])
        callbacks.append(early_callback)
    elif config['training']['stopping_metric'] == 'val_overall':
        early_callback =  EarlyStopping(monitor="val_evaluation/val_overall", mode="max", patience=50)
        callbacks.append(early_callback)
    else:
        early_callback = None
    if config['run']['save_model'] == True: 
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('saved_tune_models', save_folder), filename=save_name,)
        callbacks.append(checkpoint_callback)
    else:
        checkpoint_callback = None
    callbacks.append(TuneReportCallback(metrics=report_metrics, on="validation_end"))

    trainer = pl.Trainer(check_val_every_n_epoch=10, 
                        min_epochs = 50, 
                        max_epochs=config['training']['epochs'], 
                        accelerator="gpu",
                        devices=1, 
                        logger=True,
                        callbacks= callbacks, 
                        enable_progress_bar=False,
                        fast_dev_run=False)
    
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
    # return result
    
def tune_gkde_ood(config):


    pl.seed_everything(42)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    data, num_classes = load_ood_data(config['data'])
    sigma = 0.2
    data.alpha_prior = torch.load(f'teacher/alpha_teacher_{config["data"]["dataset"]}_{config["data"]["ood_left_out_classes"][0]}_{sigma}.pt')
    # data.probability_prior = torch.load(f'teacher/probability_teacher_{config["data"]["dataset"]}_{config["data"]["ood_left_out_classes"][0]}.pt')
    datamodule = LightningNodeData(data, loader='full', batch_size = 1)
    model = GKDE_OOD(config=config, data= data, num_classes = num_classes, device='cuda')
    
    # Create the Tune Reporting Callback
    report_metrics = ['train_evaluation/training_id_oa', 'train_evaluation/training_id_f1',
                'val_evaluation/val_id_oa', 'val_evaluation/val_id_f1', 'val_evaluation/val_id_ce', 
                'val_evaluation/test_id_oa', 'val_evaluation/test_id_f1', 
                'val_evaluation/val_dissonance_roc', 'val_evaluation/val_vacuity_roc', 
                'val_evaluation/val_entropy_roc', 'val_evaluation/val_aleatoric_roc', 
                'val_evaluation/val_dissonance_pr', 'val_evaluation/val_vacuity_pr', 
                'val_evaluation/val_entropy_pr', 'val_evaluation/val_aleatoric_pr', 
                'val_evaluation/val_overall', 
                'val_evaluation/test_dissonance_roc', 'val_evaluation/test_vacuity_roc', 
                'val_evaluation/test_entropy_roc', 'val_evaluation/test_aleatoric_roc', 
                'val_evaluation/test_dissonance_pr', 'val_evaluation/test_vacuity_pr', 
                'val_evaluation/test_entropy_pr', 'val_evaluation/test_aleatoric_pr', ]
    
    save_folder = os.path.join(f'{datetime.now().strftime("%Y_%m_%d")}',
                               config['data']['dataset'] + '_' + str(config['data']['ood_left_out_classes']),  
                               'runs-ood-GKDE')
    save_name = f'{config["model"]["model_name"]}_{datetime.now().strftime("%I_%M_%S_%p")}'
    
    callbacks = []
    if config['training']['stopping_metric'] == 'val_CE':
        early_callback =  EarlyStopping(monitor="val_evaluation/val_id_ce", mode="min", patience=config['training']['stopping_patience'])
        callbacks.append(early_callback)
    elif config['training']['stopping_metric'] == 'val_overall':
        early_callback =  EarlyStopping(monitor="val_evaluation/val_overall", mode="max", patience=config['training']['stopping_patience'])
        callbacks.append(early_callback)
    else:
        early_callback = None
        
    if config['run']['save_model'] == True: 
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('saved_models_tune', save_folder), filename=save_name,)
        callbacks.append(checkpoint_callback)
    else:
        checkpoint_callback = None
        
    callbacks.append(TuneReportCallback(metrics=report_metrics, on="validation_end"))
    
    trainer = pl.Trainer(check_val_every_n_epoch=10, 
                        min_epochs = 50, 
                        max_epochs=config['training']['epochs'], 
                        accelerator="gpu",
                        devices=1, 
                        logger=False,
                        callbacks= callbacks, 
                        enable_progress_bar=False,
                        fast_dev_run=False)
    
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
    # return result
    
def tune_gpn_mis(config):
    
    os.chdir("/home/lxy190003/data/HSI_torch")

    pl.seed_everything(42)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    # torch.backends.cudnn.determinstic = True
    # torch.backends.cudnn.benchmark = False

    data, num_classes = load_data(config['data'])
    datamodule = LightningNodeData(data, loader='full', batch_size = 1)
    model = GPN_MIS(config=config, data= data, num_classes = num_classes, device='cuda')
    
    # Create the Tune Reporting Callback
    report_metrics = ['train_evaluation/training_id_oa', 'train_evaluation/training_id_f1',
                'val_evaluation/val_id_oa', 'val_evaluation/val_id_f1', 'val_evaluation/val_id_ce', 
                'val_evaluation/test_id_oa', 'val_evaluation/test_id_f1', 
                'val_evaluation/val_dissonance_roc', 'val_evaluation/val_vacuity_roc', 
                'val_evaluation/val_entropy_roc', 'val_evaluation/val_aleatoric_roc', 
                'val_evaluation/val_dissonance_pr', 'val_evaluation/val_vacuity_pr', 
                'val_evaluation/val_entropy_pr', 'val_evaluation/val_aleatoric_pr', 
                'val_evaluation/val_overall', 
                'val_evaluation/test_dissonance_roc', 'val_evaluation/test_vacuity_roc', 
                'val_evaluation/test_entropy_roc', 'val_evaluation/test_aleatoric_roc', 
                'val_evaluation/test_dissonance_pr', 'val_evaluation/test_vacuity_pr', 
                'val_evaluation/test_entropy_pr', 'val_evaluation/test_aleatoric_pr', ]
    
    save_folder = os.path.join(f'{datetime.now().strftime("%Y_%m_%d")}',
                               config['data']['dataset'],
                               'runs-mis')
    save_name = f'{config["model"]["model_name"]}_{datetime.now().strftime("%I_%M_%S_%p")}'
    
    callbacks = []
    if config['training']['stopping_metric'] == 'val_overall': 
        early_callback =  EarlyStopping(monitor="val_evaluation/val_overall", mode="max", patience=config['training']['stopping_patience'])
        callbacks.append(early_callback)
    elif config['training']['stopping_metric'] == 'val_CE': 
        early_callback =  EarlyStopping(monitor="val_evaluation/val_id_ce", mode="min", patience=config['training']['stopping_patience'])
        callbacks.append(early_callback)
    else:
        early_callback = None
        
    if config['run']['save_model'] == True: 
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('saved_models', save_folder),filename=save_name,)
        callbacks.append(checkpoint_callback)
    else:
        checkpoint_callback = None
    callbacks.append(TuneReportCallback(metrics=report_metrics, on="validation_end"))
    
    
    trainer = pl.Trainer(check_val_every_n_epoch=10, 
                        min_epochs = 50, 
                        max_epochs=config['training']['epochs'], 
                        accelerator="gpu",
                        devices=1, 
                        logger=True,
                        callbacks= callbacks, 
                        enable_progress_bar=False,
                        fast_dev_run=False)
    
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
    

def tune_gpn_ood(config):
    

    pl.seed_everything(42)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    data, num_classes = load_ood_data(config['data'])
    datamodule = LightningNodeData(data, loader='full', batch_size = 1)
    model = GPN_OOD(config=config, data= data, num_classes = num_classes, device='cuda')
    
    # Create the Tune Reporting Callback
    report_metrics = ['train_evaluation/training_id_oa', 'train_evaluation/training_id_f1',
                'val_evaluation/val_id_oa', 'val_evaluation/val_id_f1', 'val_evaluation/val_id_ce', 
                'val_evaluation/test_id_oa', 'val_evaluation/test_id_f1', 
                'val_evaluation/val_dissonance_roc', 'val_evaluation/val_vacuity_roc', 
                'val_evaluation/val_entropy_roc', 'val_evaluation/val_aleatoric_roc', 
                'val_evaluation/val_dissonance_pr', 'val_evaluation/val_vacuity_pr', 
                'val_evaluation/val_entropy_pr', 'val_evaluation/val_aleatoric_pr', 
                'val_evaluation/val_overall', 
                'val_evaluation/test_dissonance_roc', 'val_evaluation/test_vacuity_roc', 
                'val_evaluation/test_entropy_roc', 'val_evaluation/test_aleatoric_roc', 
                'val_evaluation/test_dissonance_pr', 'val_evaluation/test_vacuity_pr', 
                'val_evaluation/test_entropy_pr', 'val_evaluation/test_aleatoric_pr', ]
    
    save_folder = os.path.join(f'{datetime.now().strftime("%Y_%m_%d")}',
                               config['data']['dataset']+ '_' + str(config['data']['ood_left_out_classes']),
                               'runs-ood-GPN-0-0')
    save_name = f'{config["model"]["model_name"]}_{datetime.now().strftime("%I_%M_%S_%p")}'
    
    callbacks = []
    if config['training']['stopping_metric'] == 'val_overall': 
        early_callback =  EarlyStopping(monitor="val_evaluation/val_overall", mode="max", patience=config['training']['stopping_patience'])
        callbacks.append(early_callback)
    elif config['training']['stopping_metric'] == 'val_CE': 
        early_callback =  EarlyStopping(monitor="val_evaluation/val_id_ce", mode="min", patience=config['training']['stopping_patience'])
        callbacks.append(early_callback)
    else:
        early_callback = None
        
    if config['run']['save_model'] == True: 
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('saved_models', save_folder),filename=save_name,)
        callbacks.append(checkpoint_callback)
    else:
        checkpoint_callback = None
    callbacks.append(TuneReportCallback(metrics=report_metrics, on="validation_end"))
    
    trainer = pl.Trainer(check_val_every_n_epoch=10, 
                        min_epochs = 50, 
                        max_epochs=config['training']['epochs'], 
                        accelerator="gpu",
                        devices=1, 
                        logger=True,
                        callbacks= callbacks, 
                        enable_progress_bar=False,
                        fast_dev_run=False)
    
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)