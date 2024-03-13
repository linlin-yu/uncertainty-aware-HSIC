import os
import ray
from ray import tune, air
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from utils.yaml import read_yaml_file
from gpn_ood import tune_gpn_ood
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='4,5,6,7'

user_root = '/home/user/data/HSI_torch/'
ray.init(include_dashboard=True, dashboard_host="0.0.0.0", log_to_driver=False, _temp_dir=f'{user_root}/tmp')
# report_parameters = ['data/ood_left_out_classes', 'model/uce_loss_weight', 'model/entropy_reg_weight', 'model/reconstruction_reg_weight',
#                     'model/tv_alpha_reg_weight', 'model/tv_vacuity_reg_weight']
#  'model/tv_vacuity_reg_weight'
#                      'model/probability_teacher_weight',
report_parameters = ['data/ood_left_out_classes', 'model/model_name',
                     'model/uce_loss_weight', 'model/entropy_reg_weight', 'model/reconstruction_reg_weight',
                    # 'model/tv_alpha_reg_weight',
                    'model/tv_vacuity_reg_weight',
                    'model/latent_dim', 'model/hidden_dim','model/radial_layers', 'model/teleport', 
                     'model/iteration_step']
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

def tune_ood_detection(config, num_samples=5, gpus_per_trial=1, save_dir='tune'):
    reporter = CLIReporter(
        parameter_columns=report_parameters,
        metric_columns=report_metrics)

    analysis = tune.run(
        tune.with_parameters(
            tune_gpn_ood),
        resources_per_trial={
            "cpu": 2,
            "gpu": gpus_per_trial
        },
        metric="val_evaluation/val_overall",
        mode="max",
        config=config,
        num_samples=num_samples,
        progress_reporter=reporter,
        local_dir = os.path.join(f'{user_root}/ray_results', save_dir),
        name="tune_gpn_ood")

    print("Best hyperparameters found were: ", analysis.best_config)
    print(analysis)
    return analysis

config = read_yaml_file(path='', directory='configs', file_name='ood_config_houston')

# config['model']['drop_prob'] = tune.grid_search([0.01, 0.001, 0.0001, 0.00001, 0])
# config['model']['hidden_dim'] = tune.grid_search([128, 256])
# config['model']['latent_dim'] = tune.grid_search([10])
# config['model']['radial_layers'] = tune.grid_search([10, 16])
# config['model']['teleport'] = tune.grid_search([0.2])

config['model']['uce_loss_weight'] = 1
config['model']['model_name'] = 'GPN'
ood = 4
config['data']['ood_left_out_classes'] = [ood]
dataset = config["data"]["dataset"]
model_name = config["model"]["model_name"]


# suffix = 'lrwder'
# config['model']['entropy_reg_weight'] = tune.grid_search([0.001, 0.0001, 0.00001])
# config['training']['weight_decay'] = tune.grid_search([0.0005, 0.0001, 0.001, 0.005])
# config['training']['lr'] = tune.grid_search([0.1, 0.01, 0.001, 0.0001])
# config['model']['reconstruction_reg_weight'] = 0
# config['model']['tv_vacuity_reg_weight'] = 0

suffix = 'reg'
config['training']['lr'] = 0.0001
config['training']['weight_decay'] = 0.001
config['model']['entropy_reg_weight'] = 0.0001
config['model']['reconstruction_reg_weight'] = tune.grid_search([1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0])
# config['model']['tv_alpha_reg_weight'] = tune.grid_search([0.0001, 0.00001, 0])
config['model']['tv_vacuity_reg_weight'] = tune.grid_search([ 0.01, 0.001, 0.0001, 0.00001, 0])


analysis = tune_ood_detection(config = config, save_dir=f'tune-ood-{dataset}-{model_name}-{ood}-{suffix}')
df = analysis.results_df
df.to_csv(f'results/tune-ood-{dataset}-{model_name}-{ood}-{suffix}.csv')
