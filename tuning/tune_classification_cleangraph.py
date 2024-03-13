import os
import ray
from ray import tune, air
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from utils.yaml import read_yaml_file
from classification_cleangraph import tune_classification_clean
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES']='4,5,6'

user_root = '/home/user/data/HSI_torch/'
ray.init(include_dashboard=True, dashboard_host="0.0.0.0", log_to_driver=False, _temp_dir=f'{user_root}/tmp')
report_parameters = ['model/hidden_dim', 'model/drop_prob', 
                     'training/lr', 'training/weight_decay']
report_metrics = ['val_evaluation/val_id_oa', 'val_evaluation/val_id_f1', 'val_evaluation/val_id_ce', 
                'val_evaluation/val_id_kappa', 'val_evaluation/val_id_aa', 
                'val_evaluation/val_entropy_roc', 'val_evaluation/val_aleatoric_roc',
                'val_evaluation/test_id_oa', 'val_evaluation/test_id_f1', 'val_evaluation/test_id_ce', 
                'val_evaluation/test_id_kappa', 'val_evaluation/test_id_aa', 
                'val_evaluation/test_entropy_roc', 'val_evaluation/test_aleatoric_roc']

def tune_main(config, num_samples=1, gpus_per_trial=1, save_dir='tune'):
    reporter = CLIReporter(
        parameter_columns=report_parameters,
        metric_columns=report_metrics)

    analysis = tune.run(
        tune.with_parameters(
            tune_classification_clean),
        resources_per_trial={
            "cpu": 8,
            "gpu": gpus_per_trial
        },
        metric="val_evaluation/val_id_oa",
        mode="max",
        config=config,
        num_samples=num_samples,
        progress_reporter=reporter,
        local_dir = os.path.join(f'{user_root}/ray_results', save_dir),
        name="tune_classification_cleangraph")

    print("Best hyperparameters found were: ", analysis.best_config)
    return analysis

config = read_yaml_file(path='', directory='configs', file_name='classification_config')
config['model']['hidden_dim'] = tune.grid_search([8,16,32,64,128])
config['model']['drop_prob'] = tune.grid_search([0, 0.1, 0.3, 0.5, 0.8])
config['training']['lr'] = tune.grid_search([0.1, 0.01,0.001])
config['training']['weight_decay'] = tune.grid_search([0.0005, 0.0001,0.001, 0.005])
analysis = tune_main(config = config, save_dir='tune-classification_config')
df = analysis.results_df
dataset = config["data"]["dataset"]
df.to_csv(f'results/tune-classification-cleangraph-{dataset}.csv')
