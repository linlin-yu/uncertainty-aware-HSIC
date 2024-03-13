from utils.utils import load_data, load_ood_data
from utils.yaml import read_yaml_file
from utils.utils import compute_kde
import torch
import os
import multiprocessing

dataset = "paviaU"
# # load settings and model
config = read_yaml_file(path="", directory="configs", file_name=f"ood_config_{dataset}")


# Define a function to process a single (sigma, i) pair and save the result
# sigma is the parameter in GKDE model
# i is the class index of OOD nodes
def generate_alpha_teacher_clean(sigma, config):
    data, num_classes = load_data(config["data"])
    alpha_prior = compute_kde(data, num_classes, sigma)
    os.makedirs("teacher", exist_ok=True)
    torch.save(
        alpha_prior, f"teacher/alpha_teacher_{config['data']['dataset']}_{sigma}.pt"
    )


def generate_alpha_teacher_ood(sigma, i, config):
    config["data"]["ood_left_out_classes"] = [
        i,
    ]
    data, num_classes = load_ood_data(config["data"])
    alpha_prior = compute_kde(data, num_classes, sigma)
    os.makedirs("teacher", exist_ok=True)
    torch.save(
        alpha_prior, f"teacher/alpha_teacher_{config['data']['dataset']}_{i}_{sigma}.pt"
    )


# '''
# Run multiple settings parallelly
# '''
# # Define the list of sigma values and i values
# sigma_values = [0.1, 0.2, 0.5, 1, 2, 5, 10]
# i_values = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

# # Create a multiprocessing pool with multiple CPU cores
# with multiprocessing.Pool(processes=32) as pool:
#     # Iterate over each sigma and i pair and map them to the pool for parallel processing
#     results = []
#     for sigma in sigma_values:
#         for i in i_values:
#             result = pool.apply_async(process_sigma_i, (sigma, i, config))
#             results.append(result)

#     # Wait for all tasks to complete
#     for result in results:
#         result.get()

"""
Run Signle settings
"""
# for OOD graph
sigma = 0.1
i = 3
generate_alpha_teacher_ood(sigma, i, config)
# # for ID graph
# sigma = 0.1
# generate_alpha_teacher_ood(sigma, config)
