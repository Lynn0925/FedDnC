import numpy as np
import torch
import torch.nn.functional as f
import copy
from prettytable import PrettyTable
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
import os
import time
from typing import Dict, List, Any, Union, Optional

def average_weights(w,idxs_users,datanumber_client,islist=False):
    """Returns the weighted average of client weights."""
    total_data_points = sum([datanumber_client[r] for r in idxs_users])

    w_avg = copy.deepcopy(w[idxs_users[0]])
    for idx in range(len(idxs_users)):
        fed_avg_freqs = datanumber_client[idxs_users[idx]] / total_data_points

        if islist:
            if idx == 0:
                w_avg = w_avg * fed_avg_freqs
            else:
                w_avg += w[idxs_users[idx]] * fed_avg_freqs
        else:
            if idx == 0:
                for key in w_avg:
                    w_avg[key] = w_avg[key] * fed_avg_freqs
            else:
                for key in w_avg:
                    w_avg[key] += w[idxs_users[idx]][key] * fed_avg_freqs

    return w_avg

def cosine_match_weights(sigma,U,V,idx_uesrs):
    n = max(idx_uesrs)
    threshold = torch.nn.Threshold(0.6, 0, inplace=False)
    lowr = [torch.zeros_like(sigma[0])] * (n+1)
    for i in idx_uesrs:
        lowr[i] = torch.matmul(U[i],V[i])
    glo_sigma = [torch.zeros_like(sigma[0])]*(n+1)
    score = torch.zeros([n+1 ,n+1])
    cos_sim = torch.nn.CosineSimilarity()
    for i in idx_uesrs:
        for j in idx_uesrs:
            score[i,j] = cos_sim(torch.flatten(lowr[i],1),torch.flatten(lowr[j],1) )
            score = threshold(score)
    score = f.softmax(score,dim=1)
    print(score)
    for i in idx_uesrs:
        for j in idx_uesrs:
            glo_sigma[i] += sigma[j] * score[i, j]
    return glo_sigma

def cluster_weights(w, datanumber):
    propmt_cluster = []
    for i in range(len(w)):
        prompt = w[i]['prompt_learner.ctx'].flatten(0).cpu()
        propmt_cluster.append(prompt.numpy())

    cluster_model = AgglomerativeClustering(n_clusters=3, linkage="average", affinity="cosine")
    cluster_model = cluster_model.fit(propmt_cluster)
    cluster_results = cluster_model.labels_
    cluster_number = max(cluster_results) + 1
    cluster_group = [[] for i in range(cluster_number)]
    w_cluster = {cluster_i: None for cluster_i in range(cluster_number)}
    w_temp = copy.deepcopy(w[0])

    for idx in range(len(cluster_results)):
        cluster_group[cluster_results[idx]].append(idx)

    for num in range(cluster_number):
        client_list = cluster_group[num]
        total_data_points = sum([datanumber[r] for r in client_list])
        fed_avg_freqs = [datanumber[r] / total_data_points for r in client_list]
        for idx in range(len(client_list)):
            if idx == 0:
                prompt_avg = w[client_list[idx]]['prompt_learner.ctx'] * fed_avg_freqs[idx]
            else:
                prompt_avg += w[client_list[idx]]['prompt_learner.ctx'] * fed_avg_freqs[idx]
        w_temp['prompt_learner.ctx'] = prompt_avg
        w_cluster[num] = w_temp

    return w_cluster, cluster_group

def count_parameters(model, model_name):
    """Count trainable parameters for a given module."""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if model_name in name:
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# === Utility functions migrated from server.py ===

def generate_performance_table(metrics: Dict[str, Dict[str, List[float]]],
                             cfg: Any, is_special_dataset: bool,
                             best_round: int, best_base_round: int,
                             is_test: bool = False) -> None:
    """Generate and print performance metrics table."""
    if is_special_dataset:
        data_type = "local"
    else:
        data_type = cfg.DATASET.SUBSAMPLE_CLASSES

    if is_test:
        print("\n------------Test Performance Metrics------------")
        table = PrettyTable()

        headers = ["Client ID"]
        if not is_special_dataset and data_type != 'new':
            headers.append("Local Acc")
        headers.append(f"{data_type} Acc")
        table.field_names = headers

        local_accs = []
        data_type_accs = []

        for client_id in range(cfg.DATASET.USERS):
            row = [client_id]

            if not is_special_dataset and data_type != 'new':
                local_acc = metrics['local']['client_acc'][client_id][0]
                row.append(f"{local_acc:.4f}")
                local_accs.append(local_acc)

            data_type_acc = metrics[data_type]['client_acc'][client_id][0]
            row.append(f"{data_type_acc:.4f}")
            data_type_accs.append(data_type_acc)

            table.add_row(row)

        row = ["Average"]
        if not is_special_dataset and data_type != 'new':
            row.append(f"{np.mean(local_accs):.4f}")
        row.append(f"{np.mean(data_type_accs):.4f}")
        table.add_row(row)

        print(table)
    else:
        print("\n------------Performance Metrics------------")

        local_table = PrettyTable()
        local_headers = ["Round"]
        for client_id in range(cfg.DATASET.USERS):
            local_headers.append(f"Client {client_id}")
        local_headers.append("Global Avg")
        local_table.field_names = local_headers

        if not is_special_dataset and data_type != "local":
            data_type_table = PrettyTable()
            data_type_headers = ["Round"]
            for client_id in range(cfg.DATASET.USERS):
                data_type_headers.append(f"Client {client_id}")
            data_type_headers.append("Global Avg")
            data_type_table.field_names = data_type_headers

        for round_idx in range(cfg.OPTIM.ROUND):
            local_row = [round_idx + 1]
            for client_id in range(cfg.DATASET.USERS):
                local_row.append(f"{metrics['local']['client_acc'][client_id][round_idx]:.4f}")
            local_row.append(f"{metrics['local']['acc'][round_idx]:.4f}")
            local_table.add_row(local_row)

            if not is_special_dataset and data_type != "local":
                data_type_row = [round_idx + 1]
                for client_id in range(cfg.DATASET.USERS):
                    data_type_row.append(f"{metrics[data_type]['client_acc'][client_id][round_idx]:.4f}")
                data_type_row.append(f"{metrics[data_type]['acc'][round_idx]:.4f}")
                data_type_table.add_row(data_type_row)

        if cfg.OPTIM.ROUND >= 5:
            local_row = ["Last 5 Avg"]
            for client_id in range(cfg.DATASET.USERS):
                local_row.append(f"{np.mean(metrics['local']['client_acc'][client_id][-5:]):.4f}")
            local_row.append(f"{np.mean(metrics['local']['acc'][-5:]):.4f}")
            local_table.add_row(local_row)

            if not is_special_dataset and data_type != "local":
                data_type_row = ["Last 5 Avg"]
                for client_id in range(cfg.DATASET.USERS):
                    data_type_row.append(f"{np.mean(metrics[data_type]['client_acc'][client_id][-5:]):.4f}")
                data_type_row.append(f"{np.mean(metrics[data_type]['acc'][-5:]):.4f}")
                data_type_table.add_row(data_type_row)

        if best_round >= 0:
            local_row = [f"Best ({best_round+1})"]
            for client_id in range(cfg.DATASET.USERS):
                local_row.append(f"{metrics['local']['client_acc'][client_id][best_round]:.4f}")
            local_row.append(f"{metrics['local']['acc'][best_round]:.4f}")
            local_table.add_row(local_row)

        if not is_special_dataset and best_base_round >= 0 and data_type != "local":
            data_type_row = [f"Best ({best_base_round+1})"]
            for client_id in range(cfg.DATASET.USERS):
                data_type_row.append(f"{metrics[data_type]['client_acc'][client_id][best_base_round]:.4f}")
            data_type_row.append(f"{metrics[data_type]['acc'][best_base_round]:.4f}")
            data_type_table.add_row(data_type_row)

        print(f"\nLocal Accuracy Table:")
        print(local_table)
        if not is_special_dataset and data_type != "local":
            print(f"\n{data_type} Accuracy Table:")
            print(data_type_table)

def setup_federated_environment(cfg: Any, args: Any) -> None:
    """Setup federated learning environment (random seed, logging)."""
    from Dassl.dassl.utils import setup_logger, set_random_seed

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    if cfg.DATASET.USEALL == True:
        setup_logger(os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.SUBSAMPLE_CLASSES))
    else:
        setup_logger(cfg.OUTPUT_DIR)

def initialize_logs(model_save_path: str, model_name: str, args: Any) -> None:
    """Initialize log files."""
    log_dir = os.path.join(model_save_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_files = [
        f"{model_name}_text_prompt_pairs.log",
        f"{model_name}_vision_prompt_pairs.log",
        f"{model_name}_text_aggregate.log",
        f"{model_name}_vision_aggregate.log"
    ]

    for log_file in log_files:
        with open(os.path.join(log_dir, log_file), "w") as f:
            f.write(f"============ {model_name} Training Log ============\n")
            f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Config: {args}\n\n")

def print_domain_distribution(client_ids: List[int], cfg: Any) -> None:
    """Print domain distribution of selected clients."""
    if not hasattr(cfg.DATASET, 'NAME') or not cfg.DATASET.SPLIT_CLIENT:
        return

    domains = {
        "DomainNet": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
        "Office": ["amazon", "caltech", "dslr", "webcam"]
    }

    dataset_name = cfg.DATASET.NAME
    if dataset_name not in domains:
        return

    num_domains = len(domains[dataset_name])
    num_clients_per_domain = cfg.DATASET.USERS // num_domains

    domain_distribution = {}
    for client_id in client_ids:
        domain_idx = client_id // num_clients_per_domain
        if domain_idx < num_domains:
            domain_name = domains[dataset_name][domain_idx]
            domain_distribution[domain_name] = domain_distribution.get(domain_name, 0) + 1

    print(f"Domain distribution of selected clients: {domain_distribution}")

def split_ivlp_params(state_dict: Dict[str, torch.Tensor]) -> tuple:
    """Split IVLP model parameters into vision and language parts."""
    vision_params = {}
    language_params = {}

    for name, param in state_dict.items():
        if 'visual' in name or 'image_encoder' in name or ('VPT' in name and not 'text_encoder' in name):
            vision_params[name] = copy.deepcopy(param)
        elif 'prompt_learner.ctx' in name or ('VPT' in name and 'text_encoder' in name):
            language_params[name] = copy.deepcopy(param)

    return vision_params, language_params

def validate_client_weights(client_weights: Dict[str, Any], num_clients: int) -> bool:
    """Validate completeness of client weights."""
    required_keys = ['global', 'local', 'components']

    for key in required_keys:
        if key not in client_weights:
            print(f"Error: Missing required key '{key}'")
            return False

    if len(client_weights['local']) != num_clients:
        print(f"Error: local weights count mismatch, expected {num_clients}, got {len(client_weights['local'])}")
        return False

    return True

def log_training_progress(round_idx: int, num_rounds: int, selected_clients: List[int],
                         avg_acc: float) -> None:
    """Log training progress."""
    print(f"Round {round_idx+1}/{num_rounds} completed")
    print(f"Participating clients: {selected_clients}")
    print(f"Average accuracy: {avg_acc:.4f}")
    print("-" * 50)
