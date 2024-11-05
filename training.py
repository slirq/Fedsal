import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

def run_selftrain_GC(clients, server, local_epoch):
    # all clients are initialized with the same weights
    for client in clients:
        client.download_from_server(server)

    allAccs = {}
    for client in clients:
        client.local_train(local_epoch)

        loss, acc = client.evaluate()
        allAccs[client.name] = [client.train_stats['trainingAccs'][-1], client.train_stats['valAccs'][-1], acc]
        print("  > {} done.".format(client.name))

    return allAccs

def run_fedavg(clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    metrics = {
        "communication_time": [],
        "convergence_speed": [],
        "overhead": []
    }
    
    start_time = time.time()

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        round_start_time = time.time()

        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            client.local_train(local_epoch)

        server.aggregate_weights(selected_clients)

        for client in selected_clients:
            client.download_from_server(server)

        acc_clients = [client.evaluate()[1] for client in clients]
        metrics["convergence_speed"].append(np.mean(acc_clients))

        round_end_time = time.time()
        metrics["communication_time"].append(round_end_time - round_start_time)
        metrics["overhead"].append(time.time() - start_time)

    frame = pd.DataFrame()
    avg_acc = []
    for client in clients:
        loss, acc = client.evaluate()
        avg_acc.append(acc)
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]
    temp_metrics = metrics
    fs = frame.style.apply(highlight_max).data
    metrics["communication_time"] = np.mean(metrics["communication_time"])
    metrics["overhead"] = np.mean(metrics["overhead"])
    metrics_frame = pd.DataFrame(metrics)
    return frame, metrics_frame,temp_metrics

def run_fedsal(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, EPS_1, EPS_2):
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    metrics = {
        "communication_time": [],
        "convergence_speed": [],
        "overhead": [],
        "saliency_norm": [],
        "num_clients_per_cluster": []
    }

    start_time = time.time()

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        round_start_time = time.time()
        if c_round % 50 == 0:
            print(f"  > round {c_round}")
        
        if c_round == 1:
            for client in clients:
                client.download_from_server_se(args, server)
        
        participating_clients = server.randomSample_clients(clients, frac=1.0)
        
        for client in participating_clients:
            client.compute_weight_update(local_epoch)
            client.compute_saliency_maps()
            client.reset()
            
        similarities = server.compute_saliency_pairwise_similarities(clients)
        cluster_indices_new = []

        max_size = get_max_size([client.saliency_maps for client in clients])
        for idc in cluster_indices:
            client_group = [clients[i] for i in idc]
            max_norm,mean_norm = server.compute_max_mean_saliency_norm(client_group, max_size)
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20:
                server.cache_model(idc, client_group[0].W, acc_clients)
                c1, c2 = server.min_cut(similarities[idc][:, idc], idc)
                cluster_indices_new += [c1, c2]
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.evaluate()[1] for client in clients]
        round_end_time = time.time()
        metrics["convergence_speed"].append(np.mean(acc_clients))
        metrics["communication_time"].append(round_end_time - round_start_time)
        metrics["overhead"].append(round_end_time - start_time)
        metrics['saliency_norm'].append(mean_norm)
        num_clients = [len(idc) for idc in cluster_indices]
        metrics["num_clients_per_cluster"].append(num_clients)
    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(results, columns=["FL Model"] + [f"Model {i}" for i in range(results.shape[1] - 1)], index=[client.name for client in clients])
    frame = frame.max(axis=1).to_frame(name='test_acc')
    
    metrics_frame = pd.DataFrame(metrics)
    return frame, metrics_frame, metrics

def get_max_size(saliency_maps_list):
    max_size = 0
    for saliency_maps in saliency_maps_list:
        size = sum(param.numel() for param in saliency_maps.values())
        if size > max_size:
            max_size = size
    return max_size