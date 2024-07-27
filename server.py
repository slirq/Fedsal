import torch
import numpy as np
import random
import networkx as nx
from dtaidistance import dtw


class Server():
    def __init__(self, model, device):
        self.model = model.to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.model_cache = []

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_weights(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()
    
                
    def compute_pairwise_similarities(self, clients):
        client_dWs = []
        for client in clients:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            client_dWs.append(dW)
        return pairwise_angles(client_dWs)

    def compute_saliency_pairwise_similarities(self, clients):
        num_clients = len(clients)
        similarity_matrix = np.zeros((num_clients, num_clients))

        # Compute saliency maps for each client
        saliency_maps_list = [client.saliency_maps for client in clients]
        max_size = get_max_size(saliency_maps_list)
        flattened_maps = [flatten_and_pad_saliency_maps(saliency_maps, max_size) for saliency_maps in saliency_maps_list]

        # Standardize each flattened saliency map
        standardized_maps = [scale_tensor(standardize_tensor(map)) for map in flattened_maps]

        for i in range(num_clients):
            for j in range(i, num_clients):
                similarity = cosine_similarity(standardized_maps[i], standardized_maps[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Since similarity is symmetric

        return similarity_matrix

    def min_cut(self, similarity, idc):
        g = nx.Graph()
        for i in range(len(similarity)):
            for j in range(len(similarity)):
                g.add_edge(i, j, weight=similarity[i][j])
        cut, partition = nx.stoer_wagner(g)
        c1 = np.array([idc[x] for x in partition[0]])
        c2 = np.array([idc[x] for x in partition[1]])
        return c1, c2

    def aggregate_clusterwise(self, client_clusters):
        for cluster in client_clusters:
            targs = []
            sours = []
            total_size = 0
            for client in cluster:
                W = {}
                dW = {}
                for k in self.W.keys():
                    
                    W[k] = client.W[k]
                    dW[k] = client.dW[k]
                targs.append(W)
                sours.append((dW, client.train_size))
                total_size += client.train_size
            # pass train_size, and weighted aggregate
            reduce_add_average(targets=targs, sources=sours, total_size=total_size)
   

    def compute_max_mean_saliency_norm(self, cluster, max_size):
        max_saliency_norm = -np.inf
        cluster_saliencies = []
        for client in cluster:
            saliency_maps = client.saliency_maps
            flattened_saliency = flatten_and_pad_saliency_maps(saliency_maps, max_size)
            standardized_saliency = standardize_tensor(flattened_saliency)
            scaled_saliency = scale_tensor(standardized_saliency)
            saliency_norm = torch.norm(scaled_saliency).item()
            if saliency_norm > max_saliency_norm:
                max_saliency_norm = saliency_norm
        
            cluster_saliencies.append(scaled_saliency)

        mean_saliency = torch.mean(torch.stack(cluster_saliencies), dim=0)
        
        return max_saliency_norm, torch.norm(mean_saliency).item()


    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs,
                              {name: params[name].data.clone() for name in params},
                              [accuracies[i] for i in idcs])]

def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])

def cosine_similarity(vec1, vec2):
    cos = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0)
    return cos.item()


def flatten_saliency_maps(saliency_maps):
    flattened = []
    for key in sorted(saliency_maps.keys()):  # Sorting keys to maintain consistency
        flattened.append(saliency_maps[key].view(-1))
    return torch.cat(flattened)

def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i, j] = torch.true_divide(torch.sum(s1 * s2), max(torch.norm(s1) * torch.norm(s2), 1e-12)) + 1

    return angles.numpy()
def get_max_size(saliency_maps_list):
    max_size = 0
    for saliency_maps in saliency_maps_list:
        size = sum(param.numel() for param in saliency_maps.values())
        if size > max_size:
            max_size = size
    return max_size
def flatten_and_pad_saliency_maps(saliency_maps, target_size):
    flattened = []
    for key in sorted(saliency_maps.keys()):  # Sorting keys to maintain consistency
        flattened.append(saliency_maps[key].view(-1))
    flattened_tensor = torch.cat(flattened)
    return pad_tensor(flattened_tensor, target_size)
def pad_tensor(tensor, target_size):
    pad_size = target_size - tensor.size(0)
    if pad_size > 0:
        padding = torch.zeros(pad_size, device=tensor.device)
        tensor = torch.cat([tensor, padding])
    return tensor
def standardize_tensor(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / (std + 1e-8)  # Adding a small value to avoid division by zero
def scale_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-8)
def reduce_add_average(targets, sources, total_size):
    for target in targets:
        for name in target:
            tmp = torch.div(torch.sum(torch.stack([torch.mul(source[0][name].data, source[1]) for source in sources]), dim=0), total_size).clone()
            target[name].data += tmp