import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import scipy.io as sio
import networkx as nx
from collections import defaultdict
import torch.nn as nn
import torch
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import copy
from sklearn.preprocessing import StandardScaler
from networkx.readwrite import json_graph
import json
import pandas as pd
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.metrics import f1_score,accuracy_score
from collections import defaultdict
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import networkx as nx
import time
import sys
import os
import operator
import math
from random import randint
import random
"""
Load data function adopted from https://github.com/williamleif/GraphSAGE
"""
WALK_LEN = 5
N_WALKS = 50




def get_batch_size(scale,batch_size):
    batch_size = int(scale*batch_size)
    base = int(math.log(batch_size,2))
    batch_size = max(32,2**(base+1))
    return batch_size








def load_data_graphsage(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        def conversion(n): return int(n)
    else:
        def conversion(n): return n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        def lab_conversion(n): return n
    else:
        def lab_conversion(n): return int(n)

    class_map = {conversion(k): lab_conversion(v)
                 for k, v in class_map.items()}

    # Remove all nodes that do not have val/test annotations
    # (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(
        broken_count))

    # Make sure the graph has edge train_removed annotations
    # (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[str(n)] for n in G.nodes(
        ) if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map


def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node, curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs


"""
Load data function adopted from https://github.com/tkipf/gcn
"""


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data_gcn(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/gcn/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/gcn/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    G = nx.from_dict_of_lists(graph)

    edges = []
    for s in G:
        for t in G[s]:
            if s!=t:
                edges += [[s, t]]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = labels.argmax(axis=1)  # pytorch require target 1d

    idx_test = test_idx_range.tolist()
    idx_train = range(len(ally)-500)
    idx_val = range(len(ally)-500, len(ally))

    edges = np.array(edges)
    adj_matrix = get_adj(edges, features.shape[0])
    
    return adj_matrix, np.array(labels), features.toarray(), \
        np.array(idx_train), np.array(idx_val), np.array(idx_test)


def preprocess_data(dataset):
    if dataset in ['ppi', 'ppi-large', 'reddit', 'flickr', 'yelp']:
        prefix = './data/{}/{}'.format(dataset, dataset)
        G, feats, id_map, walks, class_map = load_data_graphsage(prefix)

        degrees = np.zeros(len(G), dtype=np.int64)
        edges = []
        labels = []
        idx_train = []
        idx_val = []
        idx_test = []
        for s in G:
            if G.nodes[s]['test']:
                idx_test += [s]
            elif G.nodes[s]['val']:
                idx_val += [s]
            else:
                idx_train += [s]
            for t in G[s]:
                if s!=t:
                    edges += [[s, t]]
            degrees[s] = len(G[s])
            labels += [class_map[str(s)]]

        edges = np.array(edges)
        adj_matrix = get_adj(edges, feats.shape[0])


        return adj_matrix, np.array(labels), np.array(feats), \
            np.array(idx_train), np.array(idx_val), np.array(idx_test)


    elif dataset in ['cora', 'citeseer', 'pubmed']:
        # dataset=='cora' or dataset=='citeseer' or dataset=='pubmed':
        return load_data_gcn(dataset)






def load_data(prefix, AI = False, mode = False):

    if AI:
        adj_laplacian = sp.load_npz('./data/{}/processed/adj_laplacian_AI.npz'.format(prefix))
    else:
        adj_laplacian = sp.load_npz('./data/{}/processed/adj_laplacian.npz'.format(prefix))

    labels = np.load('./data/{}/processed/labels.npy'.format(prefix))

    if mode:
        features  = np.load('./data/{}/processed/features.npy'.format(prefix),mmap_mode= 'r')
    else:
        features = np.load('./data/{}/processed/features.npy'.format(prefix))

    role = np.load('./data/{}/processed/split.npz'.format(prefix))
    return adj_laplacian ,labels, features, role['tr'],role['va'],role['te']




def process_graph_data(adj_full, adj_train, feats, class_map, role):
    """
    setup vertex property map for output classes, train/val/test masks, and feats
    """
    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0],list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k,v in class_map.items():
            class_arr[k][v-offset] = 1
    return class_arr


def normalize(adj):
    """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
    rowsum = np.array(adj.sum(1)) + 1e-20
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj

def normalize_adj_diag_enhance(adj, diag_lambda=1):
    """Normalization by  A'=(D+I)^{-1}(A+I), A'=A'+lambda*diag(A')"""
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = 1.0 / (rowsum + 1e-20)
    d_mat_inv = sp.diags(d_inv, 0)
    adj = d_mat_inv.dot(adj)
    adj = adj + diag_lambda * sp.diags(adj.diagonal(), 0)
    return adj


def row_normalize(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = 1.0 / (np.maximum(1.0, rowsum))
    d_mat_inv = sp.diags(d_inv, 0)
    adj = d_mat_inv.dot(adj)
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 and len(sparse_mx.col) == 0:
        indices = torch.LongTensor([[], []])
    else:
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return indices, values, shape


def get_adj(edges, num_nodes):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    assert(adj.diagonal().sum()==0)
    assert((adj!=adj.T).nnz==0 )
    return adj







def get_laplacian(adj):
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(adj)

"""
To make plot smoother like tensorboard
"""
def smooth(scalars, weight=0.2):  # Weight between 0 and 1
    if len(scalars)==0:
        return scalars
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed






































if __name__ == '__main__':

    adj_full = sp.load_npz('./data/reddit/laplacian_matrix.npz').astype(np.bool)

    class_map = json.load(open('./data/reddit/class_map.json'))
    class_map = {int(k): v for k, v in class_map.items()}

    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k, v in class_map.items():
            class_arr[k] = v
        labels = class_arr
    else:
        labels = np.zeros(num_vertices)
        for i, v in class_map.items():
            labels[i] = v

    print(type(labels))

    np.save('./data/reddit/labels.npy',labels)






