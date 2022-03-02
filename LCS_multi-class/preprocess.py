import os.path
from pathlib import Path
import scipy.sparse as sp
import numpy as np
from utils import *


def test_file_processed(dataset):
    flag = True

    file = Path('./data/{}/processed/features.npy'.format(dataset))
    flag = flag and file.exists()

    file = Path('./data/{}/processed/labels.npy'.format(dataset))
    flag = flag and file.exists()

    file = Path('./data/{}/processed/adj_laplacian_AI.npz'.format(dataset))
    flag = flag and file.exists()

    file = Path('./data/{}/processed/adj_laplacian.npz'.format(dataset))
    flag = flag and file.exists()

    file = Path('./data/{}/processed/split.npz'.format(dataset))
    flag = flag and file.exists()

    return flag


def test_file_raw(dataset):
    flag = True

    file = Path('./data/{}/raw/role.json'.format(dataset))
    flag = flag and file.exists()

    file = Path('./data/{}/raw/feats.npy'.format(dataset))
    flag = flag and file.exists()

    file = Path('./data/{}/raw/class_map.json'.format(dataset))
    flag = flag and file.exists()

    file = Path('./data/{}/raw/adj_full.npz'.format(dataset))
    flag = flag and file.exists()

    return flag


def process(dataset):
    file = Path('./data/{}/processed'.format(dataset))
    if not file.exists():
        file.mkdir()

    if test_file_processed(dataset):
        return True
    file = Path('./data/{}/raw'.format(dataset))
    if not file.exists() or not test_file_raw(dataset):
        print('Please download the dataset')
        return False

    adj_full = sp.load_npz('./data/{}/raw/adj_full.npz'.format(dataset)).astype(np.bool)
    adj_laplacian = row_normalize(adj_full)
    adj_laplacian = adj_laplacian.tocsr()

    adj_full = adj_full.tolil()
    for i in range(adj_full.shape[0]):
        adj_full[i, i] = True
    adj_full = adj_full.tocsr()
    adj_laplacian_AI = row_normalize(adj_full)
    adj_laplacian_AI = adj_laplacian_AI.tocsr()

    sp.save_npz('./data/{}/processed/adj_laplacian_AI.npz'.format(dataset), adj_laplacian_AI)
    sp.save_npz('./data/{}/processed/adj_laplacian.npz'.format(dataset), adj_laplacian)

    features = np.load('./data/{}/raw/feats.npy'.format(dataset))
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)

    np.save('./data/{}/processed/features.npy'.format(dataset), features)

    role = json.load(open('./data/{}/raw/role.json'.format(dataset)))
    np.savez('./data/{}/processed/split.npz'.format(dataset), tr=np.array(role['tr']),
             va=np.array(role['va']), te=np.array(role['te']))

    class_map = json.load(open('./data/{}/raw/class_map.json'.format(dataset)))
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

    np.save('./data/{}/processed/labels.npy'.format(dataset), labels.astype(np.int))

    return True




def get_queue(sampler,dataset):
    adj_lap, labels, feats, train_nodes, val_nodes, test_nodes = load_data(dataset)
    train_nodes_shuffle = np.random.permutation(train_nodes)
    if len(train_nodes_shuffle) % 2048 < 1024:
        nums = len(train_nodes_shuffle) // 2048 - 1
    else:
        nums = len(train_nodes_shuffle) // 2048

    q = deque()
    for index in range(nums):
        print(index, end=' ')
        select_train_nodes = train_nodes_shuffle[index * 2048:(index + 1) * 2048]
        adjs, input_nodes = sampler.target_nodes_batch1(adj_lap,select_train_nodes)
        q.appendleft((adjs, input_nodes, select_train_nodes))
    select_train_nodes = train_nodes_shuffle[2048 * nums:len(train_nodes_shuffle)]
    adjs, input_nodes = sampler.target_nodes_batch1(adj_lap,select_train_nodes)
    q.appendleft((adjs, input_nodes, select_train_nodes))
    return q















if __name__ == '__main__':
    print(process('reddit'))