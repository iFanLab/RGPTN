from copy import deepcopy
import numpy as np
from numpy.testing import assert_array_almost_equal
import scipy.io as sio
import scipy.sparse as sp
import torch
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import os.path as osp

from utils import to_mask, reshape_mx


def load_data(dataset_str: str, root='./data/', noise_type='uniform', noise_rate=0, seed=None):
    dataset_str = dataset_str.lower()

    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        dataset = load_pyg_dataset(dataset_str, root=root)
        data = dataset[0]
        data.gtype = 'homo'
        num_classes = dataset.num_classes

    elif dataset_str in ['cora_cc', 'citeseer_cc', 'cora_ca']:
        dsname = {
            'cora_cc': 'cocitation_cora',
            'citeseer_cc': 'cocitation_citeseer',
            'cora_ca': 'coauthorship_cora',
        }[dataset_str]
        adj, features, labels = load_npz(osp.join(root, f"{dsname}.npz"))
        features = torch.from_numpy(features)
        adj = adj.tocsr()
        labels = torch.from_numpy(labels).long()
        num_classes = labels.max().item() + 1
        if True:
            n_nodes = features.shape[0]
            subset = remove_isolated_nodes(adj)
            features = features[subset]
            labels = labels[subset]
            adj = adj.tocoo()
            adj = subhypergraph_by_nodes(subset, adj.row, adj.col, n_nodes=n_nodes, relabel_nodes=True)
        data = Data(x=features, incident_matrix=adj, y=labels)
        data.gtype = 'hyper'

    elif dataset_str == 'acm':
        data = load_acm(root)
        data.gtype = 'hetero'
        tgt_ntype = data.predict_ntype
        labels = data.y[tgt_ntype]
        num_classes = labels.max().item() + 1

    else:
        raise ValueError

    if dataset_str not in ['cora', 'citeseer', 'pubmed']:
        # 1/1/8 split
        num_samples = labels.shape[0]
        num_classes = len(np.unique(labels))
        train_per_class = max(1, int(num_samples * 0.1 / num_classes))
        val_per_class = max(1, int(num_samples * 0.1 / num_classes))

        idx_train, idx_val, idx_test = train_val_test_split_class_balanced(labels, train_per_class, val_per_class, random_state=seed)
        train_mask, val_mask, test_mask = to_mask(idx_train, idx_val, idx_test, num=labels.shape[0])

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

    if dataset_str in ['acm']:
        data.y[data.predict_ntype] = generate_label_noise(data.y[data.predict_ntype], noise_type, noise_rate, num_classes, data.train_mask, data.val_mask, seed=seed)
    else:
        data.y = generate_label_noise(data.y, noise_type, noise_rate, num_classes, data.train_mask, data.val_mask, seed=seed)

    data_fmt_str = f"{str.lower(dataset_str)}_{noise_type}_{int(noise_rate*100)}"
    return data, data_fmt_str


def load_pyg_dataset(dataset_str: str, root='./data/'):
    dataset_str = dataset_str.lower()
    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root, name=dataset_str)
    else:
        raise ValueError
    return dataset


def load_npz(dataset_str):
    """Load a SparseGraph from a Numpy binary file."""
    filepath = dataset_str if '/' in dataset_str else f'data/{dataset_str}.npz'
    with np.load(filepath, allow_pickle=True) as loader:
        loader = dict(loader)

        adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']), shape=loader['adj_shape'], dtype=np.float32)

        if 'attr_data' in loader:
            features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']), shape=loader['attr_shape']).astype(np.float32).toarray()
        elif 'features' in loader:
            features = np.array(loader['features']).astype(np.float32)
        else:
            features = None

        labels = loader['labels']

    return adj, features, labels


def load_acm(root, with_p2p=False):
    data = sio.loadmat(osp.join(root, 'acm.mat'))
    p_vs_s = data['PvsL']   # (12499, 73),    paper-subject/field
    p_vs_a = data['PvsA']   # (12499, 17431), paper-author
    p_vs_t = data['PvsT']   # (12499, 1903),  paper-term, bag of words
    p_vs_c = data['PvsC']   # (12499, 14),    paper-conference, labels come from that
    p_vs_p = data['PvsP']   # (12499, 12499), paper-paper

    # Refer to DGL, we sample 5 conferences to assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MobiCOMM papers as class 2 (communication)
    conf_names = ['KDD', 'SIGMOD', 'VLDB', 'SIGCOMM', 'MobiCOMM']
    conf_ids = [i for i, conf_data in enumerate(data['C']) if conf_data[0][0] in conf_names]

    conf_label_ids = np.array([0, 1, 2, 2, 1], dtype=np.int64)

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_s = p_vs_s[p_selected]         # (4025, 73)
    p_vs_a = p_vs_a[p_selected]         # (4025, 17431)
    p_vs_t = p_vs_t[p_selected]         # (4025, 1903)
    p_vs_c = p_vs_c_filter[p_selected]  # (4025, 5),     paper-conf_idx

    conf_idx = p_vs_c.argmax(1).A1      # (4025, )
    labels_conf = conf_label_ids[conf_idx]

    predict_ntype = 'paper'
    n_node_dict = {
        'paper': 4025,
        'author': 17351,
        'subject': 72,
    }

    p_vs_s = reshape_mx(p_vs_s, (n_node_dict['paper'], n_node_dict['subject']))
    p_vs_a = reshape_mx(p_vs_a, (n_node_dict['paper'], n_node_dict['author']))
    p_vs_t = reshape_mx(p_vs_t, (n_node_dict['paper'], p_vs_t.shape[1]))
    p_vs_p = p_vs_p[p_selected, :][:, p_selected]  # (4025, 4025)

    x_dict = {
        'paper': torch.from_numpy(p_vs_t.toarray()).float()
    }

    adj_dict = {
        ('paper', 'pa', 'author'): p_vs_a,
        ('author', 'ap', 'paper'): p_vs_a.transpose(),
        ('paper', 'ps', 'subject'): p_vs_s,
        ('subject', 'sp', 'paper'): p_vs_s.transpose(),
    }
    if with_p2p:
        adj_dict[('paper', 'pp', 'paper')] = p_vs_p

    y_dict = {
        'paper': torch.from_numpy(labels_conf).long()
    }

    data = Data(x=x_dict, adj=adj_dict, y=y_dict, n_node_dict=n_node_dict, predict_ntype=predict_ntype)
    return data


def remove_isolated_nodes(adj: sp.spmatrix):
    """
    Args:
    adj: shape (N_v, N_he)
    """
    d_v = adj.sum(1).A1
    nodes_to_keep = np.where(d_v != 0)[0]
    return nodes_to_keep.astype(np.int32)


def subhypergraph_by_nodes(subset, row, col, n_nodes, n_hyperedges=None, data=None, relabel_nodes=False, return_edge_mask=False):
    if n_hyperedges is None:
        n_hyperedges = int(col.max() + 1)
    edge_index = np.stack((row, col), axis=0)

    if isinstance(subset, (list, tuple)):
        subset = np.array(subset, dtype=np.int64)

    if subset.dtype != bool:
        sorted_subset = subset
        subset = to_mask(subset, num=n_nodes)[0].cpu().numpy()
    else:
        sorted_subset = np.nonzero(subset)[0]

    node_mask = subset
    edge_mask = node_mask[row]  # Filter by node indices
    edge_index = edge_index[:, edge_mask]
    data = data[edge_mask] if data is not None else None

    if relabel_nodes:
        node_idx = np.zeros(node_mask.shape[0], dtype=np.int64)
        n_nodes = int(subset.sum())
        node_idx[sorted_subset] = np.arange(n_nodes)
        edge_index[0] = node_idx[edge_index[0]]  # Relabel node indices

    if data is None:
        adj = sp.csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(n_nodes, n_hyperedges))
    else:
        adj = sp.csr_matrix((data, (edge_index[0], edge_index[1])), shape=(n_nodes, n_hyperedges))

    if return_edge_mask:
        return adj, edge_mask
    else:
        return adj


def train_val_test_split_class_balanced(labels, train_sample_per_class=20, val_sample_per_class=30, random_state=None):
    """
    Randomly sample 20/30 instances for each class as training/validation data, and the rest instances as test data.
    """
    idx = np.arange(len(labels))
    n_classes = labels.max() + 1
    idx_train = []
    idx_val = []
    idx_test = []
    rs = np.random.RandomState(seed=random_state)
    for i in range(n_classes):
        labels_i = idx[labels == i]
        labels_i = rs.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[: train_sample_per_class])).astype(np.int32)
        idx_val = np.hstack((idx_val, labels_i[train_sample_per_class: train_sample_per_class + val_sample_per_class])).astype(np.int32)
        idx_test = np.hstack((idx_test, labels_i[train_sample_per_class + val_sample_per_class:])).astype(np.int32)
    return idx_train, idx_val, idx_test


def generate_label_noise(y, noise_type, noise_rate, n_classes, train_mask, val_mask,
                         iid_noise=True, clean_val_set=False, seed=None):
    labeled_mask = (train_mask | val_mask)

    if noise_type == 'uniform':
        P = build_uniform_P(n_classes, noise_rate=noise_rate)
    elif noise_type == 'pair':
        P = build_pair_P(n_classes, noise_rate)
    if iid_noise:
        noisy_y_train = multiclass_noisify(y[train_mask], P, seed)
        noisy_y_val = multiclass_noisify(y[val_mask], P, seed)
    else:
        noisy_y_labeled = multiclass_noisify(y[labeled_mask], P, seed)

    if not iid_noise:
        noisy_y_train = noisy_y_labeled[train_mask[labeled_mask]]
        noisy_y_val = noisy_y_labeled[val_mask[labeled_mask]]

    noisy_y = y.clone()
    noisy_y[train_mask] = noisy_y_train
    if not clean_val_set:
        noisy_y[val_mask] = noisy_y_val

    return noisy_y


def build_uniform_P(n_classes, noise_rate=0.5):
    """
    The noise transition matrix flips any class to any other with probability ``noise_rate / (#class - 1)``.

    Returns
    -------
    Noise transition matrix : np.array, (N_classes, N_classes)
    """
    assert (noise_rate >= 0.) and (noise_rate <= 1.)
    P = noise_rate / (n_classes - 1) * np.ones((n_classes, n_classes))
    np.fill_diagonal(P, 1.0 - noise_rate)
    return P


def build_pair_P(n_classes, noise_rate=0.5):
    """
    The noise transition matrix flips any class to another with probability ``noise_rate``.

    Returns
    -------
    Noise transition matrix : np.array, (N_classes, N_classes)
    """
    assert (noise_rate >= 0.) and (noise_rate <= 1.)
    P = (1.0 - noise_rate) * np.eye(n_classes)
    for i in range(n_classes):
        P[i, i - 1] = noise_rate
    return P


def multiclass_noisify(y_train, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    device = y_train.device
    y_train = y_train.cpu().data.numpy()

    assert P.shape[0] == P.shape[1]
    assert np.max(y_train) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    noisy_y_train = y_train.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(y_train.shape[0]):
        y_ground = y_train[idx]
        noisy_y_train[idx] = flipper.choice(P.shape[0], 1, p=P[y_ground, :])[0]

    return torch.LongTensor(noisy_y_train).to(device)


if __name__ == '__main__':
    dataset = load_pyg_dataset('cora', root='./data/')
    data = dataset[0]
    print(data)
