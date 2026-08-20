import numpy as np
import scipy.sparse as sp
import torch
import torch_scatter
from torch_sparse import SparseTensor


def node_prototype(x, y, n_classes=None):
    x_clu = torch_scatter.scatter(src=x, index=y, dim=0, dim_size=n_classes, reduce='mean')
    return x_clu, torch.arange(n_classes).to(y.device)


def edge_index_to_sparse_tensor(edge_index: torch.Tensor, edge_weight: torch.Tensor = None, n_nodes=None):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1)).to(edge_index.device)
    if n_nodes is None:
        n_nodes = int(edge_index.max() + 1)
    if isinstance(n_nodes, (tuple, list)):
        n_src_nodes, n_dst_nodes = n_nodes
    else:
        n_src_nodes = n_dst_nodes = n_nodes
    # Transpose the matrix for directed graphs
    return SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=(n_src_nodes, n_dst_nodes)).t()


def to_mask(*indices, num=None):
    if num is None:
        num = int(max(indices) + 1)
    masks = []
    for idx in indices:
        mask = torch.zeros(num, dtype=torch.bool)
        mask[idx] = 1
        masks.append(mask)
    return masks


def reshape_mx(mx, shp):
    """reshape sparse matrix, array-like, tensor."""
    if sp.issparse(mx):
        mx = mx.tocoo()
        return sp.csr_matrix((mx.data, (mx.row, mx.col)), shape=shp).tolil()
    elif isinstance(mx, np.ndarray):
        oshp = mx.shape
        mx = np.concatenate((mx, np.zeros((shp[0] - oshp[0], oshp[1]))), axis=0)
        return np.concatenate((mx, np.zeros((shp[0], shp[1] - oshp[1]))), axis=1)
    elif isinstance(mx, torch.Tensor):
        layout = mx.layout
        device = mx.device
        if layout == torch.strided:
            oshp = mx.shape
            dv = mx.device
            mx = torch.concat((mx, torch.zeros((shp[0] - oshp[0], oshp[1])).to(dv)), dim=0)
            return torch.concat((mx, torch.zeros((shp[0], shp[1] - oshp[1])).to(dv)), dim=1)
        elif layout == torch.sparse_coo:
            values = mx._values()
            indices = mx._indices()
        else:
            indices = mx.nonzero().t()
            values = mx[indices[0], indices[1]]
        mx = torch.sparse.FloatTensor(indices, values, shp).to(device)
        if layout != torch.sparse_coo:
            mx = mx.to_dense()
        return mx
    else:
        raise NotImplementedError


def save_checkpoint(model, filepath):
    torch.save(model.state_dict(), filepath)


def load_checkpoint(model, filepath):
    model.load_state_dict(torch.load(filepath))
