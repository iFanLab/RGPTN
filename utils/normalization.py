import numpy as np
import scipy.sparse as sp
import torch
import torch_sparse
from torch_scatter import scatter_add
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value


def normalize_adj_tensor(adj, alpha=-0.5, norm='both'):
    """Normalize adjacency matrix tensor."""
    adj_norm = adj
    if adj.layout == torch.strided:
        if norm == 'left' or norm == 'both':
            rowsum = adj.abs().sum(1)
            d_inv = rowsum.pow(alpha)
            d_inv[torch.isinf(d_inv)] = 0.
            adj_norm = adj_norm * d_inv.unsqueeze(1)
        if norm == 'right' or norm == 'both':
            colsum = adj.abs().sum(0)
            d_inv = colsum.pow(alpha)
            d_inv[torch.isinf(d_inv)] = 0.
            adj_norm = adj_norm * d_inv.unsqueeze(0)
    else:
        if adj_norm.is_coalesced():
            edge_index = adj_norm.indices()
            edge_weight = adj_norm.values()
        else:
            # Not differentiable
            edge_index = adj_norm._indices()
            edge_weight = adj_norm._values()
        n_u_nodes, n_v_nodes = adj_norm.shape
        rowsum = scatter_add(edge_weight.abs(), edge_index[0], dim=0, dim_size=n_u_nodes)
        colsum = scatter_add(edge_weight.abs(), edge_index[1], dim=0, dim_size=n_v_nodes)
        if norm == 'left' or norm == 'both':
            d_inv = rowsum.pow(alpha)
            d_inv[torch.isinf(d_inv)] = 0.
            edge_weight = edge_weight * d_inv[edge_index[0]]
        if norm == 'right' or norm == 'both':
            d_inv = colsum.pow(alpha)
            d_inv[torch.isinf(d_inv)] = 0.
            edge_weight = edge_weight * d_inv[edge_index[1]]
        adj_norm = torch.sparse.FloatTensor(edge_index, edge_weight, (n_u_nodes, n_v_nodes)).coalesce()
    return adj_norm


def normalize_edge_index(edge_index, edge_weight=None, add_self_loops=False, alpha=-0.5, norm='both', n_nodes=None):
    return pyg_normalize_adj(edge_index, edge_weight, add_self_loops=add_self_loops, norm_type=norm, norm_alpha=alpha, num_nodes=n_nodes)


def pyg_normalize_adj(edge_index, edge_weight=None, num_nodes=None, improved=False,
                      add_self_loops=True, flow="source_to_target", dtype=None,
                      norm_type='both', norm_alpha=-0.5):

    assert (norm_alpha <= 0) and (norm_type in ['both', 'left', 'right'])
    if isinstance(num_nodes, (tuple, list)):
        num_src_nodes, num_dst_nodes = num_nodes
    else:
        num_src_nodes = num_dst_nodes = num_nodes

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        out_deg = torch_sparse.sum(adj_t, dim=1)
        out_deg_norm = out_deg.pow_(norm_alpha)
        out_deg_norm.masked_fill_(out_deg_norm == float('inf'), 0.)
        in_deg = torch_sparse.sum(adj_t, dim=1)
        in_deg_norm = in_deg.pow_(norm_alpha)
        in_deg_norm.masked_fill_(in_deg_norm == float('inf'), 0.)
        if norm_type in ['left', 'both']:
            adj_t = torch_sparse.mul(adj_t, out_deg_norm.view(-1, 1))
        if norm_type in ['right', 'both']:
            adj_t = torch_sparse.mul(adj_t, in_deg_norm.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        out_deg = scatter(value, row, 0, dim_size=num_src_nodes, reduce='sum')
        in_deg = scatter(value, col, 0, dim_size=num_dst_nodes, reduce='sum')
        out_deg_norm = out_deg.pow_(norm_alpha)
        out_deg_norm.masked_fill_(out_deg_norm == float('inf'), 0)
        in_deg_norm = in_deg.pow_(norm_alpha)
        in_deg_norm.masked_fill_(in_deg_norm == float('inf'), 0)
        if norm_type in ['left', 'both']:
            value = out_deg_norm[row] * value
        if norm_type in ['right', 'both']:
            value = value * in_deg_norm[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    # idx = col if flow == 'source_to_target' else row
    # deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    # deg_inv_sqrt = deg.pow_(norm_alpha)
    # deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    # if norm_type in ['left', 'both']:
    #     edge_weight = deg_inv_sqrt[row] * edge_weight
    # if norm_type in ['right', 'both']:
    #     edge_weight = edge_weight * deg_inv_sqrt[col]

    out_deg = scatter(edge_weight, row, dim=0, dim_size=num_src_nodes, reduce='sum')
    in_deg = scatter(edge_weight, col, dim=0, dim_size=num_dst_nodes, reduce='sum')
    out_deg_norm = out_deg.pow_(norm_alpha)
    out_deg_norm.masked_fill_(out_deg_norm == float('inf'), 0)
    in_deg_norm = in_deg.pow_(norm_alpha)
    in_deg_norm.masked_fill_(in_deg_norm == float('inf'), 0)
    if norm_type in ['left', 'both']:
        edge_weight = out_deg_norm[row] * edge_weight
    if norm_type in ['right', 'both']:
        edge_weight = edge_weight * in_deg_norm[col]

    return edge_index, edge_weight


def normalize_features(features):
    """Row-normalize feature matrix"""
    if sp.issparse(features):
        rowsum = np.array(features.sum(1), dtype=np.float32)
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features
    elif isinstance(features, np.ndarray):
        rowsum = features.sum(1).astype(np.float32)
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        features = features * r_inv[:, None]
        return features
    elif isinstance(features, torch.Tensor) and features.layout == torch.strided:
        rowsum = features.sum(1).type(torch.float32)
        r_inv = torch.pow(rowsum, -1)
        r_inv[torch.isinf(r_inv)] = 0.
        features = r_inv.unsqueeze(-1) * features
        return features
    else:
        raise NotImplementedError
