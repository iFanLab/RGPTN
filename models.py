import numpy as np
import scipy.sparse as sp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import spmm, to_undirected, coalesce
from torch_geometric.data import Data

from utils import (
    EarlyStopping,
    node_prototype,
    normalize_edge_index,
    normalize_adj_tensor,
    normalize_features,
    edge_index_to_sparse_tensor,
)


class GraphPTN(nn.Module):
    def __init__(self, in_feats, out_feats, n_hid=64, n_layers=2,
                 dropout=0.5, device='cpu', data=None,
                 input_norm=None, norm=None,
                 train_args={},
                 **kwargs):
        super(GraphPTN, self).__init__()
        self.name = 'GPTN'

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.n_classes = out_feats
        self.n_layers = n_layers
        self.n_hid = n_hid
        self.dropout = dropout
        self.dropout_fn = nn.Dropout(p=dropout)
        self.act_fn = nn.ReLU()
        self.device = device
        self.train_args = train_args

        self.with_input_norm = True
        if input_norm == 'layernorm':
            self.input_norm = nn.LayerNorm(in_feats)
        elif input_norm == 'batchnorm':
            self.input_norm = nn.BatchNorm1d(in_feats)
        else:
            self.with_input_norm = False

        self.with_norm = True
        if norm == 'layernorm':
            norm_cls = nn.LayerNorm
        elif norm == 'batchnorm':
            norm_cls = nn.BatchNorm1d
        else:
            self.with_norm = False

        # Propagation
        self.n_prop_layers = kwargs.get('n_prop_layers', 10)
        self.ppr_alpha = kwargs.get('ppr_alpha', 0.1)
        self.x_norm = kwargs.get('x_norm', None)
        self.x_step_norm = kwargs.get('x_step_norm', None)  # `norm` for each propagation step

        self.intra_adj_norm = kwargs.get('intra_adj_norm', 'both')
        self.intra_adj_norm_alpha = kwargs.get('intra_adj_norm_alpha', -0.5)
        self.inter_adj_norm = kwargs.get('inter_adj_norm', 'both')
        self.inter_adj_norm_alpha = kwargs.get('inter_adj_norm_alpha', 0)
        self.add_inter_self_loops = kwargs.get('add_inter_self_loops', False)

        # Transform
        self.n_tran_layers = kwargs.get('n_tran_layers', 1)
        with_bias = kwargs.get('with_bias', False)

        self.fcs = nn.ModuleList()
        self.norms = nn.ModuleList()
        if self.n_tran_layers == 1:
            self.fcs.append(nn.Linear(in_feats, out_feats, bias=with_bias))
        else:
            self.fcs.append(nn.Linear(in_feats, self.n_hid, bias=with_bias))
            if self.with_norm:
                self.norms.append(norm_cls(self.n_hid, **self.norm_kw))
            for _ in range(self.n_tran_layers - 2):
                self.fcs.append(nn.Linear(self.n_hid, self.n_hid, bias=with_bias))
                if self.with_norm:
                    self.norms.append(norm_cls(self.n_hid, **self.norm_kw))
            self.fcs.append(nn.Linear(self.n_hid, out_feats, bias=with_bias))

        self._prop_cached = False

    def reset_parameters(self):
        for fc in self.fcs:
            nn.init.xavier_uniform_(fc.weight)
        for norm in self.norms:
            norm.reset_parameters()
        self._prop_cached = False

    def forward(self, data):
        x = data.x
        if not self._prop_cached:
            x = self.propagation(x, data.adj)
        return self.encode(x)

    def encode(self, x):
        for i, fc in enumerate(self.fcs[:-1]):
            x = fc(x)
            if self.with_norm:
                x = self.norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.act_fn(x)
        outs = self.fcs[-1](x)
        return outs

    def propagation(self, x, edge_index):
        x = self.propagate_feature(x, edge_index)
        x = self.normalize_feature(x, 'x_norm')
        return x

    def normalize_feature(self, x, norm_attr):
        if getattr(self, norm_attr) is None:
            normalized_x = x
        elif getattr(self, norm_attr) in ['linearize', 'arctan', 'tanh', 'standardize', 'layernorm', 'pyg']:
            normalized_x = normalize_features(x.cpu().data.numpy(), getattr(self, norm_attr), lim_min=0, lim_max=1)
            normalized_x = torch.from_numpy(normalized_x).to(x.device)
        elif getattr(self, norm_attr) == 'l2':
            normalized_x = F.normalize(x, p=2, dim=1)
        else:
            raise NotImplementedError(f"Not support `{getattr(self, norm_attr)}` embedding normalization")
        return normalized_x

    def propagate_feature(self, x: torch.Tensor, adj_norm_t: SparseTensor):
        alpha = self.ppr_alpha
        K = self.n_prop_layers
        cached_x = alpha * x
        for _ in range(K):
            prop_x = spmm(adj_norm_t, x)
            prop_x = self.normalize_feature(prop_x, 'x_step_norm')
            x = (1. - alpha) * prop_x + cached_x

        return x[:self.n_nodes[self.predict_ntype]].to(self.device)

    def get_data(self, data: Data):
        if data.gtype == 'homo':
            x, y = data.x, data.y
            self.n_nodes = {'Node': x.size(0)}
            self.predict_ntype = 'Node'
            row, col = data.edge_index.cpu().data.numpy()
            intra_adj = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(x.size(0), x.size(0)))
            adj_dict = {('Node', 'NN', 'Node'): intra_adj}
        elif data.gtype == 'hyper':
            intra_adj, y = data.incident_matrix, data.y
            self.n_nodes = {'Node': intra_adj.shape[0], 'Hyperedge': intra_adj.shape[1]}
            self.predict_ntype = 'Node'
            x, adj_dict = self._hyperG2foundationG(data)
        else:
            self.n_nodes = data.n_node_dict
            self.predict_ntype = data.predict_ntype
            x, adj_dict = self._heteroG2foundationG(data)
            y = data.y[data.predict_ntype]

        n_all_nodes = x.size(0)
        eidx_dict, ewgt_dict = {}, {}
        for k, v in adj_dict.items():
            eidx = torch.LongTensor(np.stack(v.nonzero(), axis=0))
            if k[0] == k[-1]:
                eidx, ewgt = normalize_edge_index(eidx, alpha=self.intra_adj_norm_alpha, norm=self.intra_adj_norm, n_nodes=n_all_nodes, add_self_loops=True)
            else:
                eidx, ewgt = normalize_edge_index(eidx, alpha=self.intra_adj_norm_alpha, norm=self.intra_adj_norm, n_nodes=n_all_nodes)
            eidx_dict[k] = eidx
            ewgt_dict[k] = ewgt
        eidx, ewgt = coalesce(torch.cat(list(eidx_dict.values()), dim=1), torch.cat(list(ewgt_dict.values()), dim=0), num_nodes=n_all_nodes)
        eidx, ewgt = normalize_edge_index(eidx, ewgt, add_self_loops=self.add_inter_self_loops, alpha=self.inter_adj_norm_alpha, norm=self.inter_adj_norm, n_nodes=n_all_nodes)
        adj_norm_t = edge_index_to_sparse_tensor(eidx, ewgt, n_nodes=n_all_nodes)

        return Data(
            x=x.to(self.device),
            adj=adj_norm_t.to(self.device),
            y=y.to(self.device)
        )

    def _hyperG2foundationG(self, data):
        N_v, N_e = data.incident_matrix.shape
        N = N_v + N_e

        x_e = data.x.new_zeros((N_e, data.x.size(1)))
        x = torch.cat((data.x, x_e), dim=0)

        A = data.incident_matrix.copy().tocoo()
        hyperedge_index = np.stack((A.row, A.col), axis=0)
        hyperedge_index[1] += N_v
        edge_weight = A.data
        intra_adj = sp.csr_matrix((edge_weight, hyperedge_index), shape=(N, N))
        foundation_adj = {
            ('Node', 'NH', 'Hyperedge'): intra_adj,
            ('Hyperedge', 'HN', 'Node'): intra_adj.T,
        }

        return x, foundation_adj

    def _heteroG2foundationG(self, data):
        tgt_ntype = data.predict_ntype
        x_dict = data.x
        adj_dict = data.adj
        n_node_dict = data.n_node_dict
        ntypes = list(n_node_dict.keys())

        predict_ntype_idx = ntypes.index(tgt_ntype)
        sorted_ntypes = [ntypes[predict_ntype_idx]] + ntypes[:predict_ntype_idx] + ntypes[predict_ntype_idx + 1:]
        node_idx_bias = {tgt_ntype: 0}
        for i in range(len(sorted_ntypes) - 1):
            node_idx_bias[sorted_ntypes[i + 1]] = node_idx_bias[sorted_ntypes[i]] + n_node_dict[sorted_ntypes[i]]
        N = node_idx_bias[sorted_ntypes[-1]] + n_node_dict[sorted_ntypes[-1]]

        # Reconstruct edges
        hetero_eidx_dict = {}
        hetero_ewgt_dict = {}
        for meta_rel, A in adj_dict.items():
            A = A.copy().tocoo()
            edge_index = np.stack((A.row, A.col), axis=0)
            edge_weight = A.data
            edge_index[0] += node_idx_bias[meta_rel[0]]
            edge_index[1] += node_idx_bias[meta_rel[-1]]
            hetero_eidx_dict[meta_rel] = edge_index
            hetero_ewgt_dict[meta_rel] = edge_weight

        # Reconstruct features
        tgt_x = x_dict[tgt_ntype]
        hetero_x_dict = {tgt_ntype: tgt_x}
        for ntype in ntypes:
            if ntype != tgt_ntype:
                zero_vec = tgt_x.new_zeros((n_node_dict[ntype], tgt_x.size(1)))
                hetero_x_dict[ntype] = zero_vec
        x = torch.cat([hetero_x_dict[k] for k in sorted_ntypes], dim=0)

        foundation_adj = {}
        for meta_rel in hetero_eidx_dict:
            eidx = hetero_eidx_dict[meta_rel]
            ewgt = hetero_ewgt_dict[meta_rel]
            foundation_adj[meta_rel] = sp.csr_matrix((ewgt, eidx), shape=(N, N))

        return x, foundation_adj

    def loss_fn(self, outs, y, mask=None):
        if mask is not None:
            outs = outs[mask]
            y = y[mask]
        loss = F.cross_entropy(outs, y)
        return loss

    @torch.no_grad()
    def predict(self, data):
        self.eval()
        return self.forward(data)

    @torch.no_grad()
    def evaluate(self, data, mask=None):
        eval_dict = dict()
        logits = self.predict(data)
        y = data.y
        if mask is not None:
            eval_dict['loss'] = self.loss_fn(logits, y, mask).item()
            eval_dict['acc'] = (logits[mask].argmax(1) == y[mask]).sum().item() / y[mask].size(0)
        else:
            eval_dict['loss'] = self.loss_fn(logits, y).item()
            eval_dict['acc'] = (logits.argmax(1) == y).sum().item() / y.size(0)
        return eval_dict

    def set_attrs(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, f'get_{k}'):
                setattr(self, k, getattr(self, f'get_{k}')(v))
            else:
                if isinstance(v, torch.Tensor):
                    setattr(self, k, v.to(self.device))
                else:
                    setattr(self, k, v)

    def __repr__(self):
        return f"{self.name}(p=({self.n_prop_layers},{self.ppr_alpha},intra=[{self.intra_adj_norm},{self.intra_adj_norm_alpha}],inter=[{self.inter_adj_norm},{self.inter_adj_norm_alpha}]), t=({self.n_tran_layers},{self.n_hid}), pn=({self.x_norm}))"


class RGPTN(GraphPTN):
    def __init__(self, in_feats, out_feats,
                 n_estimate_epochs=2,
                 k_votes=64, lam_dynamic=0.5,
                 lam_unsup=0,
                 **kwargs):
        super(RGPTN, self).__init__(in_feats, out_feats, **kwargs)

        self.name = 'RGPTN'

        self.n_estimate_epochs = n_estimate_epochs

        self.k_votes = k_votes
        self.lam_dynamic = lam_dynamic

        self.lam_unsup = lam_unsup

    def fit(self, data, train_mask, val_mask, test_mask, args, verbose=False, **kwargs):
        # Hyper parameters
        args.n_epochs = 400
        args.lr = 0.2
        args.weight_decay = 0.005
        args.early_stopping = 30
        args.stopper_type = 'acc'
        args.max_grad_norm = 1.1

        x, edge_index, y = data.x, data.adj, data.y

        x = self.propagation(x, edge_index)

        x_train = x[train_mask]
        y_train = y[train_mask]
        n_train = x_train.size(0)

        data_val = data.__class__(x=x[val_mask], adj=None, y=y[val_mask])

        if self.lam_dynamic != 0:
            if self.k_votes < 0:
                k_votes = int(n_train * 0.6)
            else:
                k_votes = min(self.k_votes, n_train - 1)
            k_votes = min(k_votes, 1024)  # OOM issue
            if k_votes == n_train - 1:
                x_train_norm = F.normalize(x_train, p=2, dim=1)
                adj_t_knn = torch.mm(x_train_norm, x_train_norm.t())
                adj_t_knn = torch.clamp(adj_t_knn, min=0)

                adj_t_knn = normalize_adj_tensor(adj_t_knn, alpha=-0.5, norm='both')
            else:
                eidx_knn, ewgt_knn = construct_knn_edge_index(x_train, k=k_votes + 1)
                ewgt_knn = torch.clamp(ewgt_knn, min=0)

                # to undirect
                eidx_knn, ewgt_knn = to_undirected(eidx_knn, ewgt_knn, num_nodes=n_train)

                adj_t_knn = SparseTensor.from_edge_index(eidx_knn, ewgt_knn, sparse_sizes=(n_train, n_train))

                # norm
                adj_t_knn = normalize_edge_index(adj_t_knn, alpha=-0.5, norm='both')
        else:
            adj_t_knn = None

        if self.n_estimate_epochs > 0:
            pre_prob, pre_clu = self.estimate_prob(x, y, train_mask, val_mask, test_mask)
            static_y_weight = pre_prob[train_mask, y_train]
        else:
            static_y_weight = torch.ones(n_train).to(self.device)

        # update args
        for k, v in self.train_args.items():
            setattr(args, k, v)

        self.reset_parameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self._prop_cached = True

        stopper = EarlyStopping(patience=args.early_stopping, stopper_type=args.stopper_type, cache_best_weight=True)

        for epoch in range(args.n_epochs):
            self.set_attrs(epoch=epoch)
            t = time.time()

            self.train()
            optimizer.zero_grad()
            outs = self.encode(x)

            preds = F.softmax(outs, dim=-1)

            if adj_t_knn is not None:
                if isinstance(adj_t_knn, SparseTensor):
                    p_votes = spmm(adj_t_knn, preds[train_mask])
                else:
                    p_votes = torch.mm(adj_t_knn, preds[train_mask])
                p_votes = F.normalize(p_votes, p=1)

                # Use inv. entropy
                h_entr = calc_entr(p_votes)
                dynamic_y_weight = 1. - h_entr / np.log(self.n_classes)
            else:
                dynamic_y_weight = 0

            y_reweight = self.lam_dynamic * dynamic_y_weight + static_y_weight
            loss_ce = F.cross_entropy(outs[train_mask], y[train_mask], reduction='none')
            loss_ce = torch.mean(loss_ce * y_reweight)

            if self.lam_unsup != 0:
                entropy = calc_entr(preds)
                group_mask = [train_mask, ~train_mask]
                group_loss = [entropy[mask].mean() for mask in group_mask]
                loss_unsup = sum(group_loss) / len(group_loss)
            else:
                loss_unsup = 0

            loss = loss_ce + self.lam_unsup * loss_unsup

            acc = (outs[train_mask].argmax(1) == y_train).sum().item() / y_train.shape[0]
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()

            # Validation
            eval_dict = self.evaluate(data_val)
            loss_val, acc_val = eval_dict['loss'], eval_dict['acc']

            if verbose:
                print(f"Epoch: {epoch+1:04d} loss_train= {loss:.5f} acc_train= {acc:.5f}",
                      f"loss_val= {loss_val:.5f} acc_val= {acc_val:.5f} time= {time.time()-t:.5f}")

            if stopper.step(self, loss=loss_val, acc=acc_val):
                break

        self.load_state_dict(stopper.best_weight)
        self._prop_cached = False

    def estimate_prob(self, h, y, train_mask, val_mask, test_mask):
        best_val = None
        best_prob = None
        best_clu = None
        clean_mask = train_mask.clone()
        h_norm = F.normalize(h, p=2, dim=1)
        for _ in range(self.n_estimate_epochs):
            h_clu, _ = node_prototype(h[clean_mask], y[clean_mask], self.n_classes)
            prob = h_norm @ F.normalize(h_clu, p=2, dim=1).T
            prob_y = prob[torch.arange(prob.shape[0]), y]
            clean_mask = (prob.argmax(1) == y).cpu() & (prob_y > 0).cpu() & train_mask

            correct_val = (prob[val_mask].argmax(1) == y[val_mask]).sum().item()
            if (best_val is None) or (correct_val > best_val):
                best_val = correct_val
                best_clu = h_clu.detach()
                best_prob = prob.detach()
        best_prob = best_prob.clamp(0, 1)
        return best_prob, best_clu

    def __repr__(self):
        return f"{self.name}(est={self.n_estimate_epochs}, dyn=({self.lam_dynamic}), un=({self.lam_unsup}))"


def calc_entr(p, eps=1e-8):
    return -torch.sum(p * torch.log(p + eps), dim=1)


@torch.no_grad()
def construct_knn_edge_index(x, k=32):
    n_nodes = x.size(0)
    x_norm = F.normalize(x, p=2, dim=1)
    batch_size = 512
    n_batches = (n_nodes + batch_size - 1) // batch_size
    edge_weight = x.new_empty(n_nodes, k)
    edge_cols = torch.LongTensor(size=(n_nodes, k)).to(x.device)
    for i in range(n_batches):
        idx_start = i * batch_size
        idx_end = (i + 1) * batch_size
        topk_values, topk_indices = torch.topk(x_norm[idx_start:idx_end] @ x_norm.T, k=k, dim=1)
        edge_weight[idx_start:idx_end] = topk_values
        edge_cols[idx_start:idx_end] = topk_indices
    edge_rows = torch.arange(n_nodes).to(x.device).repeat_interleave(k)
    edge_index = torch.stack((edge_rows, edge_cols.view(-1)), dim=0)
    return edge_index, edge_weight.view(-1)
