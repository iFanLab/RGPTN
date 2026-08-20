import numpy as np
from copy import deepcopy


class EarlyStopping(object):
    """Early stopping handler with multiple stopping strategies.

    Supported stopping types:
        - 'gcn': Early stopping based on GCN paper methodology
        - 'loss': Stop when validation loss stops improving
        - 'acc': Stop when validation accuracy stops improving
        - 'loss_and_acc': Stop when both loss and accuracy stop improving
        - 'operation': Custom stopping based on a user-defined operation
        - 'never': Never stop (train for all epochs)
    """

    def __init__(self, patience=200, stopper_type=None, cache_best_weight=False, **kwargs):
        self.default_stopper_type_list = [
            'gcn', 'loss', 'acc', 'loss_and_acc',
            'operation', 'never',
        ]
        self.patience = patience
        self.stopper_type = stopper_type if (stopper_type is not None) else 'loss_and_acc'
        assert self.stopper_type in self.default_stopper_type_list, \
            f"Invalid stopper_type '{self.stopper_type}'. Must be one of {self.default_stopper_type_list}"

        self.bad_cnt = 0
        self.best_acc = None
        self.best_loss = None
        self.prev_losses = []
        self.cache_best_weight = cache_best_weight
        self.best_weight = None
        self.stopped = False
        self.updated = False

        if self.stopper_type == 'operation':
            self.operator = kwargs.get('operator', None)
            assert self.operator is not None, "Must provide 'operator' when stopper_type is 'operation'"
            self.best_value = None

    def reset(self):
        """Reset all early stopping state."""
        self.bad_cnt = 0
        self.best_acc = None
        self.best_loss = None
        self.prev_losses = []
        self.stopped = False
        self.updated = False

        if self.stopper_type == 'operation':
            self.best_value = None

    def step(self, model=None, loss=None, acc=None, epoch=None, **kwargs):
        """Execute one step of early stopping check.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        self.updated = False
        if self.stopper_type == 'gcn':
            self.step_by_gcn(model, loss, epoch)
        elif self.stopper_type == 'loss':
            self.step_with_loss(model, loss)
        elif self.stopper_type == 'acc':
            self.step_with_acc(model, acc)
        elif self.stopper_type == 'operation':
            self.step_with_operation(**kwargs)
        elif self.stopper_type == 'never':
            self.step_always(model)
        else:
            self.step_with_loss_and_acc(model, loss, acc)
        return self.stopped

    def _save_checkpoint(self, model):
        """Helper method to save model checkpoint if needed."""
        if model is not None:
            if self.cache_best_weight:
                self.best_weight = deepcopy(model.state_dict())
            if hasattr(model, 'save_checkpoint'):
                model.save_checkpoint()

    def step_by_gcn(self, model, loss, epoch=None):
        """Early stopping based on GCN implementation.

        Reference: https://github.com/tkipf/gcn
        """
        if (epoch is None) and (model is not None):
            epoch = model.epoch
        if epoch > self.patience and loss > np.mean(self.prev_losses[-(self.patience + 1):]):
            self.stopped = True
        if (self.best_loss is None) or (loss < self.best_loss):
            self.best_loss = loss
            self._save_checkpoint(model)
            self.updated = True
        self.prev_losses.append(loss)

    def step_with_loss(self, model, loss):
        """Stop when validation loss stops improving."""
        if self.best_loss is None:
            self.best_loss = loss
            self._save_checkpoint(model)
            self.updated = True
        elif loss <= self.best_loss:
            self.bad_cnt = 0
            if loss < self.best_loss:
                self.best_loss = loss
                self._save_checkpoint(model)
                self.updated = True
        else:
            self.bad_cnt += 1
            if self.bad_cnt >= self.patience:
                self.stopped = True

    def step_with_acc(self, model, acc):
        """Stop when validation accuracy stops improving.

        Reference: https://github.com/bdy9527/FAGCN (used for disassortative graphs)
        """
        if self.best_acc is None:
            self.best_acc = acc
            self._save_checkpoint(model)
            self.updated = True
        elif acc >= self.best_acc:
            self.bad_cnt = 0
            if acc > self.best_acc:
                self.best_acc = acc
                self._save_checkpoint(model)
                self.updated = True
        else:
            self.bad_cnt += 1
            if self.bad_cnt >= self.patience:
                self.stopped = True

    def step_with_loss_and_acc(self, model, loss, acc):
        """Stop when both loss and accuracy stop improving.

        Reference: https://github.com/dmlc/dgl
        """
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self._save_checkpoint(model)
            self.updated = True
        elif (loss <= self.best_loss) or (acc >= self.best_acc):
            self.bad_cnt = 0
            self.best_loss = min(self.best_loss, loss)
            self.best_acc = max(self.best_acc, acc)
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self._save_checkpoint(model)
                self.updated = True
        else:
            self.bad_cnt += 1
            if self.bad_cnt >= self.patience:
                self.stopped = True

    def step_with_operation(self, value=None):
        """Custom stopping based on a user-defined operation."""
        if self.best_value is None:
            self.best_value = value
            self.updated = True
        elif self.operator(value, self.best_value):
            self.bad_cnt = 0
            self.best_value = value
            self.updated = True
        else:
            self.bad_cnt += 1
            if self.bad_cnt >= self.patience:
                self.stopped = True

    def step_always(self, model):
        """Never stop early, always save the latest checkpoint."""
        self._save_checkpoint(model)
        self.updated = True
