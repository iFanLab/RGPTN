import argparse
from omegaconf import OmegaConf
import os.path as osp

from datasets import load_data
from models import RGPTN
from utils import setup, save_checkpoint, load_checkpoint


def run_exp(args):
    env = setup(args)
    model_cfg = get_model_config(args)
    device = env['device']

    data, data_fmt_str = load_data(args.dataset, root=args.data_dir, noise_type=args.noise_type, noise_rate=args.noise_rate, seed=args.seed)
    if data.gtype == 'hetero':
        in_feats = data.x[data.predict_ntype].size(1)
        n_classes = int(data.y[data.predict_ntype].max() + 1)
    else:
        in_feats = data.x.size(1)
        n_classes = int(data.y.max() + 1)

    if str.lower(args.model) == 'rgptn':
        model = RGPTN(in_feats, n_classes, **model_cfg, device=device, data=data)
    else:
        raise NotImplementedError

    if args.verbose:
        print('\n=== Exp. Stats.')
        print('[Device]', device)
        # print('[Data]', data)
        print('[Model]', model, model_cfg)
        print('===\n')

    model = model.to(device)
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    model.set_attrs(data=data)

    checkpoint_path = osp.join(args.model_dir, f"{model.name}_{data_fmt_str}.pt")
    if args.verbose:
        print(f"Try to load ckpt from <{checkpoint_path}>")
    try:
        if osp.exists(checkpoint_path):
            load_checkpoint(model, checkpoint_path)
        else:
            raise FileNotFoundError(f"Not found ckpt <{model.name}_{data_fmt_str}.pt>, try to retrain model")
    except FileNotFoundError:
        model.fit(model.data, train_mask, val_mask, test_mask, args, verbose=args.verbose)
        # save_checkpoint(model, checkpoint_path)

    eval_dict = model.evaluate(model.data, test_mask)

    rst_dict = eval_dict.copy()
    rst_dict['dataset_stats'] = data_fmt_str
    rst_dict['model'] = str(model)
    print(f"\033[91m{rst_dict}\n\033[0m")
    return rst_dict


def get_args():
    parser = argparse.ArgumentParser(description='Reliability GNNs')
    parser.add_argument('--dataset', type=str, default='cora', help="dataset name")
    parser.add_argument('--model', type=str, default='rgptn', help="model")
    parser.add_argument('--seed', type=int, default=123, help="random seed")
    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--noise_type', default='uniform', choices=['uniform', 'pair'], help="noise type")
    parser.add_argument('--noise_rate', type=float, default=0.4, help="noise rate")

    parser.add_argument('--n_epochs', type=int, default=1500, help="number of training epochs")
    parser.add_argument('--lr', type=float, default=0.005, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="l2 weight decay")
    parser.add_argument('--patience', type=int, default=200, help="tolerance for early stopping (# of epochs)")
    parser.add_argument('--gpu', type=int, default=0, help="gpu")

    parser.add_argument('--model_dir', type=str, default='./ckpt', help="dir for saving model")
    parser.add_argument('--data_dir', type=str, default='./data', help="dir for saving data")
    parser.add_argument('--model_config', type=str, default='./configs/rgptn.yaml', help="get model config from file")

    args = parser.parse_args()
    return args


def get_model_config(args):
    dataset_str = str.lower(args.dataset)
    model_cfg = OmegaConf.load(args.model_config)
    model_cfg = OmegaConf.merge(model_cfg['common'], model_cfg.get(dataset_str, {}))
    return dict(model_cfg)


if __name__ == '__main__':
    args = get_args()
    run_exp(args)
