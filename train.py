import torch
from models import EBM, FC, IterativeFC, IterativeAttention, IterativeFCAttention, \
    IterativeTransformer, EBMTwin, RecurrentFC, PonderFC
import torch.nn.functional as F
import os
import pdb
from dataset import LowRankDataset, ShortestPath, Negate, Inverse, Square, Identity, \
    Det, LU, Sort, Eigen, QR, Equation, FiniteWrapper, Parity, Addition
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os.path as osp
import numpy as np
from imageio import imwrite
import argparse
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
from torchvision.utils import make_grid
import seaborn as sns


def worker_init_fn(worker_id):
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, inputs):
        batch_size = len(inputs)
        if self._next_idx >= len(self._storage):
            self._storage.extend(inputs)
        else:
            if batch_size + self._next_idx < self._maxsize:
                self._storage[self._next_idx:self._next_idx +
                              batch_size] = inputs
            else:
                split_idx = self._maxsize - self._next_idx
                self._storage[self._next_idx:] = inputs[:split_idx]
                self._storage[:batch_size - split_idx] = inputs[split_idx:]
        self._next_idx = (self._next_idx + batch_size) % self._maxsize

    def _encode_sample(self, idxes):
        inps = []
        opts = []
        targets = []
        scratchs = []

        # Store in the intermediate state of optimization problem
        for i in idxes:
            inp, opt, target, scratch = self._storage[i]
            opt = opt
            inps.append(inp)
            opts.append(opt)
            targets.append(target)
            scratchs.append(scratch)

        inps = np.array(inps)
        opts = np.array(opts)
        targets = np.array(targets)
        scratchs = np.array(scratchs)

        return inps, opts, targets, scratchs

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes), torch.Tensor(idxes)

    def set_elms(self, data, idxes):
        if len(self._storage) < self._maxsize:
            self.add(data)
        else:
            for i, ix in enumerate(idxes):
                self._storage[ix] = data[i]


"""Parse input arguments"""
parser = argparse.ArgumentParser(description='Train EBM model')

parser.add_argument('--train', action='store_true',
                    help='whether or not to train')
parser.add_argument('--cuda', action='store_true',
                    help='whether to use cuda or not')
parser.add_argument('--no_replay_buffer', action='store_true',
                    help='do not use a replay buffer to train models')
parser.add_argument('--dataset', default='negate', type=str,
                    help='dataset to evaluate')
parser.add_argument('--logdir', default='cachedir', type=str,
                    help='location where log of experiments will be stored')
parser.add_argument('--exp', default='default', type=str,
                    help='name of experiments')

# training
parser.add_argument('--resume_iter', default=0, type=int,
                    help='iteration to resume training')
parser.add_argument('--batch_size', default=512, type=int,
                    help='size of batch of input to use')
parser.add_argument('--num_epoch', default=10000, type=int,
                    help='number of epochs of training to run')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate for training')
parser.add_argument('--log_interval', default=10, type=int,
                    help='log outputs every so many batches')
parser.add_argument('--save_interval', default=1000, type=int,
                    help='save outputs every so many batches')

# data
parser.add_argument('--data_workers', default=4, type=int,
                    help='Number of different data workers to load data in parallel')

# Model specific settings
parser.add_argument('--rank', default=20, type=int,
                    help='rank of matrix to use')
parser.add_argument('--num_steps', default=10, type=int,
                    help='Steps of gradient descent for training')
parser.add_argument('--step_lr', default=100.0, type=float,
                    help='step size of latents')
parser.add_argument('--ood', action='store_true',
                    help='test on the harder ood dataset')
parser.add_argument('--recurrent', action='store_true',
                    help='utilize a recurrent model to output prediction')
parser.add_argument('--ponder', action='store_true',
                    help='utilize a ponder network model to output prediction')
parser.add_argument('--decoder', action='store_true',
                    help='utilize a decoder network to output prediction')
parser.add_argument('--iterative_decoder', action='store_true',
                    help='utilize a decoder to output prediction')
parser.add_argument('--mem', action='store_true',
                    help='add external memory to compute answers')
parser.add_argument('--no_truncate', action='store_true',
                    help='don"t truncate gradient backprop')

# Distributed training hyperparameters
parser.add_argument('--gpus', default=1, type=int,
                    help='number of gpus to train with')
parser.add_argument('--node_rank', default=0, type=int, help='rank of node')
parser.add_argument('--capacity', default=50000, type=int,
                    help='number of elements to generate')
parser.add_argument('--infinite', action='store_true',
                    help='makes the dataset have an infinite number of elements')


best_test_error_10 = 10.0
best_test_error_20 = 10.0
best_test_error_40 = 10.0
best_test_error_80 = 10.0
best_test_error = 10.0


def average_gradients(model):
    size = float(dist.get_world_size())

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def gen_answer(inp, FLAGS, model, pred, scratchpad, num_steps, create_graph=True):
    """
        Implement iterative reasoning to obtain the answer to a problem
    """

    # List of intermediate predictions
    preds = []
    im_grads = []
    energies = []
    logits = []

    if FLAGS.decoder:
        pred = model.forward(inp)
        preds = [pred]
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
    elif FLAGS.recurrent:
        preds = []
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
        state = None
        for i in range(num_steps):
            pred, state = model.forward(inp, state)
            preds.append(pred)
    elif FLAGS.ponder:
        im_merge = torch.cat([pred, inp], dim=-1)

        preds, logits = model.forward(im_merge, iters=num_steps)
        pred = preds[-1]
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
        state = None

    elif FLAGS.iterative_decoder:
        for i in range(num_steps):
            energy = torch.zeros(1)

            noise_add = (torch.rand_like(pred) - 0.5)
            out_dim = model.out_dim

            im_merge = torch.cat([pred, inp], dim=-1)
            pred = model.forward(im_merge) + pred

            preds.append(pred)
            energies.append(torch.zeros(1))
            im_grads.append(torch.zeros(1))
    else:
        with torch.enable_grad():
            pred.requires_grad_(requires_grad=True)
            s = inp.size()
            scratchpad.requires_grad_(requires_grad=True)
            preds.append(pred)

            for i in range(num_steps):
                noise = torch.rand_like(pred) - 0.5

                if FLAGS.mem:
                    im_merge = torch.cat([pred, inp, scratchpad], dim=-1)
                else:
                    im_merge = torch.cat([pred, inp], dim=-1)

                energy = model.forward(im_merge)

                if FLAGS.mem:
                    im_grad, scratchpad_grad = torch.autograd.grad(
                        [energy.sum()], [pred, scratchpad], create_graph=create_graph)
                else:

                    if FLAGS.no_truncate:
                        im_grad, = torch.autograd.grad(
                            [energy.sum()], [pred], create_graph=create_graph)
                    else:
                        if i != (num_steps - 1):
                            im_grad, = torch.autograd.grad(
                                [energy.sum()], [pred], create_graph=False)
                        else:
                            im_grad, = torch.autograd.grad(
                                [energy.sum()], [pred], create_graph=create_graph)

                pred = pred - FLAGS.step_lr * im_grad

                if FLAGS.mem:
                    scratchpad = scratchpad - FLAGS.step_lr * scratchpad
                    scratchpad = torch.clamp(scratchpad, -1, 1)

                preds.append(pred)
                energies.append(energy)
                im_grads.append(im_grad)

    return pred, preds, im_grads, energies, scratchpad, logits


def ema_model(model, model_ema, mu=0.999):
    for (model, model_ema) in zip(model, model_ema):
        for param, param_ema in zip(
                model.parameters(), model_ema.parameters()):
            param_ema.data[:] = mu * param_ema.data + (1 - mu) * param.data


def sync_model(model):
    size = float(dist.get_world_size())

    for model in model:
        for param in model.parameters():
            dist.broadcast(param.data, 0)


def init_model(FLAGS, device, dataset):
    if FLAGS.decoder:
        model = FC(dataset.inp_dim, dataset.out_dim)
    elif FLAGS.recurrent:
        model = RecurrentFC(dataset.inp_dim, dataset.out_dim)
    elif FLAGS.ponder:
        model = PonderFC(dataset.inp_dim, dataset.out_dim, FLAGS.num_steps)
    elif FLAGS.iterative_decoder:
        model = IterativeFC(dataset.inp_dim, dataset.out_dim, FLAGS.mem)
    else:
        model = EBM(dataset.inp_dim, dataset.out_dim, FLAGS.mem)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    return model, optimizer


def safe_cumprod(t, eps=1e-10, dim=-1):
    t = torch.clip(t, min=eps, max=1.)
    return torch.exp(torch.cumsum(torch.log(t), dim=dim))


def exclusive_cumprod(t, dim=-1):
    cum_prod = safe_cumprod(t, dim=dim)
    return pad_to(cum_prod, (1, -1), value=1., dim=dim)


def calc_geometric(l, dim=-1):
    return exclusive_cumprod(1 - l, dim=dim) * l


def test(train_dataloader, model, FLAGS, step=0):
    global best_test_error_10, best_test_error_20, best_test_error_40, best_test_error_80, best_test_error
    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    replay_buffer = None
    dist_list = []
    energy_list = []
    min_dist_list = []

    model.eval()
    counter = 0

    with torch.no_grad():
        for inp, im in train_dataloader:
            im = im.float().to(dev)
            inp = inp.float().to(dev)

            # Initialize prediction from random guess
            pred = (torch.rand_like(im) - 0.5) * 2
            scratch = torch.zeros_like(inp)

            pred_init = pred
            pred, preds, im_grad, energies, scratch, logits = gen_answer(
                inp, FLAGS, model, pred, scratch, 80)
            preds = torch.stack(preds, dim=0)

            if FLAGS.ponder:
                halting_probs = calc_geometric(logits.sigmoid(), dim=1)[..., 0]
                cum_halting_probs = torch.cumsum(halting_probs, dim=-1)
                rand_val = torch.rand(
                    halting_probs.size(0)).to(
                    cum_halting_probs.device)
                sort_id = torch.searchsorted(
                    cum_halting_probs, rand_val[:, None])
                sort_id = torch.clamp(sort_id, 0, sort_id.size(1) - 1)
                sort_id = sort_id[:, :, None].expand(-1, -1, pred.size(-1))

            energies = torch.stack(energies, dim=0)

            dist = (preds - im[None, :])
            dist = torch.pow(dist, 2)

            dist = dist.mean(dim=-1)
            n = dist.size(1)

            dist_energies = dist[1:, :]
            min_idx = energies[:, :, 0].argmin(dim=0)[None, :]
            dist_min_energy = torch.gather(dist_energies, 0, min_idx)
            min_dist_list.append(dist_min_energy.detach())

            dist = dist.mean(dim=-1)

            energies = energies.mean(dim=-1).mean(dim=-1)
            dist_list.append(dist.detach())
            energy_list.append(energies.detach())

            counter = counter + 1

            if counter > 10:
                dist_list = torch.stack(dist_list, dim=0)
                dist = dist_list.mean(dim=0)
                energy_list = torch.stack(energy_list, dim=0)
                energies = energy_list.mean(dim=0)
                min_dist = torch.stack(min_dist_list, dim=0).mean()

                print("Testing..................")
                print("step errors: ", dist[:20])
                print("energy values: ", energies)
                print('test at step %d done!' % step)
                break

    if FLAGS.decoder or FLAGS.ponder:
        best_test_error_10 = min(best_test_error_10, dist[0].item())
        best_test_error_20 = min(best_test_error_20, dist[0].item())
        best_test_error_40 = min(best_test_error_40, dist[0].item())
        best_test_error_80 = min(best_test_error_80, dist[0].item())
        best_test_error = min(best_test_error, dist[0].item())
    else:
        best_test_error_10 = min(best_test_error_10, dist[9].item())
        best_test_error_20 = min(best_test_error_20, dist[19].item())
        best_test_error_40 = min(best_test_error_40, dist[39].item())
        best_test_error_80 = min(best_test_error_80, dist[79].item())
        best_test_error = min(best_test_error, min_dist.item())

    print("best test error (10, 20, 40, 80, min_energy): {} {} {} {} {}".format(
            best_test_error_10, best_test_error_20, best_test_error_40,
            best_test_error_80, best_test_error))

    model.train()


def train(train_dataloader, test_dataloader, logger, model,
          optimizer, FLAGS, logdir, rank_idx):

    it = FLAGS.resume_iter
    optimizer.zero_grad()
    dev = torch.device("cuda")

    # initalize a replay buffer of solutions
    replay_buffer = ReplayBuffer(10000)

    for epoch in range(FLAGS.num_epoch):
        for inp, im in train_dataloader:
            im = im.float().to(dev)
            inp = inp.float().to(dev)

            # Initalize a solution from random
            pred = (torch.rand_like(im) - 0.5) * 2

            # Sample a proportion of samples from past optimization results
            if FLAGS.replay_buffer and len(replay_buffer) >= FLAGS.batch_size:
                replay_batch, _ = replay_buffer.sample(im.size(0))
                inp_replay, opt_replay, gt_replay, scratch_replay = replay_batch

                replay_mask = np.concatenate( [np.ones(im.size(0)), np.zeros(im.size(0))]).astype(np.bool)
                inp = torch.cat([torch.Tensor(inp_replay).cuda(), inp], dim=0)
                pred = torch.cat([torch.Tensor(opt_replay).cuda(), pred], dim=0)
                im = torch.cat([torch.Tensor(gt_replay).cuda(), im], dim=0)
            else:
                replay_mask = (
                    np.random.uniform(
                        0,
                        1,
                        im.size(0)) > 1.0)

            scratch = torch.zeros_like(inp)

            num_steps = FLAGS.num_steps
            pred, preds, im_grads, energies, scratch, logits = gen_answer(
                inp, FLAGS, model, pred, scratch, num_steps)
            energies = torch.stack(energies, dim=0)
            preds = torch.stack(preds, dim=1)

            im_grads = torch.stack(im_grads, dim=1)

            if FLAGS.ponder:
                geometric_dist = calc_geometric(torch.full(
                    (FLAGS.num_steps,), 1 / FLAGS.num_steps, device=dev))
                halting_probs = calc_geometric(logits.sigmoid(), dim=1)[..., 0]

            if FLAGS.decoder:
                im_loss = torch.pow(
                    preds[:, -1:] - im[:, None, :], 2).mean(dim=-1).mean(dim=-1)
            elif FLAGS.ponder:
                halting_probs = halting_probs / \
                    halting_probs.sum(dim=1)[:, None]
                im_loss = (torch.pow(
                    preds[:, :] - im[:, None, :], 2)).mean(dim=-1).mean(dim=-1)
            else:
                im_loss = torch.pow(
                    preds[:, -1:] - im[:, None, :], 2).mean(dim=-1).mean(dim=-1)

            loss = im_loss.mean()

            if FLAGS.ponder:
                ponder_loss = 0.01 * \
                    F.kl_div(torch.log(geometric_dist[None, :] + 1e-10), halting_probs, None, None, 'batchmean')
                loss = loss + ponder_loss

            loss.backward()

            if FLAGS.replay_buffer:
                inp_replay = inp.cpu().detach().numpy()
                pred_replay = pred.cpu().detach().numpy()
                im_replay = im.cpu().detach().numpy()
                scratch = scratch.cpu().detach().numpy()
                encode_tuple = list(zip(list(inp_replay), list(
                    pred_replay), list(im_replay), list(scratch)))

                replay_buffer.add(encode_tuple)

            if FLAGS.gpus > 1:
                average_gradients(model)

            optimizer.step()
            optimizer.zero_grad()

            if it > 10000:
                assert False

            if it % FLAGS.log_interval == 0 and rank_idx == 0:
                loss = loss.item()
                kvs = {}
                kvs['im_loss'] = im_loss.mean().item()

                if it > 10:
                    replay_mask = replay_mask
                    no_replay_mask = ~replay_mask
                    kvs['no_replay_loss'] = im_loss[no_replay_mask].mean().item()
                    kvs['replay_loss'] = im_loss[replay_mask].mean().item()

                    if FLAGS.ponder:
                        kvs['ponder_loss'] = ponder_loss

                    if (not FLAGS.iterative_decoder) and (not FLAGS.decoder) and (
                            not FLAGS.recurrent) and (not FLAGS.ponder):
                        kvs['energy_no_replay'] = energies[-1,
                                                           no_replay_mask].mean().item()
                        kvs['energy_replay'] = energies[-1,
                                                        replay_mask].mean().item()

                        kvs['energy_start_no_replay'] = energies[0,
                                                                 no_replay_mask].mean().item()
                        kvs['energy_start_replay'] = energies[0,
                                                              replay_mask].mean().item()

                mean_last_dist = torch.abs(pred - im).mean()
                kvs['mean_last_dist'] = mean_last_dist.item()

                string = "Iteration {} ".format(it)

                for k, v in kvs.items():
                    string += "%s: %.6f  " % (k, v)
                    logger.add_scalar(k, v, it)

                print(string)

            if it % FLAGS.save_interval == 0 and rank_idx == 0:
                model_path = osp.join(logdir, "model_latest.pth".format(it))
                ckpt = {'FLAGS': FLAGS}

                ckpt['model_state_dict'] = model.state_dict()
                ckpt['optimizer_state_dict'] = optimizer.state_dict()

                torch.save(ckpt, model_path)

                test(test_dataloader, model, FLAGS, step=it)

            it += 1


def main_single(rank, FLAGS):
    rank_idx = rank
    world_size = FLAGS.gpus
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)

    if not os.path.exists('result/%s' % FLAGS.exp):
        try:
            os.makedirs('result/%s' % FLAGS.exp)
        except BaseException:
            pass

    if not os.path.exists(logdir):
        try:
            os.makedirs('logdir')
        except BaseException:
            pass

    # Load Dataset
    if FLAGS.dataset == 'lowrank':
        dataset = LowRankDataset('train', FLAGS.rank, FLAGS.ood)
        test_dataset = LowRankDataset('test', FLAGS.rank, FLAGS.ood)
    elif FLAGS.dataset == 'shortestpath':
        dataset = ShortestPath('train', FLAGS.rank, FLAGS.num_steps)
        test_dataset = ShortestPath('test', FLAGS.rank, FLAGS.num_steps)
    elif FLAGS.dataset == 'negate':
        dataset = Negate('train', FLAGS.rank)
        test_dataset = Negate('test', FLAGS.rank)
    elif FLAGS.dataset == 'addition':
        dataset = Addition('train', FLAGS.rank, FLAGS.ood)
        test_dataset = Addition('test', FLAGS.rank, FLAGS.ood)
    elif FLAGS.dataset == 'inverse':
        dataset = Inverse('train', FLAGS.rank, FLAGS.ood)
        test_dataset = Inverse('test', FLAGS.rank, FLAGS.ood)
    elif FLAGS.dataset == 'square':
        dataset = Square('train', FLAGS.rank, FLAGS.num_steps)
        test_dataset = Square('test', FLAGS.rank, FLAGS.num_steps)
    elif FLAGS.dataset == 'identity':
        dataset = Identity('train', FLAGS.rank, FLAGS.num_steps)
        test_dataset = Identity('test', FLAGS.rank, FLAGS.num_steps)
    elif FLAGS.dataset == 'det':
        dataset = Det('train', FLAGS.rank)
        test_dataset = Det('test', FLAGS.rank)
    elif FLAGS.dataset == 'lu':
        dataset = LU('train', FLAGS.rank)
        test_dataset = LU('test', FLAGS.rank)
    elif FLAGS.dataset == 'sort':
        dataset = Sort('train', FLAGS.rank, FLAGS.num_steps)
        test_dataset = Sort('test', FLAGS.rank, FLAGS.num_steps)
    elif FLAGS.dataset == 'eigen':
        dataset = Eigen('train', FLAGS.rank, FLAGS.num_steps)
        test_dataset = Eigen('test', FLAGS.rank, FLAGS.num_steps)
    elif FLAGS.dataset == 'equation':
        dataset = Equation('train', FLAGS.rank, FLAGS.num_steps)
        test_dataset = Equation('test', FLAGS.rank, FLAGS.num_steps)
    elif FLAGS.dataset == 'qr':
        dataset = QR('train', FLAGS.rank, FLAGS.num_steps)
        test_dataset = QR('test', FLAGS.rank, FLAGS.num_steps)
    elif FLAGS.dataset == 'parity':
        dataset = Parity('train', FLAGS.rank, FLAGS.num_steps)
        test_dataset = Parity('test', FLAGS.rank, FLAGS.num_steps)

    if not FLAGS.infinite:
        dataset = FiniteWrapper(
            dataset,
            FLAGS.dataset,
            FLAGS.capacity,
            FLAGS.rank,
            FLAGS.num_steps)

    shuffle = True
    sampler = None

    if world_size > 1:
        group = dist.init_process_group(
            backend='nccl',
            init_method='tcp://localhost:8113',
            world_size=world_size,
            rank=rank_idx,
            group_name="default")

    torch.cuda.set_device(rank)
    device = torch.device('cuda')

    FLAGS_OLD = FLAGS

    # Load model and key arguments
    if FLAGS.resume_iter != 0:
        model_path = osp.join(
            logdir, "model_latest.pth".format(
                FLAGS.resume_iter))

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        FLAGS = checkpoint['FLAGS']

        FLAGS.resume_iter = FLAGS_OLD.resume_iter
        FLAGS.save_interval = FLAGS_OLD.save_interval
        FLAGS.gpus = FLAGS_OLD.gpus
        FLAGS.train = FLAGS_OLD.train
        FLAGS.batch_size = FLAGS_OLD.batch_size
        FLAGS.step_lr = FLAGS_OLD.step_lr
        FLAGS.num_steps = FLAGS_OLD.num_steps
        FLAGS.exp = FLAGS_OLD.exp
        FLAGS.ponder = FLAGS_OLD.ponder
        FLAGS.heatmap = FLAGS_OLD.heatmap

        model, optimizer = init_model(FLAGS, device, dataset)
        state_dict = model.state_dict()

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model, optimizer = init_model(FLAGS, device, dataset)

    if FLAGS.gpus > 1:
        sync_model(model)

    print("num_parameters: ", sum([p.numel() for p in model.parameters()]))

    train_dataloader = DataLoader(
        dataset,
        num_workers=FLAGS.data_workers,
        batch_size=FLAGS.batch_size,
        shuffle=shuffle,
        pin_memory=False,
        worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(
        test_dataset,
        num_workers=FLAGS.data_workers,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=worker_init_fn)

    logger = SummaryWriter(logdir)
    it = FLAGS.resume_iter

    if FLAGS.train:
        model.train()
    else:
        model.eval()

    if FLAGS.train:
        train(
            train_dataloader,
            test_dataloader,
            logger,
            model,
            optimizer,
            FLAGS,
            logdir,
            rank_idx)
    else:
        test(test_dataloader, model, FLAGS, step=FLAGS.resume_iter)


def main():
    FLAGS = parser.parse_args()
    FLAGS.replay_buffer = not FLAGS.no_replay_buffer
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)

    if FLAGS.recurrent:
        FLAGS.no_replay_buffer = True

    if FLAGS.decoder:
        FLAGS.no_replay_buffer = True

    if not osp.exists(logdir):
        os.makedirs(logdir)

    if FLAGS.gpus > 1:
        mp.spawn(main_single, nprocs=FLAGS.gpus, args=(FLAGS,))
    else:
        main_single(0, FLAGS)


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except BaseException:
        pass

    main()
