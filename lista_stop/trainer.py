import torch
from tqdm import tqdm
import torch.nn.functional as F
from lista_stop.common.consts import DEVICE
from lista_stop.common.utils import max_onehot
import math


def MyMSE(x_hat, x):
    return torch.sum((x_hat - x) ** 2, dim=-1).view(-1)


class ListaModelMLE:

    def __init__(self, args, lista_net, optimizer, data_loader, loss_type='weighted'):
        self.args = args
        self.lista_net = lista_net
        self.optimizer = optimizer
        self.train_data_generator = data_loader.gen_samples(args.batch_size)
        self.data_loader = data_loader
        self.loss_type = loss_type
        self.epochs = args.num_epochs
        self.min_temp = args.min_temp
        self.max_temp = args.max_temp
        self.num_itr = self.epochs * args.iters_per_eval
        self.diff_temp = (self.max_temp - self.min_temp) / self.num_itr
        if args.max_temp > 0:
            self.temp = args.min_temp
        else:
            self.temp = args.loss_temp
        # for checkpoint recovery
        self.start_epoch = 1
        self.var = self.args.var
        self.var2 = 2 * self.var

    def train(self):

        best_val_loss = None
        progress_bar = tqdm(range(self.start_epoch, self.epochs + 1))
        dsc = ''
        for epoch in progress_bar:
            epoch_log = self._train_epoch(epoch, progress_bar, dsc)

            epoch_valid_log, val_loss = self._valid_epoch()

            if math.isnan(float(val_loss.item())):
                break

            dsc = ''
            for key, value in epoch_valid_log.items():
                dsc += '%s:%0.3f,' % (key, value)

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                #print('saving model with best validation loss')
                torch.save(self.lista_net.state_dict(), self.args.val_model_dump)
            dsc += 'best:%0.3f' % best_val_loss

    def _train_epoch(self, epoch, progress_bar, dsc):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains information about training

        """
        self.lista_net.train()
        total_loss = 0.0
        for it in range(self.args.iters_per_eval):
            # get a batch training data
            y, x = next(self.train_data_generator)
            batch_size = y.shape[0]
            # forward
            self.optimizer.zero_grad()
            xhs = self.lista_net(y)

            # Loss type == 'sum' is the standard loss for training the original LISTA model.
            if self.loss_type == 'sum':
                loss = 0.0
                for t in range(self.lista_net.num_output):
                    x_hat = xhs[t]
                    loss += MyMSE(x_hat, x).mean()
                loss = loss / self.lista_net.num_output
            else:
                # Loss type == 'mle' corresponds to the training loss for the stage 1 training of LISTA-stop.
                # The loss in Eq (9) in the paper is equivalent to the following implementation of the Log
                # Marginal Likelihood Estimation.
                # It might be not so straightforward to see the equivalence. Please feel free to email
                # xinshi.chen@gatech.edu if you have questions.
                assert self.loss_type == 'mle'
                ll_t = []  # likelihood
                for t in range(self.lista_net.num_output):
                    x_hat = xhs[t]
                    mse = MyMSE(x_hat, x)
                    ll_t.append(torch.exp(-mse / self.var2))
                ll_all = torch.stack(ll_t, dim=0)
                loss = - torch.log(torch.sum(ll_all, dim=0)).mean()

            # backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_description('epoch %.2f, tr loss %.3f ' % (epoch + float(it + 1) / self.args.iters_per_eval, loss.item()) + dsc)
            if math.isnan(float(total_loss)):
                break
        log = {
            'epoch': epoch,
            'avg loss in this epoch': total_loss / self.args.iters_per_eval,
            'batch size': self.args.batch_size
        }

        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        """
        self.lista_net.eval()
        val_y, val_x = self.data_loader.static_data['val']
        batch_size = val_y.shape[0]
        # error
        with torch.no_grad():
            xhs = self.lista_net(val_y)
            # val loss of last layer
            x_hat = xhs[-1]
            val_loss_last = MyMSE(x_hat, val_x).mean()
            # val loss of best layer
            loss_t = []
            for t in range(self.lista_net.num_output):
                x_hat = xhs[t]
                loss_t.append(MyMSE(x_hat, val_x))
            loss_all = torch.stack(loss_t, dim=0)
            val_loss_best, _ = torch.min(loss_all, dim=0)
            val_loss_best = val_loss_best.mean()

        if self.loss_type == 'sum':
            val_loss = val_loss_last
        elif self.loss_type == 'mle':
            val_loss = val_loss_best

        self.lista_net.train()

        log = {
            'last layer': val_loss_last,
            'best layer': val_loss_best
        }

        return log, val_loss


class PolicyKL:
    def __init__(self, args, lista_net, score_net, train_post, nz_post, optimizer, data_loader):
        self.args = args
        self.lista_net = lista_net
        self.score_net = score_net
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.lista_net.eval()
        self.train_data_generator = data_loader.gen_samples(args.batch_size)
        self.epochs = args.num_epochs
        self.var = self.args.var
        self.train_post = train_post
        self.nz_post = nz_post

    def _train_epoch(self, epoch):
        # n outputs, n-1 nets
        self.score_net.train()

        total_loss = 0.0
        for it in range(self.args.iters_per_eval):
            # generate path
            y, x = next(self.train_data_generator)
            xhs = self.lista_net(y)

            scores = self.score_net(y, xhs)

            self.optimizer.zero_grad()
            # loss
            if self.args.kl_type == 'forward':
                loss, _ = self.forward_kl_loss(x, xhs, scores)
            else:
                assert self.args.kl_type == 'backward'
                loss, _ = self.backward_kl_loss(x, xhs, scores)

            # backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        log = {
            'epo': epoch,
            'train loss': total_loss / self.args.iters_per_eval
        }

        return log

    def _valid_epoch(self):
        """
        validation after training an epoch
        :return:
        """
        self.score_net.eval()

        y, x = self.data_loader.static_data['val']
        batch_size = y.shape[0]
        with torch.no_grad():
            xhs = self.lista_net(y)
            scores = self.score_net(y, xhs)
            stop_idx = self.q_posterior(self.args.policy_type, scores, stochastic=False)
            q = self.q_posterior(self.args.policy_type, scores, stochastic=True)

            # validation loss
            if self.args.kl_type == 'forward':
                loss, mse_all = self.forward_kl_loss(x, xhs, scores)
            else:
                assert self.args.kl_type == 'backward'
                loss, mse_all = self.backward_kl_loss(x, xhs, scores)

            mse_det = torch.sum(mse_all * stop_idx, dim=-1)
            assert mse_det.shape[0] == batch_size
            mse_det = mse_det.mean()

            mse_sto = torch.sum(mse_all * q, dim=-1)
            mse_sto = mse_sto.mean()

            if self.args.stochastic:
                val_metric = mse_sto
                log = {
                    'val loss': loss,
                    'sto q': torch.mean(q, dim=0),
                    'mse': mse_sto,
                }
            else:
                val_metric = mse_det
                log = {
                    'val loss': loss,
                    'det q': torch.mean(stop_idx, dim=0),
                    'mse': mse_det,
                }
        return log, val_metric

    @staticmethod
    def test(args, eval, score_net, lista_net, nz_post):
        lista_net.eval()
        score_net.eval()
        with torch.no_grad():
            xhs = lista_net(eval.y)
            scores = score_net(eval.y, xhs)
            stop_idx = PolicyKL.stop_idx(args.policy_type, scores, stochastic=args.stochastic)
            q_posterior = PolicyKL.q_posterior(type=args.policy_type, scores=scores, stochastic=True)
            assert stop_idx.shape[0] == eval.test_size
            x_hat_all = torch.stack(xhs, dim=1)
            x_hat_all = torch.stack([x_hat_all[:, t, :] for t in nz_post.values()], dim=1)
            x_hat_output = torch.einsum('bij,bi->bj', x_hat_all, stop_idx)
            assert x_hat_output.shape[0] == eval.test_size

            mse, nmse, mse_per_snr, nmse_per_snr = eval.compute(x_hat_output)
            print('%d samples, mse: %.5f, nmse: %.5f'
                  % (eval.test_size, mse, nmse))
            print(eval.mix)
            nmse_print = 'nmse per snr'
            for nmse in nmse_per_snr:
                nmse_print += ', %.5f' % nmse
            print(nmse_print)
        return x_hat_all, stop_idx, q_posterior

    @staticmethod
    def converge_rate(args, eval, score_net, lista_net, nz_post):
        lista_net.eval()
        score_net.eval()
        denominator = torch.sum(eval.x ** 2, dim=-1).mean()
        batch_size = eval.y.shape[0]
        # q_posterior_all = torch.zeros([batch_size, args.num_output]).to(DEVICE)
        with torch.no_grad():
            xhs = lista_net(eval.y)
            scores = score_net(eval.y, xhs)
            q_posterior = PolicyKL.q_posterior(type=args.policy_type, scores=scores, stochastic=True)

            x_hat_all = torch.stack(xhs, dim=1)
            nmse_dict = {}
            nmse_dict0 = {}
            for i, t in nz_post.items():
                if torch.sum(q_posterior[:, i]) > 1e-6:
                    mse = MyMSE(x_hat_all[:, t, :], eval.x)
                    mse_mean = torch.sum(mse * q_posterior[:, i]) / torch.sum(q_posterior[:, i])
                    nmse_dict[t] = 10 * torch.log10(mse_mean / denominator)
            for t in range(args.num_output):
                mse_mean = MyMSE(x_hat_all[:, t, :], eval.x).mean()
                nmse_dict0[t] = 10 * torch.log10(mse_mean / denominator)

        return nmse_dict, nmse_dict0

    def train(self):
        """
        training logic
        :return:
        """
        best_val_loss = None
        progress_bar = tqdm(range(self.epochs))
        for epoch in progress_bar:
            # train
            epoch_log = self._train_epoch(epoch)
            # validation
            epoch_valid_log, val_loss = self._valid_epoch()
            epoch_log = {**epoch_log, **epoch_valid_log}
            log_string = ''
            for key, value in epoch_log.items():
                if key == 'epo':
                    log_string += '%s:%d,' % (key, value)
                elif key == 'det q' or key == 'sto q':
                    log_string += '%s:[' % key
                    log_string += ','.join(['%0.2f' % value[i] for i in self.nz_post])
                    log_string += '],'
                else:
                    log_string += '%s:%0.3f,' % (key, value)

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.score_net.state_dict(), self.args.save_dir + '/best_val_policy.dump')

            log_string += 'best:%0.3f' % best_val_loss
            progress_bar.set_description(log_string)

    def forward_kl_loss(self, x, xhs, scores, p_det=True):
        batch_size = x.shape[0]
        # true posterior
        p_true, mse_all = self.true_posterior(self.args, xhs, x)
        assert batch_size == p_true.shape[0]
        p = torch.stack([p_true[:, t] for t in self.nz_post.values()], dim=1)
        mse = torch.stack([mse_all[:, t] for t in self.nz_post.values()], dim=1)

        if p_det:
            p = max_onehot(p, dim=-1)
        # parameteric log posterior
        log_q_pi = self.log_q_posterior(self.args.policy_type, scores)
        assert batch_size == log_q_pi.shape[0]

        return - torch.sum(p * log_q_pi, dim=-1).mean(), mse

    def backward_kl_loss(self, x, xhs, scores, p_det=True):
        # negative log likelihood
        nlogp, mse_all = self.nll(self.args, xhs, x, keepconst=False)
        neglogp = torch.stack([nlogp[:, t] for t in self.nz_post.values()], dim=1)
        mse = torch.stack([mse_all[:, t] for t in self.nz_post.values()], dim=1)
        n = len(self.nz_post)
        if p_det:
            p = max_onehot(-mse, dim=-1)
            p_soft = (1-p) * 1e-32 + p * (1-n*1e-32)
            neglogp = -p_soft.log()

        # parameteric posterior
        q_pi = self.q_posterior(self.args.policy_type, scores, stochastic=True)

        # entropy
        qlogq = q_pi * (q_pi+1e-32).log()
        # kl
        kl = torch.sum(q_pi * neglogp + qlogq, dim=-1)
        return kl.mean(), mse

    @staticmethod
    def log_q_posterior(type, scores):
        # this type of stop policy is not committed to this github repo
        if type == 'multiclass':
            return F.log_softmax(scores, dim=-1)
        # we only provide the code for sequential stop policy.
        if type == 'sequential':
            batch_size, num_train_post = scores.shape
            pi = F.sigmoid(scores)
            log_q_cont = torch.zeros(batch_size).to(DEVICE)
            log_q_pi = []
            for i in range(num_train_post):
                # prob of stop at t
                log_q_pi.append(((1 - pi[:, i] + 1e-32).log() + log_q_cont).view(-1, 1))
                # prob of continue to next
                log_q_cont += pi[:, i].log()
            # prob of stop at last layer
            log_q_pi.append(log_q_cont.view(-1, 1))
            log_q = torch.cat(log_q_pi, dim=-1)
            return log_q

    @staticmethod
    def q_posterior(type, scores, stochastic=True):
        if type == 'multiclass':
            if stochastic:
                return F.softmax(scores, dim=-1)
            else:
                return max_onehot(scores, dim=-1)
        if type == 'sequential':
            batch_size, num_train_post = scores.shape
            q_pi = []
            pi = F.sigmoid(scores)
            if not stochastic:
                pi = (pi > 0.5).float()
            q_cont = torch.ones(batch_size).to(DEVICE)
            for i in range(num_train_post):
                # prob of stop at t
                q_pi.append(((1 - pi[:, i]) * q_cont).view(-1, 1))
                # prob of continue to next
                q_cont = q_cont * pi[:, i]
            q_pi.append(q_cont.view(-1, 1))
            return torch.cat(q_pi, dim=-1)

    @staticmethod
    def stop_idx(type, scores, stochastic=True):
        if type == 'multiclass':
            if stochastic:
                return F.softmax(scores, dim=-1)
            else:
                return max_onehot(scores, dim=-1)
        if type == 'sequential':
            batch_size, num_train_post = scores.shape
            q_pi = []
            pi = F.sigmoid(scores)
            if not stochastic:
                pi = (pi > 0.5).float()
            else:
                pi = torch.bernoulli(pi)
            q_cont = torch.ones(batch_size).to(DEVICE)
            for i in range(num_train_post):
                # prob of stop at t
                q_pi.append(((1 - pi[:, i]) * q_cont).view(-1, 1))
                # prob of continue to next
                q_cont = q_cont * pi[:, i]
            q_pi.append(q_cont.view(-1, 1))
            return torch.cat(q_pi, dim=-1)

    @staticmethod
    def true_posterior(args, xhs, x):
        mse_all = []
        for tau in range(args.num_output):
            x_hat = xhs[tau]
            mse_all.append(MyMSE(x_hat, x))
        mse_all = torch.stack(mse_all, dim=0)
        return F.softmin(mse_all / args.var / 2, dim=0).t(), mse_all.t()

    @staticmethod
    def nll(args, xhs, x, keepconst=False):
        """
        negative log likelihood
        - log p(y|t,x) = mse / 2 / var + C
        :param keepconst: whether return C or not
        :return  mse / 2 / var if without constant
        """
        mse_all = []
        for tau in range(args.num_output):
            x_hat = xhs[tau]
            mse_all.append(MyMSE(x_hat, x))
        mse_all = torch.stack(mse_all, dim=0)
        if keepconst:
            nllp = mse_all / 2 / args.var + (args.n / 2) * math.log(2 * math.pi * args.var)
        else:
            nllp = mse_all / 2 / args.var
        return nllp.t(), mse_all.t()

