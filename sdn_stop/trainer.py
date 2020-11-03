import torch
from tqdm import tqdm
import torch.nn.functional as F
import os
import numpy as np
from aux_funcs import max_onehot, entropy, get_lr
import data
import pdb

class PolicyKL:
    def __init__(self, args, sdn_model, score_net, train_post, nz_post, optimizer, data_loader, device,
        scheduler,sdn_name=''):
        self.args = args
        self.sdn_model = sdn_model
        self.score_net = score_net
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.sdn_model.eval()
        self.train_data_generator = data_loader.val_loader
        self.epochs = args.num_epochs
        self.train_post = train_post
        self.nz_post = nz_post
        self.device = device
        self.scheduler = scheduler
        self.sdn_name = sdn_name

    def _train_epoch(self, epoch):
        # n outputs, n-1 nets
        self.score_net.train()

        total_loss = 0.0
        for i, batch in enumerate(self.train_data_generator):
            # generate path
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            batch_size = y.shape[0]

            xhs = self.sdn_model(x)
            # internal_fm = self.sdn_model.internal_fm
            # self.sdn_model.internal_fm = [None]*len(internal_fm)
            internal_fm = torch.rand(2,2)

            # if self.args.policy_type == 'sequential':
            #     pi_all = []
            #     for i, t in self.train_post.items():
            #         pi_all.append(self.policy_nets[i](y, xhs[t]).view(-1))
            # if self.args.policy_type == 'multiclass':
            #     pi_all = self.policy_nets(y, xhs)

            scores = self.score_net(x, internal_fm, xhs)

            # scores = self.score_net(x, xhs)

            self.optimizer.zero_grad()
            # loss
            if self.args.kl_type == 'forward':
                loss, _ = self.forward_kl_loss(y, xhs, scores, p_det=False)
            else:
                assert self.args.kl_type == 'backward'
                loss, _ = self.backward_kl_loss(y, xhs, scores)

            # backward
            loss.backward()
            self.optimizer.step()

            if i%self.args.iters_per_eval==0:
                print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, i, loss))

            total_loss += loss.item()

        if epoch==20 or epoch==50 or epoch==70 or epoch==99:
            self.score_net.eval()
            x,y = next(iter(self.train_data_generator))
            x = x.to(self.device)
            y = y.to(self.device)
            batch_size = y.shape[0]

            xhs = self.sdn_model(x)
            internal_fm = torch.rand(2,2)
            scores = self.score_net(x, internal_fm, xhs)

            stop_idx = self.q_posterior(self.args.policy_type, scores, 
                stochastic=False, device=self.device)
            q_p = self.q_posterior(self.args.policy_type, scores, stochastic=True,
                device=self.device)
            q_p_idx = torch.argmax(q_p, dim=-1)
            p_true, _ = self.true_posterior(self.args, xhs, y)
            p_true_b = max_onehot(p_true, dim=-1, device=self.device)
            p_true_idx = torch.argmax(p_true_b, dim=-1)
            print('Here is the policy classification training accuracy:')
            print(data.accuracy(q_p, p_true_idx))
            # p_true_max, _ = self.true_posterior_max(self.args, xhs, y)
            # p_true_max_b = max_onehot(p_true_max, dim=-1, device=self.device)
            # p_true_max_b_idx =  torch.argmax(p_true_max_b, dim=-1)

            # pdb.set_trace()
        log = {
            'epo': epoch,
            'train loss': total_loss / i
        }

        return log

    def _valid_epoch(self, epoch):
        """
        validation after training an epoch
        :return:
        """
        self.score_net.eval()
        
        x, y= next(iter(self.data_loader.test_loader))
        x = x.to(self.device)
        y = y.to(self.device)
        batch_size = y.shape[0]
        with torch.no_grad():
            xhs = self.sdn_model(x)
            # internal_fm = self.sdn_model.internal_fm
            # self.sdn_model.internal_fm = [None]*len(internal_fm)
            internal_fm = torch.rand(2,2)
            scores = self.score_net(x, internal_fm, xhs)
            # scores = self.score_net(x, xhs)
			
            stop_idx = self.q_posterior(self.args.policy_type, scores, 
                stochastic=False, device=self.device)
            q = self.q_posterior(self.args.policy_type, scores, stochastic=True,
                device=self.device)


            if epoch==20 or epoch==50 or epoch==70 or epoch==99:
                stop_idx = self.q_posterior(self.args.policy_type, scores, 
                    stochastic=False, device=self.device)
                q_p = self.q_posterior(self.args.policy_type, scores, stochastic=True,
                    device=self.device)
                q_p_idx = torch.argmax(q_p, dim=-1)
                p_true, _ = self.true_posterior(self.args, xhs, y)
                p_true_b = max_onehot(p_true, dim=-1, device=self.device)
                p_true_idx = torch.argmax(p_true_b, dim=-1)
                print('Here is the policy classification validation accuracy:')
                print(data.accuracy(q_p, p_true_idx))
                # pdb.set_trace()

            # validation loss
            if self.args.kl_type == 'forward':
                loss, _ = self.forward_kl_loss(y, xhs, scores, p_det=False)
            else:
                assert self.args.kl_type == 'backward'
                loss, _ = self.backward_kl_loss(y, xhs, scores)


            if self.args.stochastic:
                log = {
                    'val loss': loss,
                    'sto q': torch.mean(q, dim=0)
                }
            else:
                log = {
                    'val loss': loss,
                    'det q': torch.mean(stop_idx, dim=0)
                }
        return log, log

    @staticmethod
    # The test part need to be refined.
    def test(args, score_net, sdn_model, data_loader, nz_post, device):
        sdn_model.eval()
        score_net.eval()


        predictions = list()
        stops = list()
        b_y = list()
        for i, batch in enumerate(data_loader):
            val_x, val_y = batch
            val_x = val_x.to(device)
            val_y = val_y.to(device)

            with torch.no_grad():
                xhs = sdn_model(val_x)
                # internal_fm = sdn_model.internal_fm
                # sdn_model.internal_fm = [None]*len(internal_fm)

            predictions.append(torch.stack([xhs[t] for t in nz_post.values()]))

            internal_fm = torch.rand(2,2)
            scores = score_net(val_x, internal_fm, xhs)

            stop_idx = PolicyKL.stop_idx(args.policy_type, scores, stochastic=args.stochastic,
                device=device)
            q = PolicyKL.q_posterior(args.policy_type, scores, stochastic=args.stochastic,
                device=device)
            stops.append(stop_idx)
            b_y.append(val_y)
            # p_true, _ = PolicyKL.true_posterior(args, xhs, val_y)
            # p = max_onehot(p_true, dim=-1, device=device)
            # pdb.set_trace()

        stops = torch.cat(stops, axis=0) # num_sample*t
        predictions = torch.cat(predictions, axis=1) # t*samples*num_class
        b_y = torch.cat(b_y, axis=0)

        # get the first 0 in the stops
        index = torch.argmax(stops, axis=-1) # may change it to larger than 0.5
        final_prediction = list()
        for i in range(len(index)):
            final_prediction.append(predictions[index[i], i, :])
    
        pred = torch.stack(final_prediction) # sample*num_class
        prec1, prec5 = data.accuracy(pred, b_y, topk=(1, 5))
        # pdb.set_trace()
        print('Top1 Test accuracy: {}'.format(prec1))
        print('Top5 Test accuracy: {}'.format(prec5))

        # pred = final_prediction.max(1, keepdim=True)[1]
        # is_correct = pred.eq(b_y.view_as(pred))

        # print('Accuracy is {}.'%(is_correct.sum()/len(pred)))


    def train(self):
        """
        training logic
        :return:
        """
        best_val_loss = None
        progress_bar = tqdm(range(self.epochs))
        for epoch in progress_bar:
            self.scheduler.step()
            cur_lr = get_lr(self.optimizer)
            print('\nEpoch: {}/{}'.format(epoch, self.epochs))
            print('Cur lr: {}'.format(cur_lr))

            # train
            epoch_log = self._train_epoch(epoch)
            # validation
            epoch_valid_log, _ = self._valid_epoch(epoch)
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

            #log_string += 'best:%0.3f' % best_val_loss
            progress_bar.set_description(log_string)
            PolicyKL.test(args=self.args,
                          score_net=self.score_net,
                          sdn_model=self.sdn_model,
                          data_loader=self.data_loader.test_loader,
                          nz_post=self.nz_post,
                          device=self.device
                          )
            if epoch%10==0 and epoch>0:
                print('This is the performance on the train dataset:')
                PolicyKL.test(args=self.args,
                              score_net=self.score_net,
                              sdn_model=self.sdn_model,
                              data_loader=self.train_data_generator,
                              nz_post=self.nz_post,
                              device=self.device
                              )                

            torch.save(self.score_net.state_dict(), self.args.save_dir + '/{}_best_val_policy.dump'.format(self.sdn_name))

    def forward_kl_loss(self, y, xhs, scores, p_det=True):
        batch_size = y.shape[0]
        # true posterior
        p_true, _ = self.true_posterior(self.args, xhs, y)
        assert batch_size == p_true.shape[0]
        p = torch.stack([p_true[:, t] for t in self.nz_post.values()], dim=1)
        # mse = torch.stack([mse_all[:, t] for t in self.nz_post.values()], dim=1)

        if p_det:
            p = max_onehot(p, dim=-1, device=self.device)
        # parameteric log posterior
        log_q_pi = self.log_q_posterior(self.args.policy_type, scores)
        assert batch_size == log_q_pi.shape[0]
        # pdb.set_trace()
        return -torch.sum(p * log_q_pi, dim=-1).mean(), -torch.sum(p * log_q_pi, dim=-1).mean()

    def backward_kl_loss(self, y, xhs, scores):
        # negative log likelihood
        nlogp, _ = self.nll(self.args, xhs, y)
        neglogp = torch.stack([nlogp[:, t] for t in self.nz_post.values()], dim=1)
        # mse = torch.stack([mse_all[:, t] for t in self.nz_post.values()], dim=1)

        # parameteric posterior
        q_pi = self.q_posterior(self.args.policy_type, scores, stochastic=True)

        #####################
        ##### entropy, is there a problem with the sign?
        ##### why is there a MSE?
        #####################
        qlogq = q_pi * (q_pi+1e-32).log()
        # kl
        kl = torch.sum(q_pi * neglogp + qlogq, dim=-1)
        return kl.mean(), kl.mean()

    @staticmethod
    def log_q_posterior(type, scores, device='cuda'):
        if (type == 'multiclass') or (type == 'confidence'):
            return F.log_softmax(scores, dim=-1)
        if type == 'sequential':
            batch_size, num_train_post = scores.shape
            pi = F.sigmoid(scores)
            log_q_cont = torch.zeros(batch_size).to(device)
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
    def q_posterior(type, scores, stochastic=True, device='cuda'):
        if (type == 'multiclass') or (type == 'confidence'):
            if stochastic:
                return F.softmax(scores, dim=-1)
            else:
                return max_onehot(scores, dim=-1, device=device)
        if type == 'sequential':
            batch_size, num_train_post = scores.shape
            q_pi = []
            pi = F.sigmoid(scores)
            if not stochastic:
                pi = (pi > 0.5).float()
            q_cont = torch.ones(batch_size).to(device)
            for i in range(num_train_post):
                # prob of stop at t
                q_pi.append(((1 - pi[:, i]) * q_cont).view(-1, 1))
                # prob of continue to next
                q_cont = q_cont * pi[:, i]
            q_pi.append(q_cont.view(-1, 1))
            return torch.cat(q_pi, dim=-1)


    @staticmethod
    def stop_idx(type, scores, stochastic=True, device='cuda'):
        if (type == 'multiclass') or (type == 'confidence'):
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
            q_cont = torch.ones(batch_size).to(device)
            for i in range(num_train_post):
                # prob of stop at t
                q_pi.append(((1 - pi[:, i]) * q_cont).view(-1, 1))
                # prob of continue to next
                q_cont = q_cont * pi[:, i]
            q_pi.append(q_cont.view(-1, 1))
            return torch.cat(q_pi, dim=-1)



    # @staticmethod
    # def true_posterior(args, xhs, y):
    #     '''
    #     xhs: the raw score, not yet softmax, a list of batch*num_class
    #     y: the ground truth label
    #     '''
    #     softmaxed = []
    #     for score in xhs:
    #         softmaxed.append(F.softmax(score, dim=-1).gather(1, y.long().view(-1, 1)))
    #         # print(F.softmax(score, dim=-1))
    #         # print(softmaxed)
    #     softmaxed = torch.cat(softmaxed, dim=1) # batch*t
    #     # re-normalize
    #     p = softmaxed/softmaxed.sum(1).view(-1,1)
    #     return p, p

    @staticmethod
    def true_posterior(args, xhs, y):
        '''
        do not care about y, only care about the max confidence score of any class
        '''
        softmaxed = []
        for score in xhs:
            softmaxed.append(torch.max(F.softmax(score, dim=-1), dim=-1)[0])
        softmaxed = torch.stack(softmaxed, dim=1) # batch*t
        # pdb.set_trace()
        # re-normalize
        p = softmaxed/softmaxed.sum(1).view(-1,1)
        return p, p

    @staticmethod
    def nll(args, xhs, y):
        """
        negative log likelihood
        - log p(y|t,x)
        xhs: the raw score in a list
        y: label 
        return: batch*t
        """
        softmaxed = []
        for score in xhs:
            softmaxed.append(F.softmax(score, dim=-1).gather(1, y.long().view(-1, 1)))
        softmaxed = torch.cat(softmaxed, dim=1) # batch*t
        nll = -softmaxed.log()
        return nll, nll


class JointTrain:
    # auto-encoding variational bayes
    def __init__(self, args, model, score_net, train_post, nz_post, optimizer1, optimizer2, train_loader, 
        device, test_loader, sdn_name=''):
        self.args = args
        self.model = model
        self.score_net = score_net
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.epochs = args.num_epochs
        self.train_post = train_post
        self.nz_post = nz_post
        self.device = device
        self.sdn_name = sdn_name

    def _train_epoch(self, epoch):

        for i, batch in enumerate(self.train_loader):
            # generate path
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            model_loss = self._update_model(x, y)
            net_loss = 0
            # net_loss = self._update_policy(x, y)

            if i%self.args.iters_per_eval==0:
                print('Epoch: {}, Step: {}, Model loss: {}, Net loss: {}'.format(
                    epoch, i, model_loss, net_loss))

        log = {
            'epo': epoch,
        }

        return log

    def _update_model(self, x, y):
        self.model.train()
        self.score_net.eval()
        # forward
        self.optimizer1.zero_grad()
        xhs = self.model(x)
        internal_fm = torch.rand(2,2)
        scores = self.score_net(x, internal_fm, xhs)
        q_posterior = PolicyKL.q_posterior(self.args.policy_type, scores, stochastic=True)

        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        loss_t = []
        for t in self.nz_post.values():
            x_hat = xhs[t]
            loss_t.append(criterion(x_hat, y).unsqueeze(-1))
        
        loss_t = torch.cat(loss_t, -1) # batch * T
        loss = torch.sum(loss_t * q_posterior.detach(), dim=-1).mean()
        loss.backward()
        self.optimizer1.step()
        # pl = list(self.model.parameters())
        # pdb.set_trace()
        return loss

    def _update_policy(self, x, y):
        self.model.eval()
        self.score_net.train()

        # pdb.set_trace()
        xhs = self.model(x)
        internal_fm = torch.rand(2,2)
        scores = self.score_net(x, internal_fm, xhs)

        self.optimizer2.zero_grad()
        # loss
        # true posterior
        p_true, _ = PolicyKL.true_posterior(self.args, xhs, y)
        p = torch.stack([p_true[:, t] for t in self.nz_post.values()], dim=1)
        p = max_onehot(p, dim=-1)
        # parameteric log posterior
        log_q_pi = PolicyKL.log_q_posterior(self.args.policy_type, scores)
        loss = - torch.sum(p * log_q_pi, dim=-1).mean()

        # backward
        loss.backward()
        self.optimizer2.step()
        return loss

    def _valid_epoch(self):
        """
        validation after training an epoch
        :return:
        """
        PolicyKL.test(args=self.args,
                      score_net=self.score_net,
                      sdn_model=self.model,
                      data_loader=self.test_loader,
                      nz_post=self.nz_post,
                      device=self.device
                      )

    def train(self):
        """
        training logic
        :return:
        """
        for epoch in range(self.epochs):
            # train
            epoch_log = self._train_epoch(epoch)
            self._valid_epoch()
            if epoch%10==0 and epoch>0:
                print('This is the performance on the train dataset:')
                PolicyKL.test(args=self.args,
                              score_net=self.score_net,
                              sdn_model=self.model,
                              data_loader=self.train_loader,
                              nz_post=self.nz_post,
                              device=self.device
                              ) 
            torch.save(self.score_net.state_dict(), 
                os.path.join(self.args.save_dir, '{}_policy_net_joint.dump'.format(self.sdn_name)))
            torch.save(self.model.state_dict(), 
                os.path.join(self.args.save_dir, '{}_net_joint.pth'.format(self.sdn_name)))


class PolicyTrainer:
    def __init__(self, batch_size, num_epochs, iters_per_eval, num_output, save_dir,
        t, sdn_net, policy_t, policy_all, optimizer, data_loader, DEVICE,
        stochastic=True, first_loop=False):
        """
        :param t: Now training the t-th stopping policy, pi_t. (t counts from 1 to num_output, not start from 0)
        :param sdn_net:
        :param policy_t: the current training policy
        :param policy_all: all policy
        :param optimizer:
        :param data_loader:
        :param stochastic:
        :param first_loop:
        """

        self.sdn_net = sdn_net
        self.sdn_net.eval()
        self.policy_t = policy_t
        self.optimizer = optimizer
        self.train_data_generator = data_loader.train_loader
        self.data_loader = data_loader
        self.num_output = num_output
        self.epochs = num_epochs
        self.first_loop = first_loop
        # for checkpoint recovery
        self.start_epoch = 0
        self.policy_all = policy_all
        self.t = t
        self.stochastic = stochastic
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.iters_per_eval = iters_per_eval
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.device = DEVICE


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains information about training
        """
        self.policy_t.train()
        total_loss = 0.0
        cont_size = 0.0
        total_batch = 0.0
        gt_policy_all = list()
        for i, batch in enumerate(self.train_data_generator):
            # get a batch training data
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            batch_size = y.shape[0]
            # generate path
            xhs = self.sdn_net(x)
            # print(len(xhs))
            # stop flag
            continue_idx = torch.ones(batch_size).to(self.device)
            # print("The initialized idx shape: ", continue_idx.shape)
            for tau in range(self.t-1):
                self.policy_all[tau].eval()
                p = self.policy_all[tau](x, xhs[tau]).reshape(-1)
                if self.stochastic:
                    pi = torch.bernoulli(p).to(self.device)
                else:
                    pi = (p > 0.5).float()
                continue_idx = (continue_idx * pi).detach()
            batch_size = torch.sum(continue_idx)
            cont_size += batch_size
            if batch_size == 0:
                print('!!All paths are stopped in this batch!!')
                continue
            # optimal loss after t
            loss_after_t = []
            for tau in range(self.t, self.num_output):
                x_hat = xhs[tau]
                loss_after_t.append(self.criterion(x_hat, y).unsqueeze(-1))
            loss_best, _ = torch.min(torch.cat(loss_after_t, dim=-1), dim=-1)

            # loss at t
            loss_t = self.criterion(xhs[self.t-1], y)

            # check the ground truth policy
            loss_all = torch.cat([loss_t.unsqueeze(-1)]+loss_after_t, dim=-1)
            gt_policy = (torch.argmin(loss_all, dim=-1) > (self.t-1)).to(torch.float32)
            # print(torch.argmin(loss_all, dim=-1))
            gt_policy_all.append(torch.argmin(loss_all, dim=-1).cpu().numpy())

            # policy at t
            p_t = self.policy_t(x, xhs[self.t-1]).reshape(-1)

            # forward
            self.optimizer.zero_grad()
            # loss
            loss = loss_t.detach() * (1 - p_t) + loss_best.detach() * p_t
            # print('Pt shape', p_t.shape)
            # print('Loss t dimension: ', loss_t.shape)
            # print('Loss best dimension:, ', loss_best.shape)
            # print("The loss dimension: ", loss.shape)
            # print("The idx dimension: ", continue_idx.shape)
            loss = torch.sum(loss * continue_idx) / batch_size

            # print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, i, loss))

            # backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_batch += 1

        log = {
            'policy': self.t,
            'epoch': epoch,
            'avg loss in this epoch': total_loss / total_batch,
            'batch size': batch_size,
            'cont size': cont_size / total_batch
        }
        # gt_policy_all = np.concatenate(gt_policy_all, axis=None).flatten()
        # np.save('gt_policy_train.npy', gt_policy_all)
        return log


    # validation epoch need to be tunned
    def _valid_epoch(self):
        """
        Validate after training an epoch
        """
        self.policy_t.eval()
        total_loss = 0.0
        total_batch = 0.0

        gt_policy_all = list()

        for i, batch in enumerate(self.data_loader.test_loader):
            val_x, val_y = batch
            val_x = val_x.to(self.device)
            val_y = val_y.to(self.device)
            # error
            batch_size = val_y.shape[0]
            with torch.no_grad():
                xhs = self.sdn_net(val_x.to(self.device))
                # stop flag
                # while 1:
                continue_idx = torch.ones(batch_size).to(self.device)
                for tau in range(self.t-1):
                    self.policy_all[tau].eval()
                    p = self.policy_all[tau](val_x, xhs[tau]).reshape(-1)
                    if self.stochastic:
                        pi = torch.bernoulli(p).to(self.device)
                    else:
                        pi = (p > 0.5).float()
                    continue_idx = (continue_idx * pi).detach()
                cont_size = torch.sum(continue_idx)
                if cont_size < 1:
                    continue

                # optimal loss after t
                loss_after_t = []
                for tau in range(self.t, self.num_output):
                    x_hat = xhs[tau]
                    loss_after_t.append(self.criterion(x_hat, val_y).unsqueeze(-1))
                loss_best, _ = torch.min(torch.cat(loss_after_t, dim=-1), dim=-1)
                # validation loss
                loss_t = self.criterion(xhs[self.t-1], val_y)



                loss_all = torch.cat([loss_t.unsqueeze(-1)]+loss_after_t, dim=-1)
                gt_policy = (torch.argmin(loss_all, dim=-1) > (self.t-1)).to(torch.float32)
                # print(torch.argmin(loss_all, dim=-1))
                gt_policy_all.append(torch.argmin(loss_all, dim=-1).cpu().numpy())


                # policy at t
                p_t = self.policy_t(val_x, xhs[self.t-1]).reshape(-1)

                if self.stochastic:
                    pi_t = torch.bernoulli(p_t)
                else:
                    pi_t = (p_t > 0.5).float()
                loss = loss_t * (1 - pi_t) + loss_best * pi_t
                loss = torch.sum(loss * continue_idx) / cont_size
                total_loss += loss
                total_batch += 1

        log = {
            'val loss': total_loss/total_batch,
        }
        # gt_policy_all = np.concatenate(gt_policy_all, axis=None).flatten()
        # np.save('gt_policy_test.npy', gt_policy_all)
        return log, total_loss/total_batch

    def train(self):
        """
        training logic
        :return:
        """
        flag = 0
        progress_bar = tqdm(range(self.epochs))
        for epoch in progress_bar:
        # for epoch in range(self.epochs):
            epoch_log = self._train_epoch(epoch)

            epoch_valid_log, val_loss = self._valid_epoch()

            epoch_log = {**epoch_log, **epoch_valid_log}

            log_string = ''
            for key, value in epoch_log.items():
                log_string += '  %s : %0.5f' % (key, value)
            progress_bar.set_description(log_string)

            torch.save(self.policy_t.state_dict(), self.save_dir + '/policy_%d.dump' % self.t)
        return flag

