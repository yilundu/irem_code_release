import torch
from tqdm import tqdm
import torch.nn.functional as F
import os
import numpy as np
from utils import *
import data
import pdb
from torchvision.utils import save_image

class PolicyKL:
    def __init__(self, args, model, score_net, train_post, nz_post, optimizer, train_loader, 
        device, test_loader, val_loader, scheduler):
        self.args = args
        self.model = model
        self.score_net = score_net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.epochs = args.num_epochs
        self.train_post = train_post
        self.nz_post = nz_post
        self.device = device
        self.scheduler = scheduler
        self.model.eval()

    def _train_epoch(self, epoch):
        # n outputs, n-1 nets
        self.score_net.train()
        noiseset = [35, 45, 55]

        total_loss = 0.0
        for i, batch in enumerate(self.train_loader):
            # generate path
            data = batch
            data = data.to(self.device)

            noise = torch.zeros(data.size())
            stdN = np.random.choice(noiseset, size=noise.size()[0])
            for n in range(noise.size()[0]):
                sizeN = noise[0,:,:,:].size()
                noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            noise = noise.cuda()

            xhs = self.model(data+noise)
            scores = self.score_net(data+noise, xhs)

            # stop_idx = max_onehot(scores, dim=-1, device=self.device)
            # pred_idx = torch.argmax(stop_idx, dim=-1)
            # p_true, _ = self.true_posterior(self.args, xhs, noise)
            # p_true = torch.stack([p_true[:, t] for t in self.nz_post.values()], dim=1)
            # true_idx = max_onehot(p_true, dim=-1, device=self.device)
            # true_idx = torch.argmax(true_idx, dim=-1)
            # pdb.set_trace()

            self.optimizer.zero_grad()
            # loss
            if self.args.kl_type == 'forward':
                loss, _ = self.forward_kl_loss(noise, xhs, scores, p_det=True)
            else:
                assert self.args.kl_type == 'backward'
                loss, _ = self.backward_kl_loss(noise, xhs, scores)

            # backward
            loss.backward()
            self.optimizer.step()

            if i%self.args.iters_per_eval==0:
                q = self.q_posterior(self.args.policy_type, scores, stochastic=True,
                    device=self.device)
                print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, i, loss))
                print(torch.mean(q, dim=0).detach().cpu().numpy())

            total_loss += loss.item()

        log = {
            'epo': epoch,
            'train loss': total_loss / i
        }

        return log

    def _valid_epoch(self):
        """
        validation after training an epoch
        :return:
        """
        noiseset = [35, 45, 55]
        self.score_net.eval()
        loss_all = 0
        stop_true = list()
        stop_pred = list()
        q_all = list()
        # check over all
        for i, batch in enumerate(self.val_loader):
            data= batch
            data = data.to(self.device)
            with torch.no_grad():
                noise = torch.zeros(data.size())
                stdN = np.random.choice(noiseset, size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
                noise = noise.cuda()
                xhs = self.model(data+noise)
                scores = self.score_net(data+noise, xhs)
    			
                stop_idx = self.q_posterior(self.args.policy_type, scores, 
                    stochastic=False, device=self.device)
                q = self.q_posterior(self.args.policy_type, scores, stochastic=True,
                    device=self.device)
                stop_pred.append(stop_idx)
                q_all.append(q)

                p_true, _ = self.true_posterior(self.args, xhs, noise)
                p = max_onehot(p_true, dim=-1, device=self.device)
                stop_true.append(p)
                # validation loss
                if self.args.kl_type == 'forward':
                    loss, _ = self.forward_kl_loss(noise, xhs, scores, p_det=True)
                else:
                    assert self.args.kl_type == 'backward'
                    loss, _ = self.backward_kl_loss(noise, xhs, scores)
                loss_all += loss
        

        # pdb.set_trace()

        if self.args.stochastic:
            log = {
                'val loss': loss_all/i,
                'sto q': torch.mean(torch.cat(q_all, dim=0), dim=0)
            }
        else:
            log = {
                'val loss': loss_all/i,
                'det q': torch.mean(torch.cat(stop_pred, dim=0), dim=0)
            }
        return log, log

    @staticmethod
    # need to check each image one by one
    # the size of each image may be different
    def test(args, score_net, model, data_loader, nz_post, device, noiseset=[35, 45, 55, 65, 75], noiseL_B=[0,75]):
        model.eval()
        score_net.eval()

        np.random.seed(seed=args.seed)
        test_noiseL = np.random.choice(noiseset, size=len(data_loader.dataset))
        # print(test_noiseL)
        print('Average noise level: ', np.average(test_noiseL))

        predictions = list()
        stops = list()
        b_y = list()
        imgns = list()
        psnrs = list()
        for i, batch in enumerate(data_loader):
            data = batch
            data = data.to(device)

            noise = torch.FloatTensor(data.size()).normal_(mean=0, 
                std=test_noiseL[i]/255., generator=torch.manual_seed(args.seed))
            noise = noise.cuda()

            with torch.no_grad():
                imgn = data+noise
                xhs = model(imgn)

            
            scores = score_net(imgn, xhs)
            stop_idx = PolicyKL.stop_idx(args.policy_type, scores, stochastic=False,
                device=device)
            q = PolicyKL.q_posterior(args.policy_type, scores, stochastic=True,
                device=device)

            index = torch.argmax(stop_idx, axis=-1)
            # pdb.set_trace()
            prediction = xhs[nz_post[index.cpu().numpy()[0]]]
            
            psnr = batch_PSNR(torch.clamp(imgn-prediction, 0., 1.), data, 1.)
            psnrs.append(psnr)

        # print(psnrs)
        print('The test PSNR is ', np.average(psnrs))


        #     predictions.append(torch.stack(xhs))
        #     stops.append(stop_idx)
        #     b_y.append(data)
        #     imgns.append(imgn)

        # stops = torch.cat(stops, axis=0) # num_sample*t
        # predictions = torch.cat(predictions, axis=1) # t*samples*pred_n
        # imgns = torch.cat(imgns, axis=0)
        # b_y = torch.cat(b_y, axis=0)

        # # get the first 0 in the stops
        # index = torch.argmax(stops, axis=-1) # may change it to larger than 0.5
        # final_prediction = list()
        # for i in range(len(index)):
        #     final_prediction.append(predictions[index[i], i, :, :, :])
    
        # pred = torch.stack(final_prediction) # 4D tensor
        # psnr = batch_PSNR(imgns-pred, b_y, 1.)
        return np.average(psnrs)
        
    def save_imgs(args, score_net, model, data_loader, nz_post, device, folder,
            noiseset=[35, 45, 55, 65, 75], noiseL_B=[0,75]):
        if not os.path.exists(folder):
            os.makedirs(folder)
        model.eval()
        score_net.eval()
        np.random.seed(seed=args.seed)
        test_noiseL = np.random.choice(noiseset, size=len(data_loader.dataset))
        print('Average noise level: ', np.average(test_noiseL))
        predictions = list()
        stops = list()
        b_y = list()
        imgns = list()
        psnrs = list()
        img_pred = list()
        for i, batch in enumerate(data_loader):
            data = batch
            data = data.to(device)
            noise = torch.FloatTensor(data.size()).normal_(mean=0, 
                std=test_noiseL[i]/255., generator=torch.manual_seed(args.seed))
            noise = noise.cuda()
            with torch.no_grad():
                imgn = data+noise
                xhs = model(imgn)
            scores = score_net(imgn, xhs)
            stop_idx = PolicyKL.stop_idx(args.policy_type, scores, stochastic=False,
                device=device)
            q = PolicyKL.q_posterior(args.policy_type, scores, stochastic=True,
                device=device)

            index = torch.argmax(stop_idx, axis=-1)
            # pdb.set_trace()
            prediction = xhs[nz_post[index.cpu().numpy()[0]]]
            pred = torch.clamp(imgn-prediction, 0., 1.)
            psnr = batch_PSNR(pred, data, 1.)
            psnrs.append(psnr)
            # b_y.append(data)
            # imgns.append(imgn)
            # img_pred.append(pred)
            # pdb.set_trace()
            save_image(data[0], os.path.join(folder, '{}_raw.png'.format(i)))
            save_image(imgn[0], os.path.join(folder, '{}_imgn.png'.format(i)))
            save_image(pred[0], os.path.join(folder, '{}_pred.png'.format(i)))
        print('The test PSNR is ', np.average(psnrs))
        np.save(os.path.join(folder,'psnr.npy'), np.array(psnrs))


    def train(self):
        """
        training logic
        :return:
        """
        best_val_psnr = 0
        for epoch in range(self.epochs):
            self.scheduler.step()
            cur_lr = get_lr(self.optimizer)
            print('\nEpoch: {}/{}'.format(epoch, self.epochs))
            print('Cur lr: {}'.format(cur_lr))
            
            # train
            epoch_log = self._train_epoch(epoch)
            # validation
            epoch_valid_log, _ = self._valid_epoch()

            psnr_val = PolicyKL.test(args=self.args,
                        score_net=self.score_net,
                        model=self.model,
                        data_loader=self.val_loader,
                        nz_post=self.nz_post,
                        device=self.device, 
                        noiseset=[35, 45, 55]
                        )

            psnr_test = PolicyKL.test(args=self.args,
                          score_net=self.score_net,
                          model=self.model,
                          data_loader=self.test_loader,
                          nz_post=self.nz_post,
                          device=self.device,
                          noiseset=[35, 45, 55]
                          )
            # if epoch%10==0 and epoch>0:
            #     print('This is the performance on the train dataset:')
            #     PolicyKL.test(args=self.args,
            #                   score_net=self.score_net,
            #                   model=self.model,
            #                   data_loader=self.train_loader,
            #                   nz_post=self.nz_post,
            #                   device=self.device
            #                   )                
            if psnr_test>best_val_psnr:
                best_val_psnr = psnr_test
                torch.save(self.score_net.state_dict(), 
                    os.path.join(self.args.outf, '{}_policy_net.dump'.format(self.args.policy_type)))

    def forward_kl_loss(self, y, xhs, scores, p_det=True):
        batch_size = y.shape[0]
        # true posterior
        p_true, mse_all = self.true_posterior(self.args, xhs, y)
        assert batch_size == p_true.shape[0]
        p = torch.stack([p_true[:, t] for t in self.nz_post.values()], dim=1)
        mse = torch.stack([mse_all[:, t] for t in self.nz_post.values()], dim=1)

        if p_det:
            p = max_onehot(p, dim=-1, device=self.device)
        # parameteric log posterior
        log_q_pi = self.log_q_posterior(self.args.policy_type, scores)
        assert batch_size == log_q_pi.shape[0]
        # pdb.set_trace()
        return -torch.sum(p * log_q_pi, dim=-1).mean(), mse

    def backward_kl_loss(self, y, xhs, scores):
        # negative log likelihood
        nlogp, mse = self.nll(self.args, xhs, y)
        neglogp = torch.stack([nlogp[:, t] for t in self.nz_post.values()], dim=1)
        mse = torch.stack([mse_all[:, t] for t in self.nz_post.values()], dim=1)

        # parameteric posterior
        q_pi = self.q_posterior(self.args.policy_type, scores, stochastic=True)

        #####################
        ##### entropy, is there a problem with the sign?
        ##### why is there a MSE?
        #####################
        qlogq = q_pi * (q_pi+1e-32).log()
        # kl
        kl = torch.sum(q_pi * neglogp + qlogq, dim=-1)
        return kl.mean(), mse

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



    @staticmethod
    def true_posterior(args, xhs, y):
        '''
        xhs: the raw score, not yet softmax, a list of batch*num_class
        y: the ground truth label
        '''
        mse_all = []
        for output in xhs:
            mse_all.append(mse_per_sample(output, y))
        mse_all = torch.stack(mse_all, dim=1)
        return F.softmin(mse_all, dim=1), mse_all

    @staticmethod
    def nll(args, xhs, y):
        """
        negative log likelihood
        - log p(y|t,x)
        xhs: the raw score in a list
        y: label 
        return: not quite sure, just use mse
        """
        mse_all = []
        for output in xhs:
            mse_all.append(mse_per_sample(output, y))
        mse_all = torch.stack(mse_all, dim=1)
        return mse_all, mse_all


class JointTrain:
    # auto-encoding variational bayes
    def __init__(self, args, model, score_net, train_post, nz_post, optimizer1, optimizer2, train_loader, 
        device, test_loader, val_loader):
        self.args = args
        self.model = model
        self.score_net = score_net
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = args.num_epochs
        self.train_post = train_post
        self.nz_post = nz_post
        self.device = device

    def _train_epoch(self, epoch):
        noiseset = [35, 45, 55]
        for i, batch in enumerate(self.train_loader):
            # generate path
            data = batch
            data = data.to(self.device)

            noise = torch.zeros(data.size())
            stdN = np.random.choice(noiseset, size=noise.size()[0])
            for n in range(noise.size()[0]):
                sizeN = noise[0,:,:,:].size()
                noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            noise = noise.cuda()
            imgn = data+noise
            model_loss = self._update_model(imgn, noise)
            net_loss = self._update_policy(imgn, noise)

            if i%self.args.iters_per_eval==0:
                print('Epoch: {}, Step: {}, Model loss: {}, Net loss: {}'.format(
                    epoch, i, model_loss, net_loss))

        # pdb.set_trace()
        log = {
            'epo': epoch,
        }

        return log

    def _update_model(self, imgn, noise):
        self.model.train()
        self.score_net.eval()
        # forward
        self.optimizer1.zero_grad()
        xhs = self.model(imgn)
        scores = self.score_net(imgn, xhs)
        q_posterior = PolicyKL.q_posterior(self.args.policy_type, scores, stochastic=True)
        ll_t = []  # neg likelihood
        for t in self.nz_post.values():
            x_hat = xhs[t]
            ll_t.append(mse_per_sample(x_hat, noise))
        ll_all = torch.stack(ll_t, dim=0).t()
        loss = torch.sum(torch.sum(ll_all * q_posterior, dim=-1), dim=-1)/(imgn.size()[0]*2)
        # loss = torch.sum(ll_all * q_posterior, dim=-1).mean()
        loss.backward()
        self.optimizer1.step()
        return loss

    def _update_policy(self, imgn, noise):
        self.model.eval()
        self.score_net.train()

        xhs = self.model(imgn)
        scores = self.score_net(imgn, xhs)

        self.optimizer2.zero_grad()
        # loss
        # true posterior
        p_true, _ = PolicyKL.true_posterior(self.args, xhs, noise)
        p = torch.stack([p_true[:, t] for t in self.nz_post.values()], dim=1)
        p = max_onehot(p, dim=-1)
        # parameteric log posterior
        log_q_pi = PolicyKL.log_q_posterior(self.args.policy_type, scores)
        loss = - torch.sum(p * log_q_pi, dim=-1).mean()

        # backward
        loss.backward()
        self.optimizer2.step()
        return loss

    def _valid_epoch(self, data_loader):
        """
        validation after training an epoch
        :return:
        """
        psnr = PolicyKL.test(args=self.args,
                    score_net=self.score_net,
                    model=self.model,
                    data_loader=data_loader,
                    nz_post=self.nz_post,
                    device=self.device)
        return psnr

    def train(self):
        """
        training logic
        :return:
        """
        best_val_psnr = 0
        for epoch in range(self.epochs):
            # train
            epoch_log = self._train_epoch(epoch)
            psnr_val = self._valid_epoch(self.val_loader)
            print('Val PSNR: ', psnr_val)
            psnr_test = self._valid_epoch(self.test_loader)
            print('Test PSNR: ', psnr_test)
            if psnr_val>best_val_psnr:
                best_val_psnr = psnr_val
                torch.save(self.score_net.state_dict(), 
                    os.path.join(self.args.outf, '{}_policy_net_joint.dump'.format(self.args.policy_type)))
                torch.save(self.model.state_dict(), 
                    os.path.join(self.args.outf, '{}_net_joint.pth'.format(self.args.policy_type)))

            