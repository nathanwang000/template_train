import torch, time
import torch.nn as nn
from lib.utils import timeSince, AverageMeter, PrintTable, smooth, to_cuda
import pprint
import numpy as np
import warnings
import shutil
from torch.optim import lr_scheduler

class Train(object):

    def __init__(self, net, data,
                 val_data=None,                 
                 optimizer=None,
                 criterion=None,
                 n_iters=None,
                 n_print=100,
                 n_save=10,
                 n_plot=1000,
                 save_filename='models/checkpoint.pth.tar',
                 best_save_filename=None,
                 lr_decay_factor=0.1,
                 use_gpu=False): # specifies which gpu to use, if False, don't use

        batch_size = data.batch_size # data is data loader
        self.start_iter = 0
        if n_iters is None:
            n_iters = int(100000 / batch_size)

        if optimizer is None:
            optimizer = torch.optim.Adam(net.parameters())

        self.lr_decay_factor = lr_decay_factor
        self.scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        'min', patience=5,
                                                        verbose=True,
                                                        factor=lr_decay_factor)
        
        if criterion is None:
            criterion = nn.MSELoss()

        self.n_iters = n_iters
        self.batch_size = batch_size        
        self.print_every = max(int(n_iters / n_print), 1)
        self.plot_every = max(int(n_iters / n_plot), 1)
        self.save_every = max(int(n_iters / n_save), 1)
        self.use_gpu = use_gpu
            
        self.net = net
        if self.use_gpu is not False:
            self.net.cuda()
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.data = data
        self.val_data = val_data

        self.save_filename = save_filename
        if best_save_filename is None:
            split = self.save_filename.split('.')
            split[0] = split[0] + '_best'
            self.best_save_filename = ".".join(split)

        # logging terms
        self.clear_logs()

    def __repr__(self):
        return pprint.pformat({'net': self.net,
                    'optimizer': self.optimizer,
                    'criterion': self.criterion,
                    'batch_size': self.batch_size,
                    'n_iters': self.n_iters})
        
    def clear_logs(self):
        self.train_losses = []
        self.val_losses = []
        self.best_loss = None

    def save_checkpoint(self, state, is_best):
        torch.save(state, self.save_filename)
        if is_best:
            shutil.copyfile(self.save_filename, self.best_save_filename)

    def load_checkpoint(self, load_filename):
        print("=> loading checkpoint '{}'".format(load_filename))
        checkpoint = torch.load(load_filename)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_iter = checkpoint['niter']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']        
        print("=> loaded checkpoint '{}' (iteration {})"
              .format(load_filename, checkpoint['niter']))

    def smooth_loss(self, step=None):
        if step is None:
            step = self.print_every
        return smooth(np.sqrt(self.train_losses), step=step)

    def smooth_valloss(self, step=None):
        if step is None:
            step = self.print_every
        return smooth(np.sqrt(self.val_losses), step=step)
    
    def train_step(self):
        raise NotImplementedError()

    def val_loss(self):

        if self.val_data is None:
            return None

        print('==> evaluating validation loss')
        self.net.eval()
        loss_meter = AverageMeter()

        for x, y in self.val_data:

            # print(x, y)
            if self.use_gpu is not False:
                x, y = to_cuda(x), to_cuda(y)

            output = self.net(x)
            # regression loss
            loss = self.criterion(output, y)
            if self.use_gpu:
                loss = loss.cpu()
            loss_meter.update(loss.item())

        print('==> validation loss is %.3f' % loss_meter.avg)
        self.scheduler.step(loss_meter.avg)
        self.net.train()
        return loss_meter.avg
        
    def train(self):
        self.net.train()
        losses = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        table_printer = PrintTable()

        table_printer.print(['#iter', 'progress', 'total_time',
                             'batch_time', 'data_time', 'avg_loss'])
        
        start = time.time()
        end = time.time()

        iter = self.start_iter
        while True:
            if iter > self.n_iters or len(self.data.dataset) < self.batch_size: break
            
            for x, y in self.data:

                iter += 1
                if iter > self.n_iters: break

                # measure data loading time
                data_time.update(time.time() - end)

                if self.use_gpu is not False:
                    x, y = to_cuda(x), to_cuda(y)
                output, loss = self.train_step(x, y)

                losses.update(loss, self.batch_size)

                # measure elapsed time
                batch_time.update(time.time() - end)

                #################### note keeping ###########################
                # Print iter number, loss, etc.
                if iter % self.print_every == 0:
                    table_printer.print([iter,
                                         "%d%%" % (iter / self.n_iters * 100),
                                         timeSince(start),
                                         batch_time.avg,
                                         data_time.avg,
                                         losses.avg])

                if iter % self.plot_every == 0:
                    # Add current loss avg to list of losses
                    self.train_losses.append(losses.val)
                    if len(self.val_losses) == 0: 
                        self.val_losses.append(self.val_loss())
                    else:
                        self.val_losses.append(self.val_losses[-1])

                if iter % self.save_every == 0 or iter == self.n_iters:
                    val_loss = self.val_loss()
                    self.val_losses.append(val_loss)
                    is_best = False
                    if val_loss is not None:
                        if self.best_loss is None:
                            self.best_loss = val_loss
                            is_best = True
                        else:
                            is_best = self.best_loss > val_loss                        
                            self.best_loss = min(val_loss, self.best_loss)

                    self.save_checkpoint({
                        'niter': iter + 1,
                        'arch': str(type(self.net)),
                        'state_dict': self.net.state_dict(),
                        'best_loss': self.best_loss,
                        'optimizer': self.optimizer.state_dict(),
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses
                    }, is_best)
                #################### note keeping end ###########################
            
                end = time.time()    


######################################## derived models ################################
class TrainFeedForward(Train):

    def train_step(self, x, y):

        self.optimizer.zero_grad()
        output = self.net(x)
        
        loss = self.criterion(output, y) # regression loss
        loss.backward()
        
        self.optimizer.step()
        return output, loss.item()

