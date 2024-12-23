import torch
from DATA.dataset import Data
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import warnings
import random
import numpy as np
from thop import profile
from SOLUTION import mtc
warnings.filterwarnings("ignore")
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def seed_torch(seed=19971118):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class solver():
    def __init__(self, cfg, writer=True):
        self.cfg = cfg
        self.batch_size = cfg['batch_size']
        self.val_batch_size = cfg['val_batch_size']
        self.data_set = cfg['dataset']
        self.net_name = cfg['net_name']
        self.max_epoch = cfg['max_epoch']
        self.gclip = cfg['gclip']
        self.Early_Stop = cfg['Early_Stop']

        self.train_dataset = Data(cfg['data_dir'], self.data_set, is_train=True)
        self.val_dataset = Data(cfg['data_dir'], self.data_set, is_train=False)
        seed_torch(cfg['seed'])
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,pin_memory=True,drop_last=False)
        self.val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=self.val_batch_size,
                                                      shuffle=False,pin_memory=True,drop_last=False)

        self.optimizer = None
        self.net = None
        self.scheduler = None
        self.loss_function = None

        self.epoch = 1
        self.golab_step = 0
        self.best_ssim = 0
        self.best_epoch = 0
        self.best_path = './best_checkpoint/{}_{}_best_.pth'.format(self.net_name, self.data_set)
        self.is_train = True
        if writer and self.is_train:
            self.writer = SummaryWriter(comment='./{}_{}'.format(self.net_name, self.data_set))
        else:
            self.writer = None

    def count_parameters(self, print_=True):
        input1 = torch.randn(1, 4, 32, 32).cuda()
        input2 = torch.randn(1, 1, 128, 128).cuda()
        flops, params = profile(self.net, inputs=(input1, input2))
        print('the flops is {}G \n the params is {}M'.format(round(flops / (10 ** 9), 7),
                                                             round(params / (10 ** 6),
                                                                   7)))

    def save_latest_epoch(self):
        checkpoint = {
            "net": self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "epoch": self.epoch,
            'lr_schedule': self.scheduler.state_dict(),
            'best_ssim': self.best_ssim
        }
        if not os.path.isdir("./save_latest_epoch"):
            os.mkdir("./save_latest_epoch")
        torch.save(checkpoint, './save_latest_epoch/{}_{}_latest_epoch.pth'.format(self.net_name, self.data_set))

    def load_latest_epoch(self):
        path_checkpoint = "./save_latest_epoch/{}_{}_latest_epoch.pth.format(self.net_name,self.data_set)"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        self.net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        self.optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        self.epoch = checkpoint['epoch']  # 设置开始的epoch
        self.scheduler.load_state_dict(checkpoint['lr_schedule'])  # 加载lr_scheduler
        self.best_ssim = checkpoint['best_ssim']


    def train_one_epoch(self):
        self.net.train()
        train_loss = 0
        device = next(self.net.parameters()).device
        for i, (input_lr, input_pan, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.net(input_lr.to(device), input_pan.to(device))
            loss = self.loss_function(output, target.to(device))
            train_loss += loss.item() * input_lr.shape[0]
            loss.backward()
            if self.gclip > 0:
                nn.utils.clip_grad_norm(self.net.parameters(), self.gclip)
            self.optimizer.step()
            self.golab_step += 1

            if (i + 1) % (len(self.train_loader) // 5) == 0:
                print('Epoch : {}/{}  batch_id:{}  Loss : {}'.format(self.epoch, self.max_epoch, i, loss.item()))
        print()
        self.scheduler.step()

        train_loss = train_loss / len(self.train_dataset)
        print('Epoch : {}/{}  Train_Loss : {}  lr:{}'
              .format(self.epoch, self.max_epoch, train_loss,
                      self.optimizer.state_dict()['param_groups'][0]['lr']))
        if self.writer!=None:
            self.writer.add_scalar('train_loss_epoch', train_loss, self.epoch)
            self.writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], self.epoch)


    def eval_one_epoch(self, is_writer=False):
        self.net.eval()
        eval_loss = 0
        self.eval_metrics = ['ERGAS', 'SSIM', 'PSNR', 'SCC']
        self.mtc_results = {}
        for metric in self.eval_metrics:
            self.mtc_results.setdefault(metric, 0)

        device = next(self.net.parameters()).device
        for i, (input_lr, input_pan, target) in enumerate(self.val_loader):
            with torch.no_grad():
                target = target.to(device)
                output = self.net(input_lr.to(device), input_pan.to(device))
                eval_loss += self.loss_function(output, target).item() * input_lr.shape[0]
                self.mtc_results['SSIM'] += mtc.SSIM(target, output)
                self.mtc_results['PSNR']+=mtc.PSNR(target,output)
                self.mtc_results['ERGAS']+=mtc.ERGAS(target,output)
                self.mtc_results['SCC']+=mtc.SCC(target,output)

        eval_loss = eval_loss / len(self.val_dataset)
        print('eval_Loss : ', eval_loss)
        for metric in self.eval_metrics:
            self.mtc_results[metric] = self.mtc_results[metric] / len(self.val_dataset)
            print(metric, ' : ', self.mtc_results[metric])

        if self.writer!=None:
            self.writer.add_scalar('eval_loss', eval_loss, self.epoch)
            for metric in self.eval_metrics:
                self.writer.add_scalar(metric, self.mtc_results[metric], self.epoch)

        return self.mtc_results['SSIM']

    def recoder_best(self,cur_ssim):
        if self.epoch - self.best_epoch > self.Early_Stop:
            print("Model Early Stopping")
        if cur_ssim > self.best_ssim:
            self.best_ssim = cur_ssim
            self.best_epoch = self.epoch
            print('model_save!!best_ssim:epoch is: {}:{}'.format(self.best_ssim,self.best_epoch))
            if not os.path.exists('best_checkpoint'):
                os.mkdir('best_checkpoint')
            torch.save(self.net.state_dict(), self.best_path)
        else:
            pass
        if self.writer != None:
            self.writer.add_scalar('best_epoch', self.best_epoch, self.epoch)

    def load_best_checkpoint(self, path_=None):
        if path_ == None:
            path = self.best_path
        else:
            path = path_
        state_dict = torch.load(path)
        self.net.load_state_dict(state_dict, strict=False)

    def run(self):
        for i in range(self.max_epoch):
            if self.epoch > self.max_epoch:
                break
            self.train_one_epoch()
            cru_ssim=self.eval_one_epoch(is_writer=True)
            self.recoder_best(cru_ssim)
            self.save_latest_epoch()
            self.epoch += 1

