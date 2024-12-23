from MODEL.GPPNN.GPPNN import GPPNN as Net
from SOLUTION.fsj_solver import solver
import torch

from CFG import get_cgf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg=get_cgf('GF2')
cfg['data_dir']="your data path"
x = solver(cfg,writer=True)
x.loss_function = torch.nn.L1Loss().cuda()
x.net = Net().cuda()
x.count_parameters()
x.optimizer = torch.optim.Adam(x.net.parameters(), lr=0.0005, betas=cfg['betas'],
                               eps=cfg['epsilon'], weight_decay=cfg['weight_dency'])
x.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(x.optimizer, T_max=50, eta_min=5e-8, last_epoch=-1)
x.run()