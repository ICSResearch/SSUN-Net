from MODEL.SSUN.net import PS_unfolding as Net
from MODEL.SSUN.net import Loss_Function as Loss_Function
from SOLUTION.fsj_solver import solver
import torch
from CFG import get_cgf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg=get_cgf('GF2')#WV2,WV3
x = solver(cfg,writer=False)
x.loss_function = Loss_Function(s=0.1).cuda()
x.net = Net(ms_channels=4,pan_channels=1,n_feat=32,MaxT=5).cuda()
x.count_parameters()
x.optimizer = torch.optim.Adam(x.net.parameters(), lr=0.0005, betas=cfg['betas'],
                               eps=cfg['epsilon'], weight_decay=cfg['weight_dency'])
x.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(x.optimizer, T_max=50, eta_min=5e-8, last_epoch=-1)
x.load_best_checkpoint()
x.eval_one_epoch()
