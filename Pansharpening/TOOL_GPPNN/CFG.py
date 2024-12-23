def get_cgf(dataset):
    cfg = {}
    cfg['batch_size']=4
    cfg['val_batch_size']=1
    cfg['dataset']= dataset
    cfg['net_name']='GPPNN'
    cfg['max_epoch']=1000
    cfg['gclip']=40
    cfg['Early_Stop']=100
    cfg['data_dir']="your data path"
    cfg['seed']=123
    cfg['lr']=0.0005
    cfg['betas'] = (0.9, 0.999)
    cfg['gamma'] = 0.85
    cfg['epsilon'] = 1e-8
    cfg['weight_dency'] = 0
    return cfg


