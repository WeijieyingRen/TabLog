import os
import numpy as np
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.model_selection import KFold

from rrl.utils import read_csv, DBEncoder, read_info, hospital_read_csv,college_scorecard_read_csv,ASSISTments_read_csv
from rrl.models import RRL
import pdb
import csv
import torch.optim as optim

import tent


# hospital need specific read_csv: hospital_read_csv(data_path,shuffle=False) 
#DATA_DIR = '/storage/work/wjr5337/code/ICML24/Hospital_Readmission/'
DATA_DIR = '/storage/work/wjr5337/code/ICML24/ASSISTments/'
# college scorecard would be a little of difficult
#DATA_DIR = '/storage/work/wjr5337/code/ICML24/college_scorecard/'
#DATA_DIR = '/storage/work/wjr5337/code/ICML24/Sepsis/'

import pdb

# world_size GPU
# rank id: which GPU
def get_data_loader(train_data, train_label, world_size, rank, batch_size, data_split_ratio=0.95, pin_memory=False, save_best=True, num_bins = 5):
    
    data_path = os.path.join(DATA_DIR, train_data + '.csv')
    label_path = os.path.join(DATA_DIR, train_label + '.csv')
    info_path = os.path.join(DATA_DIR,  'column_name.info')
    #X_df = hospital_read_csv(data_path,shuffle=False) 
    #X_df = read_csv(data_path,shuffle=False) 
    #X_df = college_scorecard_read_csv(data_path,shuffle=False) 
    X_df = ASSISTments_read_csv(data_path,shuffle=False) 
    X_df = X_df.drop(X_df.columns[0], axis=1)
    
    y_df = read_csv(label_path,shuffle=False)
    y_df = y_df.drop(y_df.columns[0], axis=1)
    
    num_per_class = []
    
    for i in range(y_df.shape[1]+1):
        total_count = (y_df == i).sum().sum()
        num_per_class.append(total_count)
    
    f_df = read_info(info_path)
    
    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_df, y_df)

    X_data, y_data,cell_boundary_index, cell_interval_index, continuous_column_index,discrete_flen,continuous_flen = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True,num_bins= 5)
    # X_data.shape (98556, 120)
    #data_set = TensorDataset(torch.tensor(X_data.astype(np.float32)), torch.tensor(y_data.astype(np.float32)))
    data_set = TensorDataset(torch.tensor(X_data.astype(np.float32)), torch.tensor(cell_interval_index.values.astype(np.float32)),torch.tensor(y_data.astype(np.float32)))
    train_len = int(len(data_set) * data_split_ratio)
    train_sub, valid_sub = random_split(data_set, [train_len, len(data_set) - train_len])
    if not save_best:  # use all the training set for training, and no validation set used for model selections.
        train_sub = data_set

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_sub, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, sampler=train_sampler)
    valid_loader = DataLoader(valid_sub, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
 
    print ('In the {}-phase, the size of feature is {}:'.format(train_data,X_data.shape))
    return db_enc, train_loader, valid_loader,num_per_class,cell_boundary_index,continuous_column_index,discrete_flen,continuous_flen


def train_model(gpu, args):
    #rank = args.nr * args.gpus + gpu
    rank = 0
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(42)
    device_id = args.device_ids[gpu]
    torch.cuda.set_device(device_id)

    if gpu == 0:
        writer = SummaryWriter(args.folder_path)
        is_rank0 = True
    else:
        writer = None
        is_rank0 = False

    train_dataset = args.train_data_set
    train_label = args.train_label_set
    
    test_dataset = args.test_data_set
    test_label = args.test_label_set
    
    db_enc, train_loader, valid_loader, num_per_class,cell_boundary_index,continuous_column_index,discrete_flen,continuous_flen = get_data_loader(train_dataset, train_label, args.world_size, rank, args.batch_size,  data_split_ratio=args.data_split_ratio, pin_memory=True, save_best=args.save_best,num_bins = args.num_bins)
    db_enc, test_loader, _ ,_,cell_boundary_index,continuous_column_index,discrete_flen,continuous_flen= get_data_loader(test_dataset, test_label, 1, 0, args.batch_size, args.data_split_ratio, save_best=False)
    X_fname = db_enc.X_fname
    y_fname = db_enc.y_fname

    rrl = RRL(dim_list=[(discrete_flen, continuous_flen)] + list(map(int, args.structure.split('@'))) + [len(y_fname)],
              device_id=device_id,
              use_not=args.use_not,
              is_rank0=is_rank0,
              log_file=args.log,
              writer=writer,
              save_best=args.save_best,
              estimated_grad=args.estimated_grad,
              save_path=args.model)
    
    rrl.train_model(
        data_loader=train_loader,
        valid_loader=test_loader,
        lr=args.learning_rate,
        epoch=args.epoch,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_epoch=args.lr_decay_epoch,
        weight_decay=args.weight_decay,
        log_iter=args.log_iter,
        num_per_class = num_per_class,
        cell_boundary_index = cell_boundary_index,
        continuous_column_index = continuous_column_index,
        corruption_rate=args.corruption_rate
        )


def load_model(path, device_id, log_file=None, distributed=True):
    checkpoint = torch.load(path, map_location='cpu')
    saved_args = checkpoint['rrl_args']
    rrl = RRL(
        dim_list=saved_args['dim_list'],
        device_id=device_id,
        is_rank0=True,
        use_not=saved_args['use_not'],
        log_file=log_file,
        distributed=distributed,
        estimated_grad=saved_args['estimated_grad'])
    stat_dict = checkpoint['model_state_dict']
    for key in list(stat_dict.keys()):
        stat_dict[key[7:]] = stat_dict.pop(key)
    rrl.net.load_state_dict(checkpoint['model_state_dict'])
    return rrl


def test_model(args):
    rrl = load_model(args.model, args.device_ids[0], log_file=args.test_res, distributed=False)
    train_dataset = args.train_data_set
    train_label = args.train_label_set
    
    test_dataset = args.test_data_set
    test_label = args.test_label_set
    

    print ('training set')
    db_enc, train_loader, _ ,num_per_class,cell_boundary_index,continuous_column_index,_,_= get_data_loader(train_dataset,train_label, 1, 0, args.batch_size, args.data_split_ratio, save_best=False)

    print ('test set')
    db_enc, test_loader, _,_ ,cell_boundary_index,continuous_column_index,_,_= get_data_loader(test_dataset, test_label, 1, 0, args.batch_size, args.data_split_ratio, save_best=False)

    rrl.test(test_loader=test_loader, set_name='Test', num_per_class = num_per_class)
    with open(args.rrl_file, 'w') as rrl_file:
        rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader, file=rrl_file, mean=db_enc.mean, std=db_enc.std)

        
def train_main(args):
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    #mp.spawn(train_model, nprocs=args.gpus, args=(args,))
    train_model(0, args=args)

def test_time_model(args):
    rrl = load_model(args.model, args.device_ids[0], log_file=args.test_res, distributed=False)
    train_dataset = args.train_data_set
    train_label = args.train_label_set
    
    test_dataset = args.test_data_set
    test_label = args.test_label_set
    
    db_enc, test_loader, _,_ ,cell_boundary_index,continuous_column_index,_,_ = get_data_loader(test_dataset, test_label, 1, 0, args.batch_size, args.data_split_ratio, save_best=False)

    rrl.test_time_adaptation(test_loader = test_loader, set_name = 'Test Time',args = args)
    
    #with open(args.rrl_file, 'w') as rrl_file:
    #    rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader, file=rrl_file, mean=db_enc.mean, std=db_enc.std)

    
if __name__ == '__main__':
    from args import rrl_args
    train_main(rrl_args)
    test_model(rrl_args)
    test_time_model(rrl_args)

