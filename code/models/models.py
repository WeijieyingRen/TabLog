import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from collections import defaultdict

from rrl.components import BinarizeLayer
from rrl.components import UnionLayer, LRLayer
import tent
import pdb
#from torch.distributions.uniform import Uniform
import numpy as np
from rrl.loss import NTXent

TEST_CNT_MOD = 500

# adult: [(94, 6), 1, 16, 2]
def setup_optimizer(params,args):
    lr = args.test_time_lr
    weight_decay = args.test_time_weight_decay
    
    return  torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
  


def setup_tent(model,args):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params,args)
    # init in Tent
    tent_model = tent.Tent(model, optimizer,
                           steps=args.test_time_step,
                           episodic=args.test_time_reset_phase)
    return tent_model
    
class MLLP(nn.Module):
    def __init__(self, dim_list, use_not=False, left=None, right=None, estimated_grad=False):
        super(MLLP, self).__init__()

        self.dim_list = dim_list
        self.use_not = use_not
        self.left = left
        self.right = right
        self.layer_list = nn.ModuleList([])

        prev_layer_dim = dim_list[0]
        print ('------print the layer dim ------')
        #pdb.set_trace()
        for i in range(1, len(dim_list)):
            num = prev_layer_dim
            if i >= 4:
                num += self.layer_list[-2].output_dim

            if i == 1:
                #print ('*** the first layer (BinarizeLayer) ***')
                # n, input_dim
                layer = BinarizeLayer(dim_list[i], num, self.use_not, self.left, self.right)
                layer_name = 'binary{}'.format(i)
                
            elif i == len(dim_list) - 1:
                #print ('*** the final layer (LRLayer) ***')
                
                layer = LRLayer(dim_list[i], num)
                layer_name = 'lr{}'.format(i)
                
            else:
                #print ('*** union Layer (UnionLayer) ***')
                layer = UnionLayer(dim_list[i], num, estimated_grad=estimated_grad)
                layer_name = 'union{}'.format(i)
                
            prev_layer_dim = layer.output_dim
            self.add_module(layer_name, layer)
            self.layer_list.append(layer)
        #print (self.layer_list)

    def forward(self, x):
        return self.continuous_forward(x), self.binarized_forward(x)

    def continuous_forward(self, x):
        #print ('------ continuous forward -------')
        x_res = None
        for i, layer in enumerate(self.layer_list):

            if i <= 1:
                x = layer(x)
            else:
                x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                x_res = x
                x = layer(x_cat)
        return x

    def binarized_forward(self, x):
        with torch.no_grad():
            x_res = None
            for i, layer in enumerate(self.layer_list):
                if i <= 1:
                    x = layer.binarized_forward(x)
                else:
                    x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                    x_res = x
                    x = layer.binarized_forward(x_cat)
            return x


class MyDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    @property
    def layer_list(self):
        return self.module.layer_list


class RRL:
    def __init__(self, dim_list, device_id, use_not=False, is_rank0=False, log_file=None, writer=None, left=None,
                 right=None, save_best=False, estimated_grad=False, save_path=None, distributed=True):
        super(RRL, self).__init__()
        self.dim_list = dim_list
        self.use_not = use_not
        self.best_f1 = -1.

        self.device_id = device_id
        self.is_rank0 = is_rank0
        self.save_best = save_best
        self.estimated_grad = estimated_grad
        self.save_path = save_path
        if self.is_rank0:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
            if log_file is None:
                logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
            else:
                logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode='w', format=log_format)
        self.writer = writer
        self.net = MLLP(dim_list, use_not=use_not, left=left, right=right, estimated_grad=estimated_grad)

        self.net.cuda(self.device_id)
        if distributed:
            self.net = MyDistributedDataParallel(self.net, device_ids=[self.device_id])

    def clip(self):
        """Clip the weights into the range [0, 1]."""
        for layer in self.net.layer_list[: -1]:
            layer.clip()

    def data_transform(self, X, y):
        X = X.astype(np.float)
        if y is None:
            return torch.tensor(X)
        y = y.astype(np.float)
        return torch.tensor(X), torch.tensor(y)

    def replace_values_with_uniform(self,X, Xc, continuous_column_index, cell_boundary_index):
        # Create a copy of the input matrix to avoid modifying the original matrix
        # generate corrupted data
        # Xc.shape = torch.Size([64, 11])
        # continuous_column_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18]
        # cell_boundary_index.shape = (5, 11)
        modified_matrix = np.copy(X)
        for col in range(len(continuous_column_index)):
           
            cur_continuous_index = continuous_column_index[col]
            unique_intervals = cell_boundary_index.iloc[:, col]
            for interval in range(len(unique_intervals)-1):
                
                condition_indices = np.where(Xc[:, col] == interval)
                # Generate random values for the identified indices within the specified interval
                interval_min = interval - 1 if interval-1 > 0 else 0 
                interval_max =  interval+1 if interval == len(unique_intervals) -2 else (interval + 2) 
                random_values = np.random.uniform( unique_intervals[interval_min], unique_intervals[interval_max], len(condition_indices[0]))
            # Replace the values in the modified matrix
                
                modified_matrix[condition_indices, cur_continuous_index] = random_values
                
        # for categorical data
        '''
        for col in range(modified_matrix.shape[1]-len(continuous_column_index)):
            modified_matrix[:,col] = np.random.choice(list(set(modified_matrix[:,col])), size=modified_matrix.shape[0])
        '''
        return modified_matrix

    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs."""
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def train_model(self, X=None, y=None, X_validation=None, y_validation=None, data_loader=None, valid_loader=None,
                    epoch=50, lr=0.01, lr_decay_epoch=100, lr_decay_rate=0.75, batch_size=64, weight_decay=0.0,
                    log_iter=50, num_per_class = 0,cell_boundary_index = 0, continuous_column_index = 0, corruption_rate=0.3):
       
        logit = []
        for i in range(len(num_per_class)):
            log_array = torch.log(torch.FloatTensor([num_per_class[i]/sum(num_per_class)]))
            
            logit.append(log_array)
        #logit = torch.tensor(logit,requires_grad=True)
       
        if (X is None or y is None) and data_loader is None:
            raise Exception("Both data set and data loader are unavailable.")
        if data_loader is None:
            X, y = self.data_transform(X, y)
            if X_validation is not None and y_validation is not None:
                X_validation, y_validation = self.data_transform(X_validation, y_validation)
            data_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        loss_log = []
        accuracy = []
        accuracy_b = []
        f1_score = []
        f1_score_b = []

        criterion = nn.CrossEntropyLoss().cuda(self.device_id)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        cnt = -1
        avg_batch_loss_mllp = 0.0
        avg_batch_loss_rrl = 0.0
        epoch_histc = defaultdict(list)
        
        ntxent_loss = NTXent()

        for epo in range(epoch):
            print ('current training epoch is: {}/{}'.format(epo,epoch))
            optimizer = self.exp_lr_scheduler(optimizer, epo, init_lr=lr, lr_decay_rate=lr_decay_rate,
                                              lr_decay_epoch=lr_decay_epoch)
            epoch_loss_mllp = 0.0
            epoch_loss_rrl = 0.0
            abs_gradient_max = 0.0
            abs_gradient_avg = 0.0

            ba_cnt = 0
            for X, Xc, y in data_loader:   
                ba_cnt += 1 
                
                x_modified = self.replace_values_with_uniform(X, Xc, continuous_column_index, cell_boundary_index)
                x_modified = torch.tensor(x_modified).cuda(self.device_id)
                
                X = X.cuda(self.device_id, non_blocking=True)
                
                corruption_len = int(corruption_rate * X.shape[1])
    
                corruption_mask = torch.zeros_like(X, dtype=torch.bool, device=X.device)
                for i in range(X.shape[0]):
                    corruption_idx = torch.randperm(X.shape[1])[: corruption_len]
                    corruption_mask[i, corruption_idx] = True
                x_corrupted = torch.where(corruption_mask, x_modified, X)
                
                
                
                y = y.cuda(self.device_id, non_blocking=True)
   
                optimizer.zero_grad()  # Zero the gradient buffers.
                

                #y_pred_mllp[0].shape | torch.Size([64, 252]) | y_pred_mllp[1].shape | torch.Size([64, 2])
   

                y_pred_mllp_out, y_pred_rrl_out = self.net.forward(X)
                latent_emb_mllp = y_pred_mllp_out[0]
                latent_emb_rrl = y_pred_rrl_out[0]                
                y_pred_mllp = y_pred_mllp_out[1]
                y_pred_rrl = y_pred_rrl_out[1]
                
                
                corrupted_y_pred_mllp_out, corrupted_y_pred_rrl_out = self.net.forward(x_corrupted)
                corrupted_latent_emb_mllp = corrupted_y_pred_mllp_out[0]
                corrupted_latent_emb_rrl = corrupted_y_pred_rrl_out[0]            
                corrupted_y_pred_mllp = corrupted_y_pred_mllp_out[1]
                corrupted_y_pred_rrl = corrupted_y_pred_rrl_out[1]
                
                
                with torch.no_grad():
                    
                    #logit_ = torch.stack(logit).reshape(1,2).repeat(y.shape[0],1)
                    #logit = torch.tensor(logit,requires_grad=True)
                    #logit_off = logit_.cuda(self.device_id)

                    y_prob = torch.softmax(y_pred_rrl, dim=1)
                    y_arg = torch.argmax(y, dim=1)
                    loss_mllp = criterion(y_pred_mllp, y_arg)
                    loss_rrl = criterion(y_pred_rrl, y_arg)
                    ba_loss_mllp = loss_mllp.item()
                    ba_loss_rrl = loss_rrl.item()
                    epoch_loss_mllp += ba_loss_mllp
                    epoch_loss_rrl += ba_loss_rrl
                    avg_batch_loss_mllp += ba_loss_mllp
                    avg_batch_loss_rrl += ba_loss_rrl

                
                y_pred_mllp.backward((y_prob  - y) / y.shape[0], retain_graph=True)  # for CrossEntropy Loss
                loss = ntxent_loss(latent_emb_mllp, corrupted_latent_emb_mllp)
                (1*loss).backward()
                #print ('contrastive loss is {}'.format(loss))
                #print ('cross entropy loss is {}'.format(loss_mllp))
                #print ('cross entropy loss_b is {}'.format(loss_rrl))
                
                
                #y_pred_mllp.backward((y_prob  - y) / y.shape[0])
                #pdb.set_trace()
                #y_pred_mllp.backward((y_prob + logit_off - y) / y.shape[0])  # for CrossEntropy Loss
                #(logit_off+y_pred_mllp).backward((y_prob + logit_off - y) / y.shape[0])
               
                cnt += 1

                # wjyren: add here
                self.save_model() 
                
                if self.is_rank0 and cnt % log_iter == 0 and cnt != 0 and self.writer is not None:
                    self.writer.add_scalar('training_Avg_Batch_Loss_MLLP', avg_batch_loss_mllp / log_iter, cnt)
                    self.writer.add_scalar('training_Avg_Batch_Loss_GradGrafting', avg_batch_loss_rrl / log_iter, cnt)
                    avg_batch_loss_mllp = 0.0
                    avg_batch_loss_rrl = 0.0
                optimizer.step()
                if self.is_rank0:
                    for i, param in enumerate(self.net.parameters()):
                        abs_gradient_max = max(abs_gradient_max, abs(torch.max(param.grad)))
                        abs_gradient_avg += torch.sum(torch.abs(param.grad)) / (param.grad.numel())
                self.clip()

                if self.is_rank0 and cnt % TEST_CNT_MOD == 0:
                    if X_validation is not None and y_validation is not None:
                        #pdb.set_trace()
                        acc, acc_b, f1, f1_b,auc,auc_b = self.test(X_validation, y_validation, batch_size=batch_size,
                                                         need_transform=False, set_name='test',num_per_class = num_per_class)
                    elif valid_loader is not None:
                        acc, acc_b, f1, f1_b, auc,auc_b = self.test(test_loader=valid_loader, need_transform=False,
                                                         set_name='OOD Test',num_per_class = num_per_class)
                    elif data_loader is not None:
                        acc, acc_b, f1, f1_b,auc,auc_b = self.test(test_loader=data_loader, need_transform=False,
                                                         set_name='Training',num_per_class = num_per_class)
                    else:
                        acc, acc_b, f1, f1_b,auc,auc_b = self.test(X, y, batch_size=batch_size, need_transform=False,
                                                         set_name='Training',num_per_class = num_per_class)
                    if self.save_best and f1_b > self.best_f1:
                        self.best_f1 = f1_b
                        self.save_model()
                    accuracy.append(acc)
                    accuracy_b.append(acc_b)
                    f1_score.append(f1)
                    f1_score_b.append(f1_b)
                    if self.writer is not None:
                        self.writer.add_scalar('Accuracy_MLLP', acc, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('Accuracy_RRL', acc_b, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('F1_Score_MLLP', f1, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('F1_Score_RRL', f1_b, cnt // TEST_CNT_MOD)
                        
            if self.is_rank0:
                '''
                acc, acc_b, f1, f1_b, auc,auc_b = self.test(test_loader=valid_loader, set_name='Validation')
                
                print ('val f1 score is {}'.format(f1))
                print ('val f1_b score is {}'.format(f1_b))
                
                print ('val acc score is {}'.format(acc))
                print ('val acc_b score is {}'.format(acc_b))
                
                print ('val auc score is {}'.format(auc))
                print ('val auc_b score is {}'.format(auc_b))

                logging.info('epoch: {}, loss_mllp: {}, loss_rrl: {}'.format(epo, epoch_loss_mllp, epoch_loss_rrl))
                '''
                for name, param in self.net.named_parameters():
                    maxl = 1 if 'con_layer' in name or 'dis_layer' in name else 0
                    epoch_histc[name].append(torch.histc(param.data, bins=10, max=maxl).cpu().numpy())
                if self.writer is not None:
                    self.writer.add_scalar('Training_Loss_MLLP', epoch_loss_mllp, epo)
                    self.writer.add_scalar('Training_Loss_RRL', epoch_loss_rrl, epo)
                    self.writer.add_scalar('Abs_Gradient_Max', abs_gradient_max, epo)
                    self.writer.add_scalar('Abs_Gradient_Avg', abs_gradient_avg / ba_cnt, epo)
                loss_log.append(epoch_loss_rrl)
        if self.is_rank0 and not self.save_best:
            self.save_model()
        return epoch_histc

    def test(self, X=None, y=None, test_loader=None, batch_size=32, need_transform=True, set_name='OOD Test',num_per_class = 0):
        if X is not None and y is not None and need_transform:
            X, y = self.data_transform(X, y)
        
        logit = []
        for i in range(len(num_per_class)):
            log_array = torch.FloatTensor([num_per_class[i]/sum(num_per_class)])
            logit.append(log_array)
            
        with torch.no_grad():
            if X is not None and y is not None:
                test_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

            y_list = []
            for X,_, y in test_loader:
                y_list.append(y)    
        
            y_true = torch.cat(y_list, dim=0)
            y_true = y_true.cpu().numpy().astype(int)
            y_true = np.argmax(y_true, axis=1)
            data_num = y_true.shape[0]
            slice_step = data_num // 40 if data_num >= 40 else 1
            logging.debug('{} Phase: y_true: {} {}'.format(set_name,y_true.shape, y_true[:: slice_step]))

            y_pred_list = []
            y_pred_b_list = [] 
            new_y_pred_list = []
            new_y_pred_b_list = []
            
            for X, _,y in test_loader:
                X = X.cuda(self.device_id, non_blocking=True)
                
                output = self.net.forward(X)
                
                y_pred_list.append(output[0][1])
                y_pred_b_list.append(output[1][1]) 
                new_y_pred_list.append(output[0][1] - torch.stack(logit).reshape(1,2).repeat(y.shape[0],1).cuda(output[0][1].device))
                new_y_pred_b_list.append(output[1][1] - torch.stack(logit).reshape(1,2).repeat(y.shape[0],1).cuda(output[1][1].device))
                

            y_pred = torch.cat(y_pred_list).cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            logging.debug('test_{}_y_mllp: {} {}'.format(set_name,y_pred.shape, y_pred[:: slice_step]))

            y_pred_b = torch.cat(y_pred_b_list).cpu().numpy()
            y_pred_b_arg = np.argmax(y_pred_b, axis=1)
            logging.debug('test_{}_y_rrl_: {} {}'.format(set_name, y_pred_b_arg.shape, y_pred_b_arg[:: slice_step]))
            logging.debug('test_{}_y_rrl: {} {}'.format(set_name, y_pred_b.shape, y_pred_b[:: (slice_step)]))
            
            new_y_pred = torch.cat(new_y_pred_list).cpu().numpy()
            new_y_pred = np.argmax(new_y_pred, axis=1)
            
            new_y_pred_b = torch.cat(new_y_pred_b_list).cpu().numpy()
            new_y_pred_b_arg = np.argmax(new_y_pred_b, axis=1)

            f1_score = metrics.f1_score(y_true, y_pred, average='macro')
            accuracy = metrics.accuracy_score(y_true, y_pred)          
            auc = metrics.roc_auc_score(y_true, y_pred)

            f1_score_b = metrics.f1_score(y_true, y_pred_b_arg, average='macro')
            accuracy_b = metrics.accuracy_score(y_true, y_pred_b_arg)          
            auc_b = metrics.roc_auc_score(y_true, y_pred_b_arg)

            new_f1_score = metrics.f1_score(y_true, new_y_pred, average='macro')
            new_accuracy = metrics.accuracy_score(y_true, new_y_pred)          
            new_auc = metrics.roc_auc_score(y_true, new_y_pred)

            # score of _b is pretty lower.
            new_f1_score_b = metrics.f1_score(y_true, new_y_pred_b_arg, average='macro')
            new_accuracy_b = metrics.accuracy_score(y_true, new_y_pred_b_arg)          
            new_auc_b = metrics.roc_auc_score(y_true, new_y_pred_b_arg)
            

            print('{}-num of pred/true in class 0 is: {} / {}'.format(set_name,(y_pred== 0).sum().sum(),(y_true== 0).sum().sum()))

            print('{}-num of pred/true in class 1 is: {} / {}'.format(set_name,(y_pred== 1).sum().sum(),(y_true== 1).sum().sum()))
            print ('{}-corent pred/true in class 0 is: {} / {}'.format(set_name,sum((y_pred== 0)&(y_true == 0).astype(int)),(y_true== 0).sum().sum()))
            print ('{}-corent pred/true in class 1 is: {} / {}'.format(set_name,sum((y_pred== 1)&(y_true == 1).astype(int)),(y_true== 1).sum().sum()))
            print ('{}-corent pred/pred in class 0 is: {} / {}'.format(set_name,sum((y_pred== 0)&(y_true == 0).astype(int)),(y_pred== 0).sum().sum()))
            print ('{}-corent pred/pred in class 1 is: {} / {}'.format(set_name,sum((y_pred== 1)&(y_true == 1).astype(int)),(y_pred== 1).sum().sum()))
            
            print ('{}-f1_b score is {}'.format(set_name,f1_score_b))
            print ('{}-f1 score is {}'.format(set_name,f1_score)) 
            print ('{}-new_f1_b score is {}'.format(set_name,new_f1_score_b))
            print ('{}-new_f1 score is {}'.format(set_name,new_f1_score))

            print ('{}-accuracy_b score is {}'.format(set_name,accuracy_b))
            print ('{}-accuracy is {}'.format(set_name,accuracy))
            print ('{}-new_accuracy_b score is {}'.format(set_name,new_accuracy_b))
            print ('{}-new_accuracy is {}'.format(set_name,new_accuracy))
  
            print ('{}-auc_b score is {}'.format(set_name,auc_b))
            print ('{}-auc score is {}'.format(set_name,auc))
            print ('{}-new_auc_b score is {}'.format(set_name,new_auc_b))
            print ('{}-new_auc score is {}'.format(set_name,new_auc))
            
            
            logging.info('-' * 60)
            logging.info('On {} Set:\n\t test_Accuracy of RRL  Model: {}'
                         '\n\t test_F1 Score of RRL  Model: {}'
                         '\n\t test_AUC of RRL  Model: {}'.format(set_name, accuracy_b, f1_score_b,auc_b))
            logging.info('On {} Set:\n test_Performance of  RRL Model: \n{}\n{}'.format(
                set_name, metrics.confusion_matrix(y_true, y_pred_b_arg), metrics.classification_report(y_true, y_pred_b_arg)))
            logging.info('-' * 60)
        return accuracy, accuracy_b, f1_score, f1_score_b, auc,auc_b



        
    def test_time_adaptation(self, test_loader= None, set_name = 'Test Time',args= None):
   
        model = setup_tent(self.net,args)

        try:
            model.reset()

        except:
            print ('model reset error')
        
        y_pred_list = []
        y_pred_b_list= []
        
        slice_step = 100
            
        y_list = []
        for X, _,y in test_loader:
            X = X.cuda(self.device_id, non_blocking=True)
            output = model(X)
 
            y = np.argmax(y, axis=1)
            y_list.append(y)  
            # output is 2-dim
            y_pred_list.append(output[0][1])
            y_pred_b_list.append(output[1][1]) 

        
        y_true = torch.cat(y_list, dim=0)
        y_true = y_true.cpu().numpy().astype(int)
        #y_pred = torch.cat(y_pred_list).cpu().numpy()
        y_pred = torch.cat(y_pred_list).cpu().detach().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        logging.debug('test time _{}_y_mllp: {} {}'.format(set_name,y_pred.shape, y_pred[:: slice_step]))

        '''
        y_pred_b = torch.cat(y_pred_b_list).cpu().numpy()
        y_pred_b_arg = np.argmax(y_pred_b, axis=1)
        logging.debug('tes time _{}_y_rrl_: {} {}'.format(set_name, y_pred_b_arg.shape, y_pred_b_arg[:: slice_step]))

        logging.debug('test time _{}_y_rrl: {} {}'.format(set_name, y_pred_b.shape, y_pred_b[:: (slice_step)]))
        '''
        #accuracy = metrics.accuracy_score(y_true, y_pred)
        #accuracy_b = metrics.accuracy_score(y_true, y_pred_b_arg)

        f1_score = metrics.f1_score(y_true, y_pred, average='macro')
        accuracy = metrics.accuracy_score(y_true, y_pred)          
        auc = metrics.roc_auc_score(y_true, y_pred)

        # score of _b is pretty lower.
        '''
        f1_score_b = metrics.f1_score(y_true, y_pred_b_arg, average='macro')
        accuracy_b = metrics.accuracy_score(y_true, y_pred_b_arg)          
        auc_b = metrics.roc_auc_score(y_true, y_pred_b_arg)
        '''

        #print ('test time f1 score is {}'.format(f1_score_b))
        print ('test time f1 score is {}'.format(f1_score))

        #print ('test time accuracy score is {}'.format(accuracy_b))
        print ('test time accuracy is {}'.format(accuracy))

        #print ('test time auc score is {}'.format(auc_b))
        print ('test time auc score is {}'.format(auc))
        #return accuracy, accuracy_b, f1_score, f1_score_b, auc,auc_b
        return accuracy,f1_score,auc
     
            
    def save_model(self):
        rrl_args = {'dim_list': self.dim_list, 'use_not': self.use_not, 'estimated_grad': self.estimated_grad}
        torch.save({'model_state_dict': self.net.state_dict(), 'rrl_args': rrl_args}, self.save_path)

    def detect_dead_node(self, data_loader=None):
        with torch.no_grad():
            for layer in self.net.layer_list[:-1]:
                layer.node_activation_cnt = torch.zeros(layer.output_dim, dtype=torch.double, device=self.device_id)
                layer.forward_tot = 0
            for x, _,y in data_loader:
                x = x.cuda(self.device_id)
                x_res = None
                for i, layer in enumerate(self.net.layer_list[:-1]):
                    if i <= 1:
                        x = layer.binarized_forward(x)
                    else:
                        x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                        x_res = x
                        x = layer.binarized_forward(x_cat)
                    layer.node_activation_cnt += torch.sum(x, dim=0)
                    layer.forward_tot += x.shape[0]

    def rule_print(self, feature_name, label_name, train_loader, file=sys.stdout, mean=None, std=None):
        if self.net.layer_list[1] is None and train_loader is None:
            raise Exception("Need train_loader for the dead nodes detection.")
        if self.net.layer_list[1].node_activation_cnt is None:
            self.detect_dead_node(train_loader)

        # BinarizeLayer()
        bound_name = self.net.layer_list[0].get_bound_name(feature_name, mean, std)
        '''
        self.net.layer_list[1]
             UnionLayer(
             (con_layer): ConjunctionLayer()
             (dis_layer): DisjunctionLayer()
                )
        '''
        self.net.layer_list[1].get_rules(self.net.layer_list[0], None)
        self.net.layer_list[1].get_rule_description((None, bound_name))
        if len(self.net.layer_list) >= 4:
            self.net.layer_list[2].get_rules(self.net.layer_list[1], None)
            self.net.layer_list[2].get_rule_description((None, self.net.layer_list[1].rule_name), wrap=True)

        if len(self.net.layer_list) >= 5:
            for i in range(3, len(self.net.layer_list) - 1):
                self.net.layer_list[i].get_rules(self.net.layer_list[i - 1], self.net.layer_list[i - 2])
                self.net.layer_list[i].get_rule_description(
                    (self.net.layer_list[i - 2].rule_name, self.net.layer_list[i - 1].rule_name), wrap=True)
        prev_layer = self.net.layer_list[-2]
        skip_connect_layer = self.net.layer_list[-3]
        always_act_pos = (prev_layer.node_activation_cnt == prev_layer.forward_tot)
        if skip_connect_layer.layer_type == 'union':
            shifted_dim2id = {(k + prev_layer.output_dim): (-2, v) for k, v in skip_connect_layer.dim2id.items()}
            prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
            merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})
            always_act_pos = torch.cat(
                [always_act_pos, (skip_connect_layer.node_activation_cnt == skip_connect_layer.forward_tot)])
        else:
            merged_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}

        print ('calculate w')
        #pdb.set_trace()
        Wl, bl = list(self.net.layer_list[-1].parameters())
        bl = torch.sum(Wl.T[always_act_pos], dim=0) + bl
        Wl = Wl.cpu().detach().numpy()
        bl = bl.cpu().detach().numpy()

        marked = defaultdict(lambda: defaultdict(float))
        rid2dim = {}

        for label_id, wl in enumerate(Wl):
            for i, w in enumerate(wl):
                rid = merged_dim2id[i]
                if rid == -1 or rid[1] == -1:
                    continue
                marked[rid][label_id] += w
                rid2dim[rid] = i % prev_layer.output_dim


        kv_list = sorted(marked.items(), key=lambda x: max(map(abs, x[1].values())), reverse=True)
        print('RID', end='\t', file=file)
        for i, ln in enumerate(label_name):
            print('{}(b={:.4f})'.format(ln, bl[i]), end='\t', file=file)
        print('Support\tRule', file=file)
        

        for k, v in kv_list:
            rid = k
            #print(rid, end='\t', file=file)
            #for li in range(len(label_name)):
            #    print('{:.4f}'.format(v[li]), end='\t', file=file)
            now_layer = self.net.layer_list[-1 + rid[0]]
            print('({},{})'.format(now_layer.node_activation_cnt[rid2dim[rid]].item(), now_layer.forward_tot))
            print('{:.4f}'.format((now_layer.node_activation_cnt[rid2dim[rid]] / now_layer.forward_tot).item()),
                  end='\t', file=file)
            print(now_layer.rule_name[rid[1]], end='\n', file=file)
        print('#' * 60, file=file)
        return kv_list
