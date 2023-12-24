import os
import torch
import random
import copy
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.autograd as autograd            #train_adv
from torch.autograd import Variable          #train_mmd
from functools import partial                #train_mmd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from mlp import MLP
from dsn_ae import  DSNAE
from encoder_decoder import EncoderDecoder
from collections import defaultdict
from itertools import chain
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,roc_auc_score, average_precision_score, f1_score, log_loss, auc, precision_recall_curve, precision_score, recall_score

def dict_to_str(d):
    return "_".join(["_".join([k, str(v)]) for k, v in d.items()])

def safe_make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)
    else:
        print(new_folder_name, 'exists!')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
        
def self_DataLoader(data,label, batch_size):
    #resdata = []
    #resdata_label = []
    resdata = pd.DataFrame()
    resdata_label = pd.DataFrame()
    labels_unique = pd.unique(label['cell_type'])
    counts = copy.deepcopy(label['cell_type'].value_counts())
    print('counts:',counts)   #data

    c_max = max(counts)
    class_number = len(labels_unique)
    choose_number = batch_size // class_number
    #batchsize must over class_number
    if choose_number == 0:
        choose_number = 1
    piece_number = c_max // choose_number

    print(c_max, class_number, choose_number, piece_number)
    for i in range(piece_number):
        for j in labels_unique:
            # print(counts.loc[j], c_max, i, j, data)
            tmp_data = data.loc[label['cell_type'] == j, :]
            if counts.loc[j] == c_max:
                sample_data = tmp_data.sample(n=choose_number, replace=False)
            else:
                sample_data = tmp_data.sample(n=choose_number, replace=True)
            
            sample_label = label.loc[sample_data.index, :]
            #resdata += sample_data.values[:, :-1].astype('float32').tolist()
            #sample_label = np.array([choose_number, -1])
            #sample_label.fill(j)
            resdata = pd.concat([resdata, sample_data], axis=0)
            resdata_label = pd.concat([resdata_label, sample_label], axis=0)
            #resdata_label += sample_label.tolist()
            if counts.loc[j] == c_max:
                data = data.drop(sample_data.index)
    new_batchsize = choose_number * class_number
    # print(pd.DataFrame(resdata))
    #res = TensorDataset(
        #torch.Tensor(resdata),
        #torch.Tensor(resdata_label))
    res = TensorDataset(
        torch.from_numpy(resdata.values.astype('float32')),
        torch.from_numpy(resdata_label.values))
    print(res[:][0].shape)   #data
    #print(res[:][1].shape)   #label
    #train_dataloader = DataLoader(res, batch_size=new_batchsize)
    #return train_dataloader, new_batchsize
    return res, new_batchsize     

        
def get_dataloader_generator(rna_df, atac_df, rna_label, atac_label,batch_size, seed):
    set_seed(seed)
    train_rna_df, test_rna_df, train_rna_label, test_rna_label = train_test_split(rna_df, rna_label, test_size=0.2, stratify=rna_label['cell_type'])
    print('train rna size:',train_rna_df.shape)
    print('test rna size:',test_rna_df.shape)
    train_atac_df, test_atac_df, train_atac_label, test_atac_label = train_test_split(atac_df, atac_label, test_size=len(test_rna_df) / len(atac_df), stratify=atac_label, random_state=seed)
    print('train atac size:',train_atac_df.shape)
    print('test atac size:',test_atac_df.shape)
    
    train_rna_dataset, train_rna_batch_size = self_DataLoader(train_rna_df,train_rna_label, batch_size)
    test_rna_dataset, test_rna_batch_size = self_DataLoader(test_rna_df,test_rna_label, batch_size)
    train_atac_dataset, train_atac_batch_size = self_DataLoader(train_atac_df,train_atac_label, batch_size)
    test_atac_dataset, test_atac_batch_size = self_DataLoader(test_atac_df,test_atac_label, batch_size) 
    print(train_rna_batch_size,test_rna_batch_size,train_atac_batch_size,test_atac_batch_size)   
    new_batch_size = train_rna_batch_size
    '''
    train_rna_dataset = TensorDataset(
        torch.from_numpy(train_rna_df.values.astype('float32')),
        torch.from_numpy(train_rna_label['cell_type'].values))
    test_rna_dataset = TensorDataset(
        torch.from_numpy(test_rna_df.values.astype('float32')),
        torch.from_numpy(test_rna_label['cell_type'].values))

    train_atac_dataset = TensorDataset(
        torch.from_numpy(train_atac_df.values.astype('float32')),
        torch.from_numpy(train_atac_label['cell_type'].values))
    test_atac_dataset = TensorDataset(
        torch.from_numpy(test_atac_df.values.astype('float32')),
        torch.from_numpy(test_atac_label['cell_type'].values))
        
    return (train_rna_dataset, test_rna_dataset), (train_atac_dataset, test_atac_dataset)
    '''
    return (train_rna_dataset, test_rna_dataset), (train_atac_dataset, test_atac_dataset), new_batch_size  


def single_cluster_align_train_step(s_dsnae, t_dsnae, s_dataset, t_dataset, device, optimizer, history, threshold=20, alpha=1.0, scheduler=None):
    #print('alpha:',alpha)
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    s_dsnae.train()
    t_dsnae.train()
    s_x = s_dataset[0].to(device)
    s_y = s_dataset[1].to(device) 
    t_x = t_dataset[0].to(device)
    t_y = t_dataset[1].to(device)
    s_code = s_dsnae.s_encode(s_x)
    t_code = t_dsnae.s_encode(t_x) 
    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))
        
    optimizer.zero_grad()    
    ca_loss = cluster_alignment_loss(source_features=s_code, target_features=t_code, source_label=s_y, target_label= t_y, threshold=threshold, device=device)
    #print('cluster alignment loss:',ca_loss)
    loss = s_loss_dict['loss'] + t_loss_dict['loss'] + alpha* ca_loss 
    loss.backward()        
    optimizer.step()
    
    if scheduler is not None:
        scheduler.step()
    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}
    #print('loss_dict: ',loss_dict)

    for k, v in loss_dict.items():
        if k!='loss':
            history[k].append(v)
    history['loss'].append(loss.cpu().detach().item())
    history['ca_loss'].append(ca_loss.cpu().detach().item())
    #print('history:',history)
   
    return history    
    

def single_cluster_align_train_step2(s_dsnae, t_dsnae, s_dataset, t_data,t_pseudo_label, device, optimizer, history, threshold=20, scheduler=None):
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    s_dsnae.train()
    t_dsnae.train()
    s_x = s_dataset[0].to(device)
    s_y = s_dataset[1].to(device) 
    t_x = t_data.to(device)
    t_y = t_pseudo_label.to(device)
    s_code = s_dsnae.s_encode(s_x)
    t_code = t_dsnae.s_encode(t_x) 
    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))
        
    optimizer.zero_grad()    
    ca_loss = cluster_alignment_loss(source_features=s_code, target_features=t_code, source_label=s_y, target_label= t_y, threshold=threshold, device=device)
    print('cluster alignment loss:',ca_loss)
    loss = s_loss_dict['loss'] + t_loss_dict['loss'] + ca_loss # + alpha * main_classifier_loss - beta * confounder_classifier_loss        
    loss.backward()        
    optimizer.step()
    
    if scheduler is not None:
        scheduler.step()
    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)
    history['ca_loss'].append(ca_loss.cpu().detach().item())
   
    return history  

def eval_dsnae_epoch(model, data_loader, device, history):
    model.eval()    
    avg_loss_dict = defaultdict(float)
    for x_batch in data_loader:
        x_batch = x_batch[0].to(device)
        with torch.no_grad():
            loss_dict = model.loss_function(*(model(x_batch)))
            for k, v in loss_dict.items():
                avg_loss_dict[k] += v.cpu().detach().item() / len(data_loader)
    
    for k, v in avg_loss_dict.items():
        history[k].append(v)
    return history

def euclidean_distance(x, y):
    if not len(x.shape) == len(y.shape):
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    dist = torch.sum((x - y) ** 2, 1)
    dist = torch.transpose(dist, 0, 1)
    #output = dist.sqrt() 
    output = dist*1.0
    return output
    
def cluster_loss(data, label, threshold):
    distance = euclidean_distance(data, data)
    cluster_assign = euclidean_distance(label,label)
    zero = torch.zeros_like(cluster_assign)
    one = torch.ones_like(cluster_assign)
    cluster_assign= torch.where(cluster_assign > 0, zero, one)   
    #torch.where((threshold-distance) > 0, (threshold-distance), zero)
    c_loss = torch.mean(cluster_assign*distance + (1-cluster_assign)*torch.where((threshold-distance) > 0, (threshold-distance), zero))
    return c_loss    
    
def cal_center(data, label, device):
    centers = torch.empty((0, data.shape[1])).to(device)
    for i in torch.unique(label):
        index = torch.where(label == i)[0]
        cluster_samples = data.index_select(0, index)
        #centers = torch.cat([centers, torch.mean(cluster_samples, dim=0).unsqueeze(0)], dim=0)
        centers = torch.cat([centers, torch.mean(cluster_samples.float(), dim=0).unsqueeze(0)], dim=0)
    return centers
    
def cluster_alignment_loss(source_features, target_features, source_label, target_label,threshold, device):
    source_label=source_label.unsqueeze(1)
    target_label=target_label.unsqueeze(1)
    #print('source_label size:',source_label.shape)
    #print('target_label size:',target_label.shape)
    cluster_loss_value = cluster_loss(source_features,source_label,threshold) + cluster_loss(target_features,target_label,threshold)
    s_n_clusters = len(torch.unique(source_label))
    t_n_clusters = len(torch.unique(target_label))
    s_centers = cal_center(source_features, source_label, device)
    t_centers = cal_center(target_features, target_label, device)
    if not t_n_clusters == s_n_clusters:
        print('Source data has ', s_n_clusters,' clusters, but target data only has ',t_n_clusters,' clusters.')        
    #else:
     #   print('Both source data and target data have same number of clusters.')

    s_centers_update = torch.empty((0, s_centers.shape[1])).to(device)
    #print('torch.unique(source_label):',torch.unique(source_label))
    #print('torch.unique(target_label):',torch.unique(target_label))
    #print('s_centers size',s_centers.shape)
    for i in torch.unique(target_label):
        #print(i)
        center = s_centers.index_select(0, i)
        s_centers_update = torch.cat([s_centers_update, center], dim=0)
                
    align_loss_value = torch.sum((s_centers_update-t_centers)**2)/t_n_clusters
    loss_value = cluster_loss_value + align_loss_value
    return loss_value

def model_save_check(history, metric_name, tolerance_count=5, reset_count=1):
    save_flag = False
    stop_flag = False
    if 'best_index' not in history:
        history['best_index'] = 0
    if metric_name.endswith('loss'):
        if history[metric_name][-1] <= history[metric_name][history['best_index']]:
            save_flag = True
            history['best_index'] = len(history[metric_name]) - 1
    else:
        if history[metric_name][-1] >= history[metric_name][history['best_index']]:
            save_flag = True
            history['best_index'] = len(history[metric_name]) - 1

    if len(history[metric_name]) - history['best_index'] > tolerance_count * reset_count and history['best_index'] > 0:
        stop_flag = True

    return save_flag, stop_flag

def train_dsnae_ca(s_datasets, t_datasets, test_data, test_label, kwargs):
    s_train_dataloader = DataLoader(s_datasets[0],
                                    batch_size=kwargs['batch_size'],
                                    shuffle=True, drop_last=True)
    s_test_dataloader = DataLoader(s_datasets[1],
                                   batch_size=kwargs['batch_size'],
                                   shuffle=True)
    t_train_dataloader = DataLoader(t_datasets[0],
                                    batch_size=kwargs['batch_size'],
                                    shuffle=True)
    t_test_dataloader = DataLoader(t_datasets[1],
                                   batch_size=kwargs['batch_size'],
                                   shuffle=True)  

    shared_encoder = MLP(input_dim=kwargs['input_dim'],
                         output_dim=kwargs['latent_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'],
                         dop=kwargs['dop']).to(kwargs['device'])
                         
    shared_decoder = MLP(input_dim=2+kwargs['latent_dim'],
                         output_dim=kwargs['input_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'][::-1],
                         dop=kwargs['dop']).to(kwargs['device'])

    s_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    t_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    device = kwargs['device']

    
    dsnae_train_history = defaultdict(list)
    dsnae_val_history = defaultdict(list)
    
    main_classification_pretrain_history = defaultdict(list)
    main_classification_eval_test_history = defaultdict(list)
    main_classification_eval_t_train_history = defaultdict(list)
    main_classification_eval_t_test_history = defaultdict(list)
    
    s_test_alt_val_history = defaultdict(list)
    t_test_alt_val_history = defaultdict(list)

    ae_params = [t_dsnae.private_encoder.parameters(),
                 s_dsnae.private_encoder.parameters(),
                 shared_decoder.parameters(),
                 shared_encoder.parameters()]
                     
    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=kwargs['lr'])
    print('train_num_epochs:',kwargs['train_num_epochs'])
        
    # start DSNAE pretraining        
    for epoch in range(int(kwargs['train_num_epochs'])):
        print(f'----Training Epoch {epoch} ----')                                                    
        dsnae_train_history = single_cluster_align_train_step(s_dsnae=s_dsnae,
                                                              t_dsnae=t_dsnae,
                                                              s_dataset=s_datasets[0][0:],
                                                              t_dataset=t_datasets[0][0:],
                                                              device=device,
                                                              optimizer=ae_optimizer,
                                                              history=dsnae_train_history)
                                                  
        dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
                                             data_loader=s_test_dataloader,
                                             device=device,
                                             history=dsnae_val_history)
        dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
                                             data_loader=t_test_dataloader,
                                             device=device,
                                             history=dsnae_val_history)
            
        for k in dsnae_val_history:
            if k != 'best_index':
                dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                dsnae_val_history[k].pop()

        save_flag, stop_flag = model_save_check(dsnae_val_history, metric_name='loss', tolerance_count=50)
        if kwargs['es_flag']:
            if save_flag:
                torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
                torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))
            if stop_flag:
                break
            
    if kwargs['es_flag']:
        s_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt')))
        t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt')))

    print('pre-train done!')
                                                                                  
    torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
    torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))

    return t_dsnae.shared_encoder, s_dsnae.private_encoder, t_dsnae.private_encoder, (dsnae_train_history, dsnae_val_history)
    
    
#dsnae_ca_kmeans
def train_dsnae_ca_kmeans(s_datasets, t_datasets, test_data, test_label, kwargs):
    s_train_dataloader = DataLoader(s_datasets[0],
                                    batch_size=kwargs['batch_size'],
                                    shuffle=True, drop_last=True)
    s_test_dataloader = DataLoader(s_datasets[1],
                                   batch_size=kwargs['batch_size'],
                                   shuffle=True)
    t_train_dataloader = DataLoader(t_datasets[0],
                                    batch_size=kwargs['batch_size'],
                                    shuffle=True)
    t_test_dataloader = DataLoader(t_datasets[1],
                                   batch_size=kwargs['batch_size'],
                                   shuffle=True)  

    #pred_label=KMeans(n_clusters=14, random_state=2022).fit_predict(t_datasets[0][0:][0])
    pred_label=KMeans(n_clusters=6, random_state=2022).fit_predict(t_datasets[0][0:][0])
    print('Initial ATAC Accuracy:',accuracy_score(y_true=t_datasets[0][0:][1], y_pred=pred_label))
    #a=torch.from_numpy(pred_label.values.astype('float32'))
    
    shared_encoder = MLP(input_dim=kwargs['input_dim'],
                         output_dim=kwargs['latent_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'],
                         dop=kwargs['dop']).to(kwargs['device'])
                         
    shared_decoder = MLP(input_dim=2+kwargs['latent_dim'],
                         output_dim=kwargs['input_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'][::-1],
                         dop=kwargs['dop']).to(kwargs['device'])

    s_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    t_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    device = kwargs['device']

    
    dsnae_train_history = defaultdict(list)
    dsnae_val_history = defaultdict(list)
    
    main_classification_pretrain_history = defaultdict(list)
    main_classification_eval_test_history = defaultdict(list)
    main_classification_eval_t_train_history = defaultdict(list)
    main_classification_eval_t_test_history = defaultdict(list)
    
    s_test_alt_val_history = defaultdict(list)
    t_test_alt_val_history = defaultdict(list)

    ae_params = [t_dsnae.private_encoder.parameters(),
                 s_dsnae.private_encoder.parameters(),
                 shared_decoder.parameters(),
                 shared_encoder.parameters()]
                     
    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=kwargs['lr'])
    print('train_num_epochs:',kwargs['train_num_epochs'])
        
    # start DSNAE pretraining        
    for epoch in range(int(kwargs['train_num_epochs'])):
        #if epoch % 50 == 0:
        print(f'----Training Epoch {epoch} ----')                                                    
        dsnae_train_history = single_cluster_align_train_step2(s_dsnae=s_dsnae,
                                                              t_dsnae=t_dsnae,
                                                              s_dataset=s_datasets[0][0:],
                                                              t_data=t_datasets[0][0:][0],
                                                              t_pseudo_label=torch.from_numpy(pred_label.astype('int32')),
                                                              device=device,
                                                              optimizer=ae_optimizer,
                                                              history=dsnae_train_history)
                                                  
        dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
                                             data_loader=s_test_dataloader,
                                             device=device,
                                             history=dsnae_val_history)
        dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
                                             data_loader=t_test_dataloader,
                                             device=device,
                                             history=dsnae_val_history)
        #use kmeans to update pseudo-label of ATAC data 
        atac_hidden= t_dsnae.shared_encoder(t_datasets[0][0:][0].to(device)).cpu().detach().numpy() 
        print(atac_hidden.shape)       
        #pred_label=KMeans(n_clusters=14, random_state=2022).fit_predict(atac_hidden)
        pred_label=KMeans(n_clusters=6, random_state=2022).fit_predict(atac_hidden)
        print('ATAC Accuracy:',accuracy_score(y_true=t_datasets[0][0:][1], y_pred=pred_label))
        
            
        for k in dsnae_val_history:
            if k != 'best_index':
                dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                dsnae_val_history[k].pop()

        save_flag, stop_flag = model_save_check(dsnae_val_history, metric_name='loss', tolerance_count=50)
        if kwargs['es_flag']:
            if save_flag:
                torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
                torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))
            if stop_flag:
                break
            
    if kwargs['es_flag']:
        s_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt')))
        t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt')))


    print('pre-train done!')
    #test_prob = main_classifier(test_data.to(device)).cpu().detach().numpy()
    #test_pred = np.argmax(test_prob,axis=1)
    #print('ATAC classification Acc:', accuracy_score(y_true=test_label['cell_type'], y_pred=test_pred))  
                                                                                  
    torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
    torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))

    return t_dsnae.shared_encoder, s_dsnae.private_encoder, t_dsnae.private_encoder, (dsnae_train_history, dsnae_val_history)
   

######################################
#For ADV

def dataloader_generator(rna_df, atac_df, rna_label, atac_label, seed, batch_size):
    set_seed(seed)
    train_rna_df, test_rna_df, train_rna_label, test_rna_label = train_test_split(rna_df, rna_label, test_size=0.2, stratify=rna_label['cell_type'])
    print('train rna size:',train_rna_df.shape)
    print('test rna size:',test_rna_df.shape)
    train_atac_df, test_atac_df, train_atac_label, test_atac_label = train_test_split(atac_df, atac_label, test_size=len(test_rna_df) / len(atac_df), stratify=atac_label, random_state=seed)
    print('train atac size:',train_atac_df.shape)
    print('test atac size:',test_atac_df.shape)
    '''
    atac_dataset = TensorDataset(
        torch.from_numpy(atac_df.values.astype('float32'))
    )

    rna_dataset = TensorDataset(
        torch.from_numpy(rna_df.values.astype('float32'))
    )
    '''
    train_rna_dataset = TensorDataset(
        torch.from_numpy(train_rna_df.values.astype('float32')),
        torch.from_numpy(train_rna_label['cell_type'].values),
        torch.from_numpy(train_rna_label['cell_cycle'].values))
    test_rna_dataset = TensorDataset(
        torch.from_numpy(test_rna_df.values.astype('float32')),
        torch.from_numpy(test_rna_label['cell_type'].values),
        torch.from_numpy(test_rna_label['cell_cycle'].values))

    train_atac_dataset = TensorDataset(
        torch.from_numpy(train_atac_df.values.astype('float32')),
        torch.from_numpy(train_atac_label['cell_type'].values))
    test_atac_dataset = TensorDataset(
        torch.from_numpy(test_atac_df.values.astype('float32')),
        torch.from_numpy(test_atac_label['cell_type'].values))
    
    train_rna_dataloader = DataLoader(train_rna_dataset,
                                      batch_size=batch_size,
                                      shuffle=True, drop_last=True)
    test_rna_dataloader = DataLoader(test_rna_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)
    train_atac_dataloader = DataLoader(train_atac_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
    test_atac_dataloader = DataLoader(test_atac_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
    return (train_rna_dataloader, test_rna_dataloader), (train_atac_dataloader, test_atac_dataloader)


def dsn_ae_train_step(s_dsnae, t_dsnae, s_batch, t_batch, device, optimizer, history, scheduler=None):
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    s_dsnae.train()
    t_dsnae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))

    optimizer.zero_grad()
    loss = s_loss_dict['loss'] + t_loss_dict['loss']
    loss.backward()

    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)

    return history


def critic_dsn_train_step(critic, s_dsnae, t_dsnae, s_batch, t_batch, device, optimizer, history, scheduler=None,
                          clip=None, gp=None):
    critic.zero_grad()
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    s_dsnae.eval()
    t_dsnae.eval()
    critic.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_code = s_dsnae.encode(s_x)
    t_code = t_dsnae.encode(t_x)
    
    #print('s_code:', s_code.shape)
    #print('t_code:', t_code.shape)

    loss = torch.mean(critic(t_code)) - torch.mean(critic(s_code))

    if gp is not None:
        gradient_penalty = compute_gradient_penalty(critic,
                                                    real_samples=s_code,
                                                    fake_samples=t_code,
                                                    device=device)
        loss = loss + gp * gradient_penalty

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if clip is not None:
        for p in critic.parameters():
            p.data.clamp_(-clip, clip)
    if scheduler is not None:
        scheduler.step()

    history['critic_loss'].append(loss.cpu().detach().item())

    return history


def gan_dsn_gen_train_step(critic, s_dsnae, t_dsnae, s_batch, t_batch, device, optimizer, alpha, history,
                           scheduler=None):
    critic.zero_grad()
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    critic.eval()
    s_dsnae.train()
    t_dsnae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    t_code = t_dsnae.encode(t_x)

    optimizer.zero_grad()
    gen_loss = -torch.mean(critic(t_code))
    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))
    recons_loss = s_loss_dict['loss'] + t_loss_dict['loss']
    loss = recons_loss + alpha * gen_loss
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)
    history['gen_loss'].append(gen_loss.cpu().detach().item())

    return history
   

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.shape[0], 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    fakes = torch.ones((real_samples.shape[0], 1)).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=fakes,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
    

def train_adv(s_dataloaders, t_dataloaders, kwargs):
    s_train_dataloader = s_dataloaders[0]
    s_test_dataloader = s_dataloaders[1]

    t_train_dataloader = t_dataloaders[0]
    t_test_dataloader = t_dataloaders[1]

    shared_encoder = MLP(input_dim=kwargs['input_dim'],
                         output_dim=kwargs['latent_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'],
                         dop=kwargs['dop']).to(kwargs['device'])

    shared_decoder = MLP(input_dim=2+kwargs['latent_dim'],
                         output_dim=kwargs['input_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'][::-1],
                         dop=kwargs['dop']).to(kwargs['device'])

    s_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    t_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    confounding_classifier = MLP(input_dim=kwargs['latent_dim']+1,         #shared + private
                                 output_dim=1,
                                 hidden_dims=kwargs['classifier_hidden_dims'],
                                 dop=kwargs['dop']).to(kwargs['device'])

    #params are wight matrix and bias vector
    ae_params = [t_dsnae.private_encoder.parameters(),
                 s_dsnae.private_encoder.parameters(),
                 shared_decoder.parameters(),
                 shared_encoder.parameters()
                 ]
    t_ae_params = [t_dsnae.private_encoder.parameters(),
                   s_dsnae.private_encoder.parameters(),
                   shared_decoder.parameters(),
                   shared_encoder.parameters()
                   ]

    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=kwargs['lr'])
    classifier_optimizer = torch.optim.RMSprop(confounding_classifier.parameters(), lr=kwargs['lr'])
    t_ae_optimizer = torch.optim.RMSprop(chain(*t_ae_params), lr=kwargs['lr'])

    dsnae_train_history = defaultdict(list)
    dsnae_val_history = defaultdict(list)
    critic_train_history = defaultdict(list)
    gen_train_history = defaultdict(list)


    # start dsnae pre-training
    for epoch in range(int(kwargs['pretrain_num_epochs'])):
        #if epoch % 50 == 0:
        print(f'---------AE training epoch {epoch}---------')
        for step, s_batch in enumerate(s_train_dataloader):
            t_batch = next(iter(t_train_dataloader))
            dsnae_train_history = dsn_ae_train_step(s_dsnae=s_dsnae,
                                                        t_dsnae=t_dsnae,
                                                        s_batch=s_batch,
                                                        t_batch=t_batch,
                                                        device=kwargs['device'],
                                                        optimizer=ae_optimizer,
                                                        history=dsnae_train_history)
        dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
                                                 data_loader=s_test_dataloader,
                                                 device=kwargs['device'],
                                                 history=dsnae_val_history)
        dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
                                                 data_loader=t_test_dataloader,
                                                 device=kwargs['device'],
                                                 history=dsnae_val_history)
        for k in dsnae_val_history:
            if k != 'best_index':
                dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                dsnae_val_history[k].pop()
        if kwargs['es_flag']:
            save_flag, stop_flag = model_save_check(dsnae_val_history, metric_name='loss', tolerance_count=20)
            if save_flag:
                torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'a_s_dsnae.pt'))
                torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'a_t_dsnae.pt'))
            if stop_flag:
                break
    if kwargs['es_flag']:
        s_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'a_s_dsnae.pt')))
        t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'a_t_dsnae.pt')))

    for epoch in range(int(kwargs['train_num_epochs'])):
        #if epoch % 50 == 0:
        print(f'---------confounder wgan training epoch {epoch}---------')
        for step, s_batch in enumerate(s_train_dataloader):
            t_batch = next(iter(t_train_dataloader))
            critic_train_history = critic_dsn_train_step(critic=confounding_classifier,
                                                             s_dsnae=s_dsnae,
                                                             t_dsnae=t_dsnae,
                                                             s_batch=s_batch,
                                                             t_batch=t_batch,
                                                             device=kwargs['device'],
                                                             optimizer=classifier_optimizer,
                                                             history=critic_train_history,
                                                             # clip=0.1,
                                                             gp=10.0)
            if (step + 1) % 5 == 0:
                gen_train_history = gan_dsn_gen_train_step(critic=confounding_classifier,
                                                               s_dsnae=s_dsnae,
                                                               t_dsnae=t_dsnae,
                                                               s_batch=s_batch,
                                                               t_batch=t_batch,
                                                               device=kwargs['device'],
                                                               optimizer=t_ae_optimizer,
                                                               alpha=1.0,
                                                               history=gen_train_history)

    torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'a_s_dsnae.pt'))
    torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'a_t_dsnae.pt'))

    return t_dsnae.shared_encoder,s_dsnae.private_encoder,t_dsnae.private_encoder, (dsnae_train_history, dsnae_val_history, critic_train_history, gen_train_history)


######################################
#For MMD
######################################
def pairwise_distance(x, y):
    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)
    return output


def gaussian_kernel_matrix(x, y, sigmas):
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)

def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost


def mmd_loss(source_features, target_features, device):
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
        gaussian_kernel_matrix, sigmas=Variable(torch.Tensor(sigmas), requires_grad=False).to(device)
    )

    loss_value = maximum_mean_discrepancy(source_features, target_features, kernel=gaussian_kernel)
    loss_value = loss_value

    return loss_value

def dsn_ae_mmd_train_step(s_dsnae, t_dsnae, s_batch, t_batch, lambda1, device, optimizer, history, scheduler=None):
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    s_dsnae.train()
    t_dsnae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_code = s_dsnae.encode(s_x)
    t_code = t_dsnae.encode(t_x)

    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))

    optimizer.zero_grad()
    m_loss = mmd_loss(source_features=s_code, target_features=t_code, device=device)
    #print('mmd loss:', m_loss)
    loss = lambda1*m_loss + s_loss_dict['loss'] + t_loss_dict['loss'] 
    loss.backward()

    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)
    history['mmd_loss'].append(m_loss.cpu().detach().item())

    return history

def train_mmd(s_dataloaders, t_dataloaders, kwargs):
    s_train_dataloader = s_dataloaders[0]
    s_test_dataloader = s_dataloaders[1]

    t_train_dataloader = t_dataloaders[0]
    t_test_dataloader = t_dataloaders[1]

    shared_encoder = MLP(input_dim=kwargs['input_dim'],
                         output_dim=kwargs['latent_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'],
                         dop=kwargs['dop']).to(kwargs['device'])

    shared_decoder = MLP(input_dim=2+kwargs['latent_dim'],
                         output_dim=kwargs['input_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'][::-1],
                         dop=kwargs['dop']).to(kwargs['device'])

    s_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    t_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])
    
    device = kwargs['device']

    dsnae_train_history = defaultdict(list)
    dsnae_val_history = defaultdict(list)

    #if kwargs['retrain_flag']:
    ae_params = [t_dsnae.private_encoder.parameters(),
                 s_dsnae.private_encoder.parameters(),
                 shared_decoder.parameters(),
                 shared_encoder.parameters()]

    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=kwargs['lr'])
        
    print('train_num_epochs:',kwargs['train_num_epochs'])
        # start autoencoder pretraining
    for epoch in range(int(kwargs['train_num_epochs'])):
        #if epoch % 50 == 0:
        print(f'---------Autoencoder Pre-Training Epoch {epoch}---------')
        for step, s_batch in enumerate(s_train_dataloader):
            t_batch = next(iter(t_train_dataloader))
            dsnae_train_history = dsn_ae_mmd_train_step(s_dsnae=s_dsnae,
                                                        t_dsnae=t_dsnae,
                                                        s_batch=s_batch,
                                                        t_batch=t_batch,
                                                        device=device,
                                                        optimizer=ae_optimizer,
                                                        history=dsnae_train_history)
        dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
                                             data_loader=s_test_dataloader,
                                             device=device,
                                             history=dsnae_val_history
                                             )
        dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
                                             data_loader=t_test_dataloader,
                                             device=device,
                                             history=dsnae_val_history
                                             )
        for k in dsnae_val_history:
            if k != 'best_index':
                dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                dsnae_val_history[k].pop()

        save_flag, stop_flag = model_save_check(dsnae_val_history, metric_name='loss', tolerance_count=50)
        if kwargs['es_flag']:
            if save_flag:
                torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
                torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))
            if stop_flag:
                break

    if kwargs['es_flag']:
        s_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt')))
        t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt')))

    torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
    torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))

    return t_dsnae.shared_encoder,s_dsnae.private_encoder,t_dsnae.private_encoder, (dsnae_train_history, dsnae_val_history)


def weighted_DataLoader(dataset, batch_size):
    labels_unique, counts = torch.unique(dataset[:][1], return_counts =True)
    #print("Unique labels: {}".format(labels_unique))
    class_weights = [sum(counts)/c for c in counts]
    weights = [class_weights[e] for e in dataset[:][1]]
    sampler = WeightedRandomSampler(weights, len(dataset[:][1]))
    train_dataloader = DataLoader(dataset, sampler = sampler, batch_size = batch_size)
    return train_dataloader
    
def train_dsnae_ca_batch_size(s_datasets, t_datasets, test_data, test_label, kwargs):
    s_train_dataloader = weighted_DataLoader(s_datasets[0], batch_size=kwargs['batch_size'])
    s_test_dataloader = weighted_DataLoader(s_datasets[1], batch_size=kwargs['batch_size'])
    t_train_dataloader = weighted_DataLoader(t_datasets[0], batch_size=kwargs['batch_size'])
    t_test_dataloader = weighted_DataLoader(t_datasets[1], batch_size=kwargs['batch_size'])

    shared_encoder = MLP(input_dim=kwargs['input_dim'],
                         output_dim=kwargs['latent_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'],
                         dop=kwargs['dop']).to(kwargs['device'])
                         
    shared_decoder = MLP(input_dim=2+kwargs['latent_dim'],
                         output_dim=kwargs['input_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'][::-1],
                         dop=kwargs['dop']).to(kwargs['device'])

    s_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])
    #print('s_dsnae:',s_dsnae)

    t_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])
    #print('t_dsnae:',t_dsnae)

    device = kwargs['device']
    alpha = kwargs['alpha']

    
    dsnae_train_history = defaultdict(list)
    dsnae_val_history = defaultdict(list)
    
    main_classification_pretrain_history = defaultdict(list)
    main_classification_eval_test_history = defaultdict(list)
    main_classification_eval_t_train_history = defaultdict(list)
    main_classification_eval_t_test_history = defaultdict(list)
    
    s_test_alt_val_history = defaultdict(list)
    t_test_alt_val_history = defaultdict(list)

    ae_params = [t_dsnae.private_encoder.parameters(),
                 s_dsnae.private_encoder.parameters(),
                 shared_decoder.parameters(),
                 shared_encoder.parameters()]
                     
    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=kwargs['lr'])
    print('train_num_epochs:',kwargs['train_num_epochs'])
        
    # start DSNAE pretraining        
    for epoch in range(int(kwargs['train_num_epochs'])):
        if epoch % 50 == 0:
            print(f'----Training Epoch {epoch} ----')   
        for step, s_batch in enumerate(s_train_dataloader):
            t_batch = next(iter(t_train_dataloader))                                                 
            dsnae_train_history = single_cluster_align_train_step(s_dsnae=s_dsnae,
                                                                  t_dsnae=t_dsnae,
                                                                  s_dataset=s_batch,
                                                                  t_dataset=t_batch,
                                                                  device=device,
                                                                  alpha=alpha,
                                                                  optimizer=ae_optimizer,
                                                                  history=dsnae_train_history)
                                                  
        dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
                                             data_loader=s_test_dataloader,
                                             device=device,
                                             history=dsnae_val_history)
        dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
                                             data_loader=t_test_dataloader,
                                             device=device,
                                             history=dsnae_val_history)
            
        for k in dsnae_val_history:
            if k != 'best_index':
                dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                dsnae_val_history[k].pop()

        save_flag, stop_flag = model_save_check(dsnae_val_history, metric_name='loss', tolerance_count=50)
        if kwargs['es_flag']:
            if save_flag:
                torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
                torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))
            if stop_flag:
                break
            
    if kwargs['es_flag']:
        s_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt')))
        t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt')))

    print('pre-train done!')
                                                                                  
    torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
    torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))

    return t_dsnae.shared_encoder, s_dsnae.private_encoder, t_dsnae.private_encoder, (dsnae_train_history, dsnae_val_history)


def ca_mmd_train_step(s_dsnae, t_dsnae, s_dataset, t_dataset, device, optimizer, history, threshold=20, scheduler=None):
    #print('alpha:',alpha)
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    s_dsnae.train()
    t_dsnae.train()
    s_x = s_dataset[0].to(device)
    s_y = s_dataset[1].to(device) 
    t_x = t_dataset[0].to(device)
    t_y = t_dataset[1].to(device)
    s_code = s_dsnae.s_encode(s_x)
    t_code = t_dsnae.s_encode(t_x) 
    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))
        
    optimizer.zero_grad()
    m_loss = mmd_loss(source_features=s_code, target_features=t_code, device=device)
    ca_loss = cluster_alignment_loss(source_features=s_code, target_features=t_code, source_label=s_y, target_label= t_y, threshold=threshold, device=device)
    #print('cluster alignment loss:',ca_loss)
    loss = s_loss_dict['loss'] + t_loss_dict['loss'] + m_loss + ca_loss 
    #print(s_loss_dict['loss'], t_loss_dict['loss'], m_loss , ca_loss )
    loss.backward()        
    optimizer.step()
    
    if scheduler is not None:
        scheduler.step()
    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}
    #print('loss_dict: ',loss_dict)

    for k, v in loss_dict.items():
        if k!='loss':
            history[k].append(v)
    history['loss'].append(loss.cpu().detach().item())
    history['ca_loss'].append(ca_loss.cpu().detach().item())
    history['mmd_loss'].append(m_loss.cpu().detach().item())
    #print('history:',history)
   
    return history

def train_ca_mmd(s_datasets, t_datasets, test_data, test_label, kwargs):
    s_train_dataloader = weighted_DataLoader(s_datasets[0], batch_size=kwargs['batch_size'])
    s_test_dataloader = weighted_DataLoader(s_datasets[1], batch_size=kwargs['batch_size'])
    t_train_dataloader = weighted_DataLoader(t_datasets[0], batch_size=kwargs['batch_size'])
    t_test_dataloader = weighted_DataLoader(t_datasets[1], batch_size=kwargs['batch_size'])

    shared_encoder = MLP(input_dim=kwargs['input_dim'],
                         output_dim=kwargs['latent_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'],
                         dop=kwargs['dop']).to(kwargs['device'])
                         
    shared_decoder = MLP(input_dim=2+kwargs['latent_dim'],
                         output_dim=kwargs['input_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'][::-1],
                         dop=kwargs['dop']).to(kwargs['device'])

    s_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])
    #print('s_dsnae:',s_dsnae)

    t_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])
    #print('t_dsnae:',t_dsnae)

    device = kwargs['device']
    #alpha = kwargs['alpha']

    
    dsnae_train_history = defaultdict(list)
    dsnae_val_history = defaultdict(list)
    
    main_classification_pretrain_history = defaultdict(list)
    main_classification_eval_test_history = defaultdict(list)
    main_classification_eval_t_train_history = defaultdict(list)
    main_classification_eval_t_test_history = defaultdict(list)
    
    s_test_alt_val_history = defaultdict(list)
    t_test_alt_val_history = defaultdict(list)

    ae_params = [t_dsnae.private_encoder.parameters(),
                 s_dsnae.private_encoder.parameters(),
                 shared_decoder.parameters(),
                 shared_encoder.parameters()]
                     
    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=kwargs['lr'])
    print('train_num_epochs:',kwargs['train_num_epochs'])
        
    # start DSNAE pretraining        
    for epoch in range(int(kwargs['train_num_epochs'])):
        if epoch % 50 == 0:
            print(f'----Training Epoch {epoch} ----')   
        for step, s_batch in enumerate(s_train_dataloader):
            t_batch = next(iter(t_train_dataloader))                                                 
            dsnae_train_history = ca_mmd_train_step(s_dsnae=s_dsnae,
                                                    t_dsnae=t_dsnae,
                                                    s_dataset=s_batch,
                                                    t_dataset=t_batch,
                                                    device=device,
                                                    optimizer=ae_optimizer,
                                                    history=dsnae_train_history)
                                                  
        dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
                                             data_loader=s_test_dataloader,
                                             device=device,
                                             history=dsnae_val_history)
        dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
                                             data_loader=t_test_dataloader,
                                             device=device,
                                             history=dsnae_val_history)
            
        for k in dsnae_val_history:
            if k != 'best_index':
                dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                dsnae_val_history[k].pop()

        save_flag, stop_flag = model_save_check(dsnae_val_history, metric_name='loss', tolerance_count=50)
        if kwargs['es_flag']:
            if save_flag:
                torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
                torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))
            if stop_flag:
                break
            
    if kwargs['es_flag']:
        s_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt')))
        t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt')))

    print('pre-train done!')
                                                                                  
    torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
    torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))

    return t_dsnae.shared_encoder, s_dsnae.private_encoder, t_dsnae.private_encoder, (dsnae_train_history, dsnae_val_history)
    
    
#############################################
#pretrain functions     recons+mmd+ortho+classfication
#customized_dsn_ae_train_step() include: s_DSNAE, t_DSNAE,  main classifier
def main_classification_train_step(classifier, batch, loss_fn, device, optimizer, history, scheduler=None, clip=None):
    classifier.zero_grad()
    classifier.train()

    x = batch[0].to(device)
    y = batch[1].to(device)
    #loss = loss_fn(model(x), y.double().unsqueeze(1))
    loss = loss_fn(classifier(x), y.squeeze(-1))

    optimizer.zero_grad()
    loss.backward()
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    history['main_loss'].append(loss.cpu().detach().item())
    return history

def evaluate_main_classification_epoch(classifier, dataloader, datatype, device, history):
    y_truths = np.array([])
    #y_preds = np.array([])
    y_preds = np.empty(shape=(0,6))
    classifier.eval()
    
    if datatype == 'source':
       for x_batch, y_batch, yy_batch in dataloader:
           x_batch = x_batch.to(device)
           y_batch = y_batch.to(device)
           with torch.no_grad():
                y_truths = np.concatenate([y_truths, y_batch.cpu().detach().numpy().ravel()])
                y_pred = torch.sigmoid(classifier(x_batch)).detach()
            #y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])
                y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy()])
    else:       
       for x_batch, y_batch in dataloader:
           x_batch = x_batch.to(device)
           y_batch = y_batch.to(device)
           with torch.no_grad():
                y_truths = np.concatenate([y_truths, y_batch.cpu().detach().numpy().ravel()])
                y_pred = torch.sigmoid(classifier(x_batch)).detach()
            #y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])
                y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy()])
            
    y_preds=np.argmax(y_preds,axis=1)
    history['acc'].append(accuracy_score(y_true=y_truths, y_pred=y_preds))
    #history['f1_macro'].append(f1_score(y_true=y_truths, y_pred=y_preds,average='macro'))
    #history['f1_micro'].append(f1_score(y_true=y_truths, y_pred=y_preds,average='micro'))
    #history['precision_macro'].append(precision_score(y_true=y_truths, y_pred=y_preds,average='macro'))
    #history['precision_micro'].append(precision_score(y_true=y_truths, y_pred=y_preds,average='micro'))
    #history['recall_macro'].append(recall_score(y_true=y_truths, y_pred=y_preds,average='macro'))
    #history['recall_micro'].append(recall_score(y_true=y_truths, y_pred=y_preds,average='micro'))    
    return history    

def customized_dsnae_classifier_train_step(main_classifier, s_dsnae, t_dsnae, s_batch, t_batch, loss_fn, lambda1, device, optimizer, history, scheduler=None):
    main_classifier.zero_grad()
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    main_classifier.eval()
    s_dsnae.train()
    t_dsnae.train()

    s_x = s_batch[0].to(device)
    s_y = s_batch[1].to(device) #cell type label
    t_x = t_batch[0].to(device)

    
    main_classifier_loss = loss_fn(main_classifier(s_x), s_y.squeeze(-1))
    

    s_code = s_dsnae.encode(s_x)
    t_code = t_dsnae.encode(t_x)

    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))

    m_loss = mmd_loss(source_features=s_code, target_features=t_code, device=device)
    loss = main_classifier_loss + lambda1* m_loss + s_loss_dict['loss'] + t_loss_dict['loss'] 
    #print('s loss:', s_loss_dict['loss'],s_loss_dict['recons_loss'],s_loss_dict['ortho_loss'])
    #print('t loss:', t_loss_dict['loss'],t_loss_dict['recons_loss'],t_loss_dict['ortho_loss'])
    #print('main_classifier_loss:',main_classifier_loss)
    #print('mmd_loss:',m_loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()
    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)
    #history['loss'].append(loss.cpu().detach().item())
    #history['mmd_loss'].append(m_loss.cpu().detach().item())

    return history



def pre_train(s_dataloaders, t_dataloaders, test_data, test_label, kwargs):    
    s_train_dataloader = s_dataloaders[0]
    s_test_dataloader = s_dataloaders[1]

    t_train_dataloader = t_dataloaders[0]
    t_test_dataloader = t_dataloaders[1]

    shared_encoder = MLP(input_dim=kwargs['input_dim'],
                         output_dim=kwargs['latent_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'],
                         dop=kwargs['dop']).to(kwargs['device'])
                         
    shared_decoder = MLP(input_dim=2+kwargs['latent_dim'],
                         output_dim=kwargs['input_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'][::-1],
                         dop=kwargs['dop']).to(kwargs['device'])

    s_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    lambda2=kwargs['lambda'][1],
                    lambda3=kwargs['lambda'][2],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    t_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    lambda2=kwargs['lambda'][1],
                    lambda3=kwargs['lambda'][2],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    #print('t_dsnae:',t_dsnae)
    
    #cell type classifier 
    classifier = MLP(input_dim=kwargs['latent_dim'],
                     output_dim=6,
                     hidden_dims=kwargs['classifier_hidden_dims'],
                     dop=kwargs['dop']).to(kwargs['device'])
    main_classifier = EncoderDecoder(encoder=s_dsnae.shared_encoder, decoder=classifier).to(kwargs['device'])
    #print('Main classifier structure:',main_classifier)

    device = kwargs['device']

    classification_loss = nn.CrossEntropyLoss()
    
    dsnae_train_history = defaultdict(list)
    dsnae_val_history = defaultdict(list)

    main_classification_pretrain_history = defaultdict(list)
    main_classification_eval_test_history = defaultdict(list)
    main_classification_eval_t_train_history = defaultdict(list)
    main_classification_eval_t_test_history = defaultdict(list)


    ae_params = [t_dsnae.private_encoder.parameters(),
                 s_dsnae.private_encoder.parameters(),
                 shared_decoder.parameters(),
                 shared_encoder.parameters()]
                     
    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=kwargs['lr'])
    main_classifier_optimizer = torch.optim.AdamW(main_classifier.decoder.parameters(), lr=kwargs['lr'])
        
    print('train_num_epochs:',kwargs['train_num_epochs'])
    # start DSNAE pretraining        
    for epoch in range(int(kwargs['train_num_epochs'])):
        if epoch % 100 == 0:
            print(f'----Autoencoder  Pre-training Epoch {epoch} ----')
        for step, s_batch in enumerate(s_train_dataloader):
            t_batch = next(iter(t_train_dataloader))
            dsnae_train_history = dsn_ae_mmd_train_step(s_dsnae=s_dsnae,
                                                        t_dsnae=t_dsnae,
                                                        s_batch=s_batch,
                                                        t_batch=t_batch,
                                                        lambda1=kwargs['lambda'][0],
                                                        device=device,
                                                        optimizer=ae_optimizer,
                                                        history=dsnae_train_history)
        dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
                                                 data_loader=s_test_dataloader,
                                                 device=device,
                                                 history=dsnae_val_history
                                                 )
        dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
                                                 data_loader=t_test_dataloader,
                                                 device=device,
                                                 history=dsnae_val_history
                                                 )
        for k in dsnae_val_history:
            if k != 'best_index':
                dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                dsnae_val_history[k].pop()

        save_flag, stop_flag = model_save_check(dsnae_val_history, metric_name='loss', tolerance_count=50)
        if kwargs['es_flag']:
            if save_flag:
                torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
                torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))
            if stop_flag:
                break

    if kwargs['es_flag']:
        s_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt')))
        t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt')))
        
        
        
    # start main classifier pre-training
    for epoch in range(int(kwargs['train_num_epochs'])):
        if epoch % 100 == 0:
            print(f'----Cell Type Classifier Pre-training Epoch {epoch}----')
        for step, s_batch in enumerate(s_train_dataloader):
            #t_batch = next(iter(t_train_dataloader))
            main_classification_pretrain_history = main_classification_train_step(classifier=main_classifier,
                                                                                      batch=s_batch,
                                                                                      loss_fn=classification_loss,
                                                                                      device=kwargs['device'],
                                                                                      optimizer=main_classifier_optimizer,
                                                                                      history=main_classification_pretrain_history)

        main_classification_eval_test_history = evaluate_main_classification_epoch(classifier=main_classifier,
                                                                                       dataloader=s_test_dataloader,
                                                                                       datatype='source',
                                                                                       device=kwargs['device'],
                                                                                       history=main_classification_eval_test_history)
        main_classification_eval_t_train_history = evaluate_main_classification_epoch(classifier=main_classifier,
                                                                                          dataloader=t_train_dataloader,
                                                                                        datatype='target',
                                                                                        device=kwargs['device'],
                                                                                        history=main_classification_eval_t_train_history)
        main_classification_eval_t_test_history = evaluate_main_classification_epoch(classifier=main_classifier,
                                                                                         dataloader=t_test_dataloader,
                                                                                         datatype='target',
                                                                                         device=kwargs['device'],
                                                                                         history=main_classification_eval_t_test_history)

        save_flag, stop_flag = model_save_check(history=main_classification_eval_test_history, metric_name='acc',
                                                    tolerance_count=50)
        if kwargs['es_flag']:
            if save_flag:
                torch.save(confounder_classifier.state_dict(),
                           os.path.join(kwargs['model_save_folder'], 'adv_classifier.pt'))
            if stop_flag:
                break

    if kwargs['es_flag']:
        main_classifier.load_state_dict(
        torch.load(os.path.join(kwargs['model_save_folder'], 'main_classifier.pt')))

    print('pre-train done!')
    test_prob= main_classifier(test_data.to(device)).cpu().detach().numpy()
    test_pred=np.argmax(test_prob,axis=1)
    print('ATAC classification Acc:', accuracy_score(y_true=test_label['cell_type'], y_pred=test_pred))        

    #start alternating training
    for epoch in range(int(kwargs['train_num_epochs'])):
        if epoch % 100 == 0:
            print(f'----Alternative training epoch {epoch}----')
        # start autoencoder training epoch
        for step, s_batch in enumerate(s_train_dataloader):
            t_batch = next(iter(t_train_dataloader))
                
            dsnae_train_history = customized_dsnae_classifier_train_step(main_classifier=main_classifier,
                                                                         s_dsnae=s_dsnae,
                                                                         t_dsnae=t_dsnae,
                                                                         s_batch=s_batch,
                                                                         t_batch=t_batch,
                                                                         loss_fn=classification_loss,
                                                                         lambda1=kwargs['lambda'][0],
                                                                         device=device,
                                                                         optimizer=ae_optimizer,
                                                                         history=dsnae_train_history)
        dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
                                             data_loader=s_test_dataloader,
                                             device=device,
                                             history=dsnae_val_history
                                             )
        dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
                                             data_loader=t_test_dataloader,
                                             device=device,
                                             history=dsnae_val_history
                                             )                                                                 
            
        for k in dsnae_val_history:
            if k != 'best_index':
                dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                dsnae_val_history[k].pop()
                                                                                     
        main_classification_eval_test_history = evaluate_main_classification_epoch(classifier=main_classifier,
                                                                                   dataloader=s_test_dataloader,
                                                                                   datatype='source',
                                                                                   device=kwargs['device'],
                                                                                   history=main_classification_eval_test_history)
        main_classification_eval_t_train_history = evaluate_main_classification_epoch(classifier=main_classifier,
                                                                                      dataloader=t_train_dataloader,
                                                                                      datatype='target',
                                                                                      device=kwargs['device'],
                                                                                      history=main_classification_eval_t_train_history)
        main_classification_eval_t_test_history = evaluate_main_classification_epoch(classifier=main_classifier,
                                                                                     dataloader=t_test_dataloader,
                                                                                     datatype='target',
                                                                                     device=kwargs['device'],
                                                                                     history=main_classification_eval_t_test_history)                    


                                                                                                   
    torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
    torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))
    torch.save(main_classifier.state_dict(), os.path.join(kwargs['model_save_folder'], 'main_classifier.pt'))

    return t_dsnae.shared_encoder, s_dsnae.private_encoder, t_dsnae.private_encoder, (dsnae_train_history, dsnae_val_history), main_classifier,(main_classification_pretrain_history,main_classification_eval_test_history,main_classification_eval_t_train_history,main_classification_eval_t_test_history)

#####################
#train_ca_classifier
def evaluate_main_classification_epoch2(classifier, dataloader, datatype, device, history):
    y_truths = np.array([])
    #y_preds = np.array([])
    y_preds = np.empty(shape=(0,13))        #Change the number of cell clusters
    classifier.eval()
    
    if datatype == 'source':
       for x_batch, y_batch in dataloader:
           x_batch = x_batch.to(device)
           y_batch = y_batch.to(device)
           with torch.no_grad():
                y_truths = np.concatenate([y_truths, y_batch.cpu().detach().numpy().ravel()])
                y_pred = torch.sigmoid(classifier(x_batch)).detach()
                #y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])
                y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy()])
    else:       
       for x_batch, y_batch in dataloader:
           x_batch = x_batch.to(device)
           y_batch = y_batch.to(device)
           with torch.no_grad():
                y_truths = np.concatenate([y_truths, y_batch.cpu().detach().numpy().ravel()])
                y_pred = torch.sigmoid(classifier(x_batch)).detach()
            #y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])
                y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy()])
            
    y_preds=np.argmax(y_preds,axis=1)
    history['acc'].append(accuracy_score(y_true=y_truths, y_pred=y_preds))
    #history['f1_macro'].append(f1_score(y_true=y_truths, y_pred=y_preds,average='macro'))
    #history['f1_micro'].append(f1_score(y_true=y_truths, y_pred=y_preds,average='micro'))
    #history['precision_macro'].append(precision_score(y_true=y_truths, y_pred=y_preds,average='macro'))
    #history['precision_micro'].append(precision_score(y_true=y_truths, y_pred=y_preds,average='micro'))
    #history['recall_macro'].append(recall_score(y_true=y_truths, y_pred=y_preds,average='macro'))
    #history['recall_micro'].append(recall_score(y_true=y_truths, y_pred=y_preds,average='micro'))    
    return history    

def customized_ca_classifier_train_step(main_classifier, s_dsnae, t_dsnae, s_dataset, t_dataset,loss_fn, device, optimizer, history, threshold=20, alpha=1.0, scheduler=None):
    main_classifier.zero_grad()
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    main_classifier.eval()
    s_dsnae.train()
    t_dsnae.train()

    s_x = s_dataset[0].to(device)
    s_y = s_dataset[1].to(device) 
    t_x = t_dataset[0].to(device)
    t_y = t_dataset[1].to(device)

    
    main_classifier_loss = loss_fn(main_classifier(s_x), s_y.squeeze(-1))
    

    s_code = s_dsnae.encode(s_x)
    t_code = t_dsnae.encode(t_x)

    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))

    ca_loss = cluster_alignment_loss(source_features=s_code, target_features=t_code, source_label=s_y, target_label= t_y, threshold=threshold, device=device)
    #print('cluster alignment loss:',ca_loss)
    #loss = s_loss_dict['loss'] + t_loss_dict['loss'] + alpha* ca_loss 

    #m_loss = mmd_loss(source_features=s_code, target_features=t_code, device=device)
    loss = s_loss_dict['loss'] + t_loss_dict['loss']  + alpha * main_classifier_loss + ca_loss #- beta * confounder_classifier_loss 
    #print('s loss:', s_loss_dict['loss'],s_loss_dict['recons_loss'],s_loss_dict['ortho_loss'])
    #print('t loss:', t_loss_dict['loss'],t_loss_dict['recons_loss'],t_loss_dict['ortho_loss'])
    #print('main_classifier_loss:',main_classifier_loss)
    #print('mmd_loss:',m_loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()
    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}
    #print('loss_dict: ',loss_dict)

    for k, v in loss_dict.items():
        if k!='loss':
            history[k].append(v)
    history['loss'].append(loss.cpu().detach().item())
    history['ca_loss'].append(ca_loss.cpu().detach().item())
    #print('history:',history)

    return history

def train_ca_classifier(s_datasets, t_datasets, test_data, test_label, kwargs):    
    s_train_dataloader = weighted_DataLoader(s_datasets[0], batch_size=kwargs['batch_size'])
    s_test_dataloader = weighted_DataLoader(s_datasets[1], batch_size=kwargs['batch_size'])
    t_train_dataloader = weighted_DataLoader(t_datasets[0], batch_size=kwargs['batch_size'])
    t_test_dataloader = weighted_DataLoader(t_datasets[1], batch_size=kwargs['batch_size'])


    shared_encoder = MLP(input_dim=kwargs['input_dim'],
                         output_dim=kwargs['latent_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'],
                         dop=kwargs['dop']).to(kwargs['device'])
                         
    shared_decoder = MLP(input_dim=2+kwargs['latent_dim'],
                         output_dim=kwargs['input_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'][::-1],
                         dop=kwargs['dop']).to(kwargs['device'])

    s_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    t_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    #print('t_dsnae:',t_dsnae)
    
    #cell type classifier 
    classifier = MLP(input_dim=kwargs['latent_dim'],
                     output_dim=6,
                     hidden_dims=kwargs['classifier_hidden_dims'],
                     dop=kwargs['dop']).to(kwargs['device'])
    main_classifier = EncoderDecoder(encoder=s_dsnae.shared_encoder, decoder=classifier).to(kwargs['device'])
    #print('Main classifier structure:',main_classifier)

    device = kwargs['device']

    classification_loss = nn.CrossEntropyLoss()
    
    dsnae_train_history = defaultdict(list)
    dsnae_val_history = defaultdict(list)

    main_classification_pretrain_history = defaultdict(list)
    main_classification_eval_test_history = defaultdict(list)
    main_classification_eval_t_train_history = defaultdict(list)
    main_classification_eval_t_test_history = defaultdict(list)


    ae_params = [t_dsnae.private_encoder.parameters(),
                 s_dsnae.private_encoder.parameters(),
                 shared_decoder.parameters(),
                 shared_encoder.parameters()]
                     
    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=kwargs['lr'])
    main_classifier_optimizer = torch.optim.AdamW(main_classifier.decoder.parameters(), lr=kwargs['lr'])
        
    print('train_num_epochs:',kwargs['train_num_epochs'])
    # start DSNAE pretraining        
    for epoch in range(int(kwargs['train_num_epochs'])):
        #if epoch % 50 == 0:
        print(f'----Autoencoder  Pre-training Epoch {epoch} ----')
        for step, s_batch in enumerate(s_train_dataloader):
            t_batch = next(iter(t_train_dataloader))
            dsnae_train_history = single_cluster_align_train_step(s_dsnae=s_dsnae,
                                                                  t_dsnae=t_dsnae,
                                                                  s_dataset=s_batch,
                                                                  t_dataset=t_batch,
                                                                  device=device,
                                                                  alpha=1.0,
                                                                  optimizer=ae_optimizer,
                                                                  history=dsnae_train_history)
        dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
                                                 data_loader=s_test_dataloader,
                                                 device=device,
                                                 history=dsnae_val_history
                                                 )
        dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
                                                 data_loader=t_test_dataloader,
                                                 device=device,
                                                 history=dsnae_val_history
                                                 )
        for k in dsnae_val_history:
            if k != 'best_index':
                dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                dsnae_val_history[k].pop()

        save_flag, stop_flag = model_save_check(dsnae_val_history, metric_name='loss', tolerance_count=50)
        if kwargs['es_flag']:
            if save_flag:
                torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
                torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))
            if stop_flag:
                break

    if kwargs['es_flag']:
        s_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt')))
        t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt')))
        
        
        
    # start main classifier pre-training
    for epoch in range(int(kwargs['train_num_epochs'])):
        #if epoch % 50 == 0:
        print(f'----Cell Type Classifier Pre-training Epoch {epoch}----')
        for step, s_batch in enumerate(s_train_dataloader):
            #t_batch = next(iter(t_train_dataloader))
            main_classification_pretrain_history = main_classification_train_step(classifier=main_classifier,
                                                                                      batch=s_batch,
                                                                                      loss_fn=classification_loss,
                                                                                      device=kwargs['device'],
                                                                                      optimizer=main_classifier_optimizer,
                                                                                      history=main_classification_pretrain_history)

        main_classification_eval_test_history = evaluate_main_classification_epoch2(classifier=main_classifier,
                                                                                       dataloader=s_test_dataloader,
                                                                                       datatype='source',
                                                                                       device=kwargs['device'],
                                                                                       history=main_classification_eval_test_history)
        main_classification_eval_t_train_history = evaluate_main_classification_epoch2(classifier=main_classifier,
                                                                                          dataloader=t_train_dataloader,
                                                                                        datatype='target',
                                                                                        device=kwargs['device'],
                                                                                        history=main_classification_eval_t_train_history)
        main_classification_eval_t_test_history = evaluate_main_classification_epoch2(classifier=main_classifier,
                                                                                         dataloader=t_test_dataloader,
                                                                                         datatype='target',
                                                                                         device=kwargs['device'],
                                                                                         history=main_classification_eval_t_test_history)

        save_flag, stop_flag = model_save_check(history=main_classification_eval_test_history, metric_name='acc',
                                                    tolerance_count=50)
        if kwargs['es_flag']:
            if save_flag:
                torch.save(confounder_classifier.state_dict(),
                           os.path.join(kwargs['model_save_folder'], 'adv_classifier.pt'))
            if stop_flag:
                break

    if kwargs['es_flag']:
        main_classifier.load_state_dict(
        torch.load(os.path.join(kwargs['model_save_folder'], 'main_classifier.pt')))

    print('pre-train done!')
    test_prob= main_classifier(test_data.to(device)).cpu().detach().numpy()
    test_pred=np.argmax(test_prob,axis=1)
    print('ATAC classification Acc:', accuracy_score(y_true=test_label['cell_type'], y_pred=test_pred))        

    #start alternating training
    for epoch in range(int(kwargs['train_num_epochs'])):
        #if epoch % 50 == 0:
        print(f'----Alternative training epoch {epoch}----')
        # start autoencoder training epoch
        for step, s_batch in enumerate(s_train_dataloader):
            t_batch = next(iter(t_train_dataloader))
                
            dsnae_train_history = customized_ca_classifier_train_step(main_classifier=main_classifier,
                                                                             s_dsnae=s_dsnae,
                                                                             t_dsnae=t_dsnae,
                                                                             s_dataset=s_batch,
                                                                             t_dataset=t_batch,
                                                                             loss_fn=classification_loss,
                                                                             device=device,
                                                                             optimizer=ae_optimizer,
                                                                             history=dsnae_train_history)
        dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
                                                 data_loader=s_test_dataloader,
                                                 device=device,
                                                 history=dsnae_val_history
                                                 )
        dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
                                                 data_loader=t_test_dataloader,
                                                 device=device,
                                                 history=dsnae_val_history
                                                 )                                                                 
            
        for k in dsnae_val_history:
            if k != 'best_index':
                dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                dsnae_val_history[k].pop()
                                                                                     
        main_classification_eval_test_history = evaluate_main_classification_epoch2(classifier=main_classifier,
                                                                                       dataloader=s_test_dataloader,
                                                                                       datatype='source',
                                                                                       device=kwargs['device'],
                                                                                       history=main_classification_eval_test_history)
        main_classification_eval_t_train_history = evaluate_main_classification_epoch2(classifier=main_classifier,
                                                                                          dataloader=t_train_dataloader,
                                                                                        datatype='target',
                                                                                        device=kwargs['device'],
                                                                                        history=main_classification_eval_t_train_history)
        main_classification_eval_t_test_history = evaluate_main_classification_epoch2(classifier=main_classifier,
                                                                                         dataloader=t_test_dataloader,
                                                                                         datatype='target',
                                                                                         device=kwargs['device'],
                                                                                         history=main_classification_eval_t_test_history)                    


                                                                                                   
    torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
    torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))
    torch.save(main_classifier.state_dict(), os.path.join(kwargs['model_save_folder'], 'main_classifier.pt'))

    return t_dsnae.shared_encoder, s_dsnae.private_encoder, t_dsnae.private_encoder, (dsnae_train_history, dsnae_val_history), main_classifier,(main_classification_pretrain_history,main_classification_eval_test_history,main_classification_eval_t_train_history,main_classification_eval_t_test_history)




################################
#train_ca_mmd_classifier
def customized_ca_mmd_classifier_train_step(main_classifier, s_dsnae, t_dsnae, s_dataset, t_dataset,loss_fn, device, optimizer, history, threshold=20, alpha=1.0, scheduler=None):
    main_classifier.zero_grad()
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    main_classifier.eval()
    s_dsnae.train()
    t_dsnae.train()

    s_x = s_dataset[0].to(device)
    s_y = s_dataset[1].to(device) 
    t_x = t_dataset[0].to(device)
    t_y = t_dataset[1].to(device)

    
    main_classifier_loss = loss_fn(main_classifier(s_x), s_y.squeeze(-1))
    

    s_code = s_dsnae.encode(s_x)
    t_code = t_dsnae.encode(t_x)

    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))

    ca_loss = cluster_alignment_loss(source_features=s_code, target_features=t_code, source_label=s_y, target_label= t_y, threshold=threshold, device=device)
    #print('cluster alignment loss:',ca_loss)
    #loss = s_loss_dict['loss'] + t_loss_dict['loss'] + alpha* ca_loss 

    m_loss = mmd_loss(source_features=s_code, target_features=t_code, device=device)
    loss = s_loss_dict['loss'] + t_loss_dict['loss']  + alpha * main_classifier_loss + ca_loss + m_loss #- beta * confounder_classifier_loss 
    #print('s loss:', s_loss_dict['loss'],s_loss_dict['recons_loss'],s_loss_dict['ortho_loss'])
    #print('t loss:', t_loss_dict['loss'],t_loss_dict['recons_loss'],t_loss_dict['ortho_loss'])
    #print('main_classifier_loss:',main_classifier_loss)
    #print('mmd_loss:',m_loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()
    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}
    #print('loss_dict: ',loss_dict)

    for k, v in loss_dict.items():
        if k!='loss':
            history[k].append(v)
    history['loss'].append(loss.cpu().detach().item())
    history['ca_loss'].append(ca_loss.cpu().detach().item())
    history['mmd_loss'].append(m_loss.cpu().detach().item())
    #print('history:',history)

    return history


def train_ca_mmd_classifier(s_datasets, t_datasets, test_data, test_label, kwargs): 
    '''
    s_train_dataloader = weighted_DataLoader(s_datasets[0], batch_size=kwargs['batch_size'])
    s_test_dataloader = weighted_DataLoader(s_datasets[1], batch_size=kwargs['batch_size'])
    t_train_dataloader = weighted_DataLoader(t_datasets[0], batch_size=kwargs['batch_size'])
    t_test_dataloader = weighted_DataLoader(t_datasets[1], batch_size=kwargs['batch_size'])
    '''
    
    s_train_dataloader = DataLoader(s_datasets[0], batch_size=kwargs['batch_size'])
    s_test_dataloader = DataLoader(s_datasets[1], batch_size=kwargs['batch_size'])
    t_train_dataloader = DataLoader(t_datasets[0], batch_size=kwargs['batch_size'])
    t_test_dataloader = DataLoader(t_datasets[1], batch_size=kwargs['batch_size'])
    
    
    shared_encoder = MLP(input_dim=kwargs['input_dim'],
                         output_dim=kwargs['latent_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'],
                         dop=kwargs['dop']).to(kwargs['device'])
                         
    shared_decoder = MLP(input_dim=2+kwargs['latent_dim'],
                         output_dim=kwargs['input_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'][::-1],
                         dop=kwargs['dop']).to(kwargs['device'])

    s_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    lambda2=kwargs['lambda'][1],
                    lambda3=kwargs['lambda'][2],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    t_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    lambda2=kwargs['lambda'][1],
                    lambda3=kwargs['lambda'][2],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    #print('t_dsnae:',t_dsnae)
    
    #cell type classifier 
    classifier = MLP(input_dim=kwargs['latent_dim'],
                     output_dim=13,                 #Change cluster numbers
                     hidden_dims=kwargs['classifier_hidden_dims'],
                     dop=kwargs['dop']).to(kwargs['device'])
    main_classifier = EncoderDecoder(encoder=s_dsnae.shared_encoder, decoder=classifier).to(kwargs['device'])
    #print('Main classifier structure:',main_classifier)

    device = kwargs['device']

    classification_loss = nn.CrossEntropyLoss()
    
    dsnae_train_history = defaultdict(list)
    dsnae_val_history = defaultdict(list)

    main_classification_pretrain_history = defaultdict(list)
    main_classification_eval_test_history = defaultdict(list)
    main_classification_eval_t_train_history = defaultdict(list)
    main_classification_eval_t_test_history = defaultdict(list)


    ae_params = [t_dsnae.private_encoder.parameters(),
                 s_dsnae.private_encoder.parameters(),
                 shared_decoder.parameters(),
                 shared_encoder.parameters()]
                     
    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=kwargs['lr'])
    main_classifier_optimizer = torch.optim.AdamW(main_classifier.decoder.parameters(), lr=kwargs['lr'])
        
    print('train_num_epochs:',kwargs['train_num_epochs'])
    # start DSNAE pretraining        
    for epoch in range(int(kwargs['train_num_epochs'])):
        if epoch % 100 == 0:
            print(f'---- Autoencoder  Pre-training Epoch {epoch} ----')
        for step, s_batch in enumerate(s_train_dataloader):
            t_batch = next(iter(t_train_dataloader))
            #s_label_unique = torch.unique(s_batch[1], return_counts =False)
            #print("s_batch data label number: {}".format(s_label_unique))
            #t_label_unique = torch.unique(t_batch[1], return_counts =False)
            #print("t_batch data label number: {}".format(t_label_unique))
            dsnae_train_history = ca_mmd_train_step(s_dsnae=s_dsnae,
                                                    t_dsnae=t_dsnae,
                                                    s_dataset=s_batch,
                                                    t_dataset=t_batch,
                                                    device=device,
                                                    optimizer=ae_optimizer,
                                                    history=dsnae_train_history)
        dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
                                                 data_loader=s_test_dataloader,
                                                 device=device,
                                                 history=dsnae_val_history
                                                 )
        dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
                                                 data_loader=t_test_dataloader,
                                                 device=device,
                                                 history=dsnae_val_history
                                                 )
        for k in dsnae_val_history:
            if k != 'best_index':
                dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                dsnae_val_history[k].pop()

        save_flag, stop_flag = model_save_check(dsnae_val_history, metric_name='loss', tolerance_count=50)
        if kwargs['es_flag']:
            if save_flag:
                torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
                torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))
            if stop_flag:
                break

    if kwargs['es_flag']:
        s_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt')))
        t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt')))
        
        
        
    # start main classifier pre-training
    for epoch in range(int(kwargs['train_num_epochs'])):
        if epoch % 100 == 0:
            print(f'----Cell Type Classifier Pre-training Epoch {epoch}----')
        for step, s_batch in enumerate(s_train_dataloader):
            #t_batch = next(iter(t_train_dataloader))
            main_classification_pretrain_history = main_classification_train_step(classifier=main_classifier,
                                                                                      batch=s_batch,
                                                                                      loss_fn=classification_loss,
                                                                                      device=kwargs['device'],
                                                                                      optimizer=main_classifier_optimizer,
                                                                                      history=main_classification_pretrain_history)

        main_classification_eval_test_history = evaluate_main_classification_epoch2(classifier=main_classifier,
                                                                                       dataloader=s_test_dataloader,
                                                                                       datatype='source',
                                                                                       device=kwargs['device'],
                                                                                       history=main_classification_eval_test_history)
        main_classification_eval_t_train_history = evaluate_main_classification_epoch2(classifier=main_classifier,
                                                                                          dataloader=t_train_dataloader,
                                                                                        datatype='target',
                                                                                        device=kwargs['device'],
                                                                                        history=main_classification_eval_t_train_history)
        main_classification_eval_t_test_history = evaluate_main_classification_epoch2(classifier=main_classifier,
                                                                                         dataloader=t_test_dataloader,
                                                                                         datatype='target',
                                                                                         device=kwargs['device'],
                                                                                         history=main_classification_eval_t_test_history)

        save_flag, stop_flag = model_save_check(history=main_classification_eval_test_history, metric_name='acc',
                                                    tolerance_count=50)
        if kwargs['es_flag']:
            if save_flag:
                torch.save(confounder_classifier.state_dict(),
                           os.path.join(kwargs['model_save_folder'], 'adv_classifier.pt'))
            if stop_flag:
                break

    if kwargs['es_flag']:
        main_classifier.load_state_dict(
        torch.load(os.path.join(kwargs['model_save_folder'], 'main_classifier.pt')))

    print('pre-train done!')
    test_prob= main_classifier(test_data.to(device)).cpu().detach().numpy()
    test_pred=np.argmax(test_prob,axis=1)
    print('ATAC classification Acc:', accuracy_score(y_true=test_label['cell_type'], y_pred=test_pred))        

    #start alternating training
    for epoch in range(int(kwargs['train_num_epochs'])):
        if epoch % 100 == 0:
            print(f'----Alternative training epoch {epoch}----')
        # start autoencoder training epoch
        for step, s_batch in enumerate(s_train_dataloader):
            t_batch = next(iter(t_train_dataloader))
                
            dsnae_train_history = customized_ca_mmd_classifier_train_step(main_classifier=main_classifier,
                                                                             s_dsnae=s_dsnae,
                                                                             t_dsnae=t_dsnae,
                                                                             s_dataset=s_batch,
                                                                             t_dataset=t_batch,
                                                                             loss_fn=classification_loss,
                                                                             device=device,
                                                                             optimizer=ae_optimizer,
                                                                             history=dsnae_train_history)
        dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
                                                 data_loader=s_test_dataloader,
                                                 device=device,
                                                 history=dsnae_val_history
                                                 )
        dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
                                                 data_loader=t_test_dataloader,
                                                 device=device,
                                                 history=dsnae_val_history
                                                 )                                                                 
            
        for k in dsnae_val_history:
            if k != 'best_index':
                dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                dsnae_val_history[k].pop()
                                                                                     
        main_classification_eval_test_history = evaluate_main_classification_epoch2(classifier=main_classifier,
                                                                                       dataloader=s_test_dataloader,
                                                                                       datatype='source',
                                                                                       device=kwargs['device'],
                                                                                       history=main_classification_eval_test_history)
        main_classification_eval_t_train_history = evaluate_main_classification_epoch2(classifier=main_classifier,
                                                                                          dataloader=t_train_dataloader,
                                                                                        datatype='target',
                                                                                        device=kwargs['device'],
                                                                                        history=main_classification_eval_t_train_history)
        main_classification_eval_t_test_history = evaluate_main_classification_epoch2(classifier=main_classifier,
                                                                                         dataloader=t_test_dataloader,
                                                                                         datatype='target',
                                                                                         device=kwargs['device'],
                                                                                         history=main_classification_eval_t_test_history)                    


                                                                                                   
    torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_s_dsnae.pt'))
    torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'm_t_dsnae.pt'))
    torch.save(main_classifier.state_dict(), os.path.join(kwargs['model_save_folder'], 'main_classifier.pt'))

    return t_dsnae.shared_encoder, s_dsnae.private_encoder, t_dsnae.private_encoder, (dsnae_train_history, dsnae_val_history), main_classifier,(main_classification_pretrain_history,main_classification_eval_test_history, main_classification_eval_t_train_history, main_classification_eval_t_test_history)

