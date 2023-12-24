import torch
import pandas as pd
import data
from functions import *
from dsn_ae import DSNAE
from mlp import MLP
import itertools
import pickle

def main(update_params_dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'

    rna_df = pd.read_csv(data.snare_adbrain_rna_file, index_col=0)
    atac_df = pd.read_csv(data.snare_adbrain_atac_file, index_col=0)
    rna_label = pd.read_csv(data.snare_adbrain_rna_label_file, index_col=0)
    atac_label = pd.read_csv(data.snare_adbrain_atac_label_file, index_col=0) 

        
    output_folder = '/data/jliu25/MyProject/code_snare_adbrain_balanced_supervised/output/asap/ca_mmd_classifier_balanced_117/'
    safe_make_dir(output_folder)
    training_params={
        'device': device,
        'batch_size': 128,
        'encoder_hidden_dims': [512, 256],
        'classifier_hidden_dims': [32, 16],
        'latent_dim': 64,
        'input_dim': rna_df.shape[-1],
        'lr': 1e-3,
        'es_flag': False,
        'norm_flag': True,
        'lambda': [1.0, 1.0, 0.2],
        'dop': 0.0,
        'es_flag': False}
    
    update_params_dict['lambda'] = training_params['lambda']
    param_str = dict_to_str(update_params_dict)
    training_params.update(update_params_dict)
    training_params.update({'model_save_folder': os.path.join(output_folder, param_str)})
    safe_make_dir(training_params['model_save_folder'])

    s_datasets, t_datasets, new_batch_size = get_dataloader_generator(
        rna_df=rna_df,
        atac_df=atac_df,
        rna_label=rna_label,
        atac_label=atac_label,
        batch_size=training_params['batch_size'],
        seed=2022)
    
    training_params['batch_size'] = new_batch_size
    print('training_params:',training_params)
    
    rna_dataset = torch.from_numpy(rna_df.values.astype('float32'))
    atac_dataset = torch.from_numpy(atac_df.values.astype('float32'))    

       
    shared_encoder,s_private_encoder,t_private_encoder, historys, classifier, classfier_historys = train_ca_mmd_classifier(s_datasets=s_datasets,
                                                                                                                           t_datasets=t_datasets,
                                                                                                                           test_data=atac_dataset,
                                                                                                                           test_label=atac_label,
                                                                                                                           kwargs=training_params)

    rna_hidden= shared_encoder(rna_dataset.to(device)).cpu().detach().numpy()
    atac_hidden= shared_encoder(atac_dataset.to(device)).cpu().detach().numpy()

    train_prob= classifier(rna_dataset.to(device)).cpu().detach().numpy()
    train_pred=np.argmax(train_prob,axis=1)
    print('RNA classification Acc:', accuracy_score(y_true=rna_label['cell_type'], y_pred=train_pred))
    test_prob= classifier(atac_dataset.to(device)).cpu().detach().numpy()
    test_pred=np.argmax(test_prob,axis=1)
    print('ATAC classification Acc:', accuracy_score(y_true=atac_label['cell_type'], y_pred=test_pred))  
    np.savetxt(os.path.join(training_params['model_save_folder'], f'atac_pseudolabel.csv'), test_pred ,delimiter=',')

    np.savetxt(os.path.join(training_params['model_save_folder'], f'rna_hidden.csv'), rna_hidden ,delimiter=',')
    np.savetxt(os.path.join(training_params['model_save_folder'], f'atac_hidden.csv'), atac_hidden ,delimiter=',')
    rna_pseudotime= s_private_encoder(rna_dataset.to(device)).cpu().detach().numpy()
    atac_pseudotime= t_private_encoder(atac_dataset.to(device)).cpu().detach().numpy()
    np.savetxt(os.path.join(training_params['model_save_folder'], f'rna_pseudotime.csv'), rna_pseudotime ,delimiter=',')
    np.savetxt(os.path.join(training_params['model_save_folder'], f'atac_pseudotime.csv'), atac_pseudotime ,delimiter=',')
    
    with open(os.path.join(training_params['model_save_folder'], f'dsnae_train_history.pickle'),'wb') as f:
        pickle.dump(historys[0], f)
    with open(os.path.join(training_params['model_save_folder'], f'dsnae_test_history.pickle'),'wb') as f:
        pickle.dump(historys[1], f)  
    with open(os.path.join(training_params['model_save_folder'], f'classifier_train_history.pickle'),'wb') as f:
        pickle.dump(classfier_historys[0], f)
    with open(os.path.join(training_params['model_save_folder'], f'classifier_test_history.pickle'),'wb') as f:
        pickle.dump(classfier_historys[1], f)
    with open(os.path.join(training_params['model_save_folder'], f'classifier_t_train_history.pickle'),'wb') as f:
        pickle.dump(classfier_historys[2], f)
    with open(os.path.join(training_params['model_save_folder'], f'classifier_t_test_history.pickle'),'wb') as f:
        pickle.dump(classfier_historys[3], f)
        
if __name__ == '__main__':
    params_grid = {
        "train_num_epochs": [2, 1000, 1500, 2000, 2500, 3000],
        #"train_num_epochs": [2],
        "dop": [0.0]
    }

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for param_dict in update_params_dict_list:
        main(update_params_dict=param_dict)