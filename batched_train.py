import os
import glob
import json
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

from readers.memo_ga import MEMODataset
from utilities.model_utils import setup_data_for_gcn
from utilities.train_infer import get_model, calc_loss, stack_av_labels, get_model_nparams

torch.set_grad_enabled(True) 

def batched_train(config, trainset:MEMODataset, validset:MEMODataset, testset:MEMODataset):
    
    torch.manual_seed(config.seed)
    # Logger
    sw = SummaryWriter(os.path.join(config.checkpoint_path, 'logs'))
    best_loss = -1*float('inf')
    best_arous_loss = 0.0
    best_valen_loss = 0.0
    
    best_epoch = 0
    patience = config.patience
    best_model_weights = None

    # Data Loader 
    train_loader = DataLoader(trainset, num_workers=config.get('num_workers', 0), shuffle=False, sampler=None, batch_size=config.batch_size, pin_memory=True, drop_last=True)
    test_loader = DataLoader(testset, num_workers=config.get('num_workers', 0), shuffle=False, sampler=None, batch_size=config.batch_size, pin_memory=True, drop_last=True)
    
    # Calculating number of features used
    for i, batch in enumerate(train_loader):
        grp, ses, onsets, offsets, arous_mean, valen_mean, arous_gt, valen_gt, features = batch
        config.__dict__["nfeatures"] = features.shape[-1]
        break
    
    # Load Model (new or from checkpoint, using config)
    model = get_model(config)
    model.train()
    print("Model #params = ", (get_model_nparams(model)))

    # Optimizer
    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.AdamW(model.parameters(), config.learning_rate, betas=[config.adam_b1, config.adam_b2], weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)

    for epoch in range(config.nepoch):
        print("Epoch: ", epoch, " and Current Patience: ", patience)
        # Iterator Batch
        for i, batch in enumerate(train_loader):
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            
            grp, ses, onsets, offsets, arous_mean, valen_mean, arous_gt, valen_gt, features = batch
            y_train = stack_av_labels(config, arous_gt, valen_gt, arous_mean, valen_mean)
            
            # Normalize batch
            # Make this also compatible with 3D features, so transform only the last dimension
            scaler = StandardScaler()
            if features.dim() == 3:
                # Apply scaler to each slice of the last dimension
                num_segments, num_interloc, num_features = features.shape
                for t in range(num_interloc):
                    features[:, t, :] = torch.from_numpy(scaler.fit_transform(features[:, t, :].numpy()))
            else:
                features = torch.from_numpy(scaler.fit_transform(features.numpy()))
            
            # Forward Pass
            if config.model_name == "gcn":
                data = setup_data_for_gcn(grp, ses, features, y_train, config.batch_size)
                y_train_pred, attn_wgts = model.forward(data.x, data.edge_index)
            else:
                y_train_pred = model.forward(features)

            # Loss Calc (using config)
            arous_loss, valen_loss, total_train_loss = calc_loss(config.loss_term, y_train, y_train_pred)
            total_train_loss = 1 - total_train_loss
            # Back Propagation
            total_train_loss.backward()
            # Adjust learning weights
            optimizer.step() 
                
        # Tensorboard summary logging
        if epoch % config.train_summary_interval == 0:            
            sw.add_scalar("training/arous_"+config.loss_term, arous_loss, epoch)
            sw.add_scalar("training/valen_"+config.loss_term, valen_loss, epoch)
            sw.add_scalar("training/total_"+config.loss_term, (arous_loss+valen_loss)/2, epoch)
        
        # Tensorboard summary logging
        if epoch % config.eval_summary_interval == 0:
            # Print/Logger loss values
            # print("Evaluating Validation Set !!!!")
            infer(config, model, validset, sw, log_tag="validation", epoch=epoch)
            print()
            # print("Evaluating Test Set !!!!")
            infer(config, model, testset, sw, log_tag="test", epoch=epoch)
            print()

        # Early-Stopping
        infer_arous_loss, infer_valen_loss, infer_total_loss = infer(config, model, testset, sw, log_tag="test", epoch=epoch, tb_log=False)
        
        # Keep track of best model and save it
        if infer_total_loss > best_loss:
            print("Updating Best Model......")
            best_loss = infer_total_loss
            best_arous_loss = infer_arous_loss
            best_valen_loss = infer_valen_loss
            
            best_epoch = epoch
            # best_model_weights = deepcopy(model.state_dict())  # Deep Best model weights copy    
            save_best_model(model, optimizer, epoch, best_loss, file_path="{}/best_model_{}".format(config.checkpoint_path, epoch))  
            patience = config.patience  # Reset patience counter
        else:
            patience = patience - 1
            if patience == 0:
                break
       
        scheduler.step()
        model.train()

    torch.save(best_model_weights, "{}/best_model_{}".format(config.checkpoint_path, best_epoch))
    best_losses_dict = {"best_loss": best_loss.item(), "best_arous_loss":best_arous_loss.item(), "best_valen_loss":best_valen_loss.item(), "best_epoch":best_epoch}
    print(best_losses_dict)
    with open("{}/best_loss.json".format(config.checkpoint_path), 'w') as fp:
        json.dump(best_losses_dict, fp)
    
# Function to save the best model
def save_best_model(model, optimizer, epoch, best_loss, file_path="best_model.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss
    }
    torch.save(checkpoint, file_path) 

def infer(config, model, dataset, sw, log_tag, epoch, tb_log=True):
    loader = DataLoader(dataset, num_workers=config.get('num_workers', 0), shuffle=False, sampler=None, batch_size=config.batch_size, pin_memory=True, drop_last=True)

    model.eval()

    total_arous_loss = 0.0
    total_valen_loss = 0.0
    total_infer_loss = 0.0
    
    with torch.no_grad():
        # Iterator Batch
        for i, batch in enumerate(loader):
            grp, ses, onsets, offsets, arous_mean, valen_mean, arous_gt, valen_gt, features = batch
            
            y = stack_av_labels(config, arous_gt, valen_gt, arous_mean, valen_mean)
            
            # Normalize batch
            # Make this also compatible with 3D features, so transform only the last dimension
            scaler = StandardScaler()
            if features.dim() == 3:
                # Apply scaler to each slice of the last dimension
                num_segments, num_interloc, num_features = features.shape
                for t in range(num_interloc):
                    features[:, t, :] = torch.from_numpy(scaler.fit_transform(features[:, t, :].numpy()))
            else:
                features = torch.from_numpy(scaler.fit_transform(features.numpy()))
            
            # Forward Pass
            if config.model_name == "gcn":
                data = setup_data_for_gcn(grp, ses, features.float(), y, config.batch_size)
                y_pred, attn_wgts = model.forward(data.x, data.edge_index)
            else:
                y_pred = model.forward(features.float())
                
            # Loss Calc (using config)
            arous_loss, valen_loss, total_loss = calc_loss(config.loss_term, y, y_pred)

            total_arous_loss = total_arous_loss + arous_loss
            total_valen_loss = total_valen_loss + valen_loss
            total_infer_loss = total_infer_loss + total_loss
   
    total_arous_loss = total_arous_loss/len(loader)
    total_valen_loss = total_valen_loss/len(loader)
    total_infer_loss = (total_arous_loss + total_valen_loss)/2
    
    if tb_log:
        # print("Arousal loss CCC: ", total_arous_loss)    
        # print("Valence loss CCC: ", total_valen_loss)    
        # print("Total loss CCC: ", total_infer_loss)    
        
        sw.add_scalar(log_tag+"/arous_"+config.loss_term, total_arous_loss, epoch)
        sw.add_scalar(log_tag+"/valen_"+config.loss_term, total_valen_loss, epoch)
        sw.add_scalar(log_tag+"/total_"+config.loss_term, total_infer_loss, epoch)
    
    return total_arous_loss, total_valen_loss, total_infer_loss