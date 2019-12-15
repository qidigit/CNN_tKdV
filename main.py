import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
# Matplotlib
import matplotlib.pyplot as plt
import scipy.io

import os
import sys
import urllib
import gflags
import time

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init


from config import cfg
from model_peri import MSDNet  # periodic boundary in space

def train(net, train_loader, optimizer, epochs, scheduler=None, device="cpu", weights=None, save_epoch=100, continue_epoch=1):
    # Default arguments
    ncuda = torch.cuda.device_count()
    if weights is None:
        weights=cfg.WEIGHTS
    
    # load training data set
    data_full, target_full = train_loader
    nbatch = cfg.BATCH_SIZE
    sbatch = int(data_full.size()[0] / nbatch)
    losses = np.zeros(epochs*sbatch)
    mean_losses = np.zeros(epochs*sbatch)
    accs = np.zeros(epochs*sbatch)
    mean_accs = np.zeros(epochs*sbatch)

    all_losses = []
    all_accs = []

    loss_type = cfg.LOSS_TYPE
    if loss_type == 'mse':
        criterion = nn.MSELoss(reduction='mean')
    elif loss_type == 'l1e':
        criterion = nn.L1Loss(reduction='mean')
    elif loss_type == 'kld':
        criterion = nn.KLDivLoss(reduction='batchmean')
        scale_t = cfg.KLD_TEMP
        loss_alpha = cfg.LOSS_WEIGHT
    iter_ = 0
    
    start_t = time.time()
    for e in range(continue_epoch, epochs + 1):
        net.train()

        epoch_losses = []
        epoch_accs = []
        
        rand_idx = random.sample(range(0,data_full.size()[0]),data_full.size()[0])
        data_full = data_full[rand_idx, :,:,:]
        target_full = target_full[rand_idx, :,:,:]
        for ib in range(0, sbatch):
            inputs = data_full[ib*nbatch:(ib+1)*nbatch, :, :, :].to(device)
            targets_org = target_full[ib*nbatch:(ib+1)*nbatch, :, :, :].to(device)
            optimizer.zero_grad()
            outputs_org = net(inputs)
            if loss_type == 'kld':
                outputs = F.log_softmax(outputs_org/scale_t[0], dim=2)
                targets = F.softmax(targets_org/scale_t[0], dim=2)  # normalize target as a distribution
                outputs_n = F.log_softmax(outputs_org/scale_t[1], dim=2)
                targets_n = F.softmax(targets_org/scale_t[1], dim=2)
            elif loss_type == 'l1e' or loss_type == 'mse':
                outputs = outputs_org
                targets = targets_org
                
            loss = criterion(outputs, targets)
            if loss_type == 'kld':
                loss += loss_alpha * criterion(outputs_n, targets_n);
                

            loss.backward()
            optimizer.step()
            
            # Get current images
            init = inputs.data.cpu().numpy()[0,0,:,:]
            pred = outputs_org.data.cpu().numpy()[0,0,:,:]
            gt = targets_org.data.cpu().numpy()[0,0,:,:]
            pred_norm = np.exp(outputs.data.cpu().numpy()[0,0,:,:])
            gt_norm = targets.data.cpu().numpy()[0,0,:,:]

            # get current loss
            epoch_losses.append(loss.data.cpu().numpy())
            # get current accuracy
            acc = float(np.square(pred - gt).sum() / np.square(gt).sum())
            acc_norm = float(np.square(pred_norm - gt_norm).sum() / np.square(gt_norm).sum())
            epoch_accs.append(acc)

            losses[iter_] = loss.data.cpu().numpy()
            accs[iter_] = acc
            mean_losses[iter_] = np.mean(losses[max(0,iter_-10):iter_+1])
            mean_accs[iter_] = np.mean(accs[max(0,iter_-10):iter_+1])
            iter_ += 1
            if iter_ % 20 == 0:
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc_norm: {:.2f}\tAccuracy: {:.2f}'.format(
                    e, epochs, ib, sbatch,
                    100. * ib / sbatch, np.mean(epoch_losses), acc_norm, np.mean(epoch_accs)))

            del(inputs, targets, outputs, loss)
            if loss_type == 'kld':
                del(outputs_n, targets_n)
            torch.cuda.empty_cache()

        if scheduler is not None:
                scheduler.step()
        # Save current epoch loss and accuracy
        current_epoch_loss = np.mean(epoch_losses)
        current_epoch_acc = np.mean(epoch_accs)

        all_losses.append(current_epoch_loss)
        all_accs.append(current_epoch_acc)

        end_t = time.time()
        print('-'*80)
        print("Epoch {}/{} |\t Loss: {} |\t Accuracy: {}".format(e, epochs, current_epoch_loss, current_epoch_acc))
        print("collapse time = {}s".format(end_t - start_t))
        print('-'*80)
        
        # Check if current loss is better than last 3 losses
        if e % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            torch.save(net.state_dict(), cfg.exp_dir + cfg.CHECK_PATH.format(e)+'_cuda{}epochs{}_'.format(ncuda, epochs)+cfg.LOSS_TYPE)
            
    nlayers = cfg.N_LAYERS
    torch.save({'model_state_dict': net.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()}, cfg.exp_dir + cfg.TRAIN_PATH + cfg.DATASET+
                '_final'+cfg.LABELS[0].format(nlayers, cfg.FIL_SIZE[0],cfg.DIL_M[0], int(loss_alpha*10), epochs)+cfg.LOSS_TYPE)

    # Save all losses and Accuracies in a numpy array
    np.savez(cfg.exp_dir + cfg.TRAIN_PATH + cfg.DATASET+
             '_loss'+cfg.LABELS[0].format(nlayers, cfg.FIL_SIZE[0],cfg.DIL_M[0],int(loss_alpha*10), epochs)+cfg.LOSS_TYPE, 
            all_losses=np.asarray(all_losses), mean_losses=np.asarray(mean_losses))
    np.savez(cfg.exp_dir + cfg.TRAIN_PATH + cfg.DATASET+
             '_acc'+cfg.LABELS[0].format(nlayers, cfg.FIL_SIZE[0],cfg.DIL_M[0],int(loss_alpha*10), epochs)+cfg.LOSS_TYPE, 
             all_accs=np.asarray(all_accs), mean_accs=np.asarray(mean_accs), 
             init=init, gt=gt, pred=pred, gt_norm=gt_norm, pred_norm=pred_norm)



def test(net, all=False, stride=None, batch_size=None, window_size=None, device='cpu'):
    # Default params
    if stride is None:
        stride=cfg.WINDOW_WIDTH  #32
    
    if batch_size is None:
        batch_size=cfg.BATCH_SIZE  #100
    
    if window_size is None:
        window_size=cfg.WINDOW_HEIGHT  #(32, 50)
    
    # Use the network on the test set
    test_files = cfg.TEST_FILE
    test_load = scipy.io.loadmat(test_files)
    samp_test = np.transpose(test_load.get('samp_u'), (1,0,2))
    npred = window_size
    nstart = 0
    ntarg = nstart+npred
    init_set = torch.from_numpy(samp_test[:batch_size, :, nstart:nstart+npred])
    init = init_set[:, None, :,:]
    targ_set = torch.from_numpy(samp_test[:batch_size, :, ntarg:ntarg+npred])
    targ = targ_set[:, None, :,:]
    if cfg.precision == 'float':
        init = init_set.float()
        targ = targ_set.float()

    del(init_set, targ_set, samp_test)

    with torch.no_grad():
        # Switch the network to inference mode
        net.eval()

        # Do the inference
        init = init.to(device)
        pred = net(init)
    
    pred = pred.data.cpu().numpy()
    targ = targ.data.cpu().numpy()
    init = init.data.cpu().numpy()

    acc = np.zeros(batch_size)
    acc_norm = np.zeros(batch_size)
    w_measure = window_size
    # Start a loop to get accuracy
    for ib in range(0, batch_size):
        inp = pred[ib,0,:stride,:w_measure]
        out = targ[ib,0,:stride,:w_measure]
        acc[ib] = float( np.square(inp-out).sum() / np.square(out).sum() )
        inp_norm = np.exp(inp)
        out_norm = np.exp(out)
        acc_norm[ib] = float( np.square(inp_norm-out_norm).sum() / np.square(out_norm).sum() )

    print('Prediction Error Mean = {:.6f}, Normed Mean = {:.6f}'.format(np.mean(acc), np.mean(acc_norm)))
    print('Prediction Error Variance = {:.6f}, Normed Variance = {:.6f}'.format(np.var(acc), np.var(acc_norm)))

    if all:
        return acc, acc_norm, pred, targ, init
    else:
        return acc

def main(save=False, pretrained=True, task='test'):
    # instantiate the network
    if cfg.precision == 'double':
        net = MSDNet().double()
    elif cfg.precision == 'float':
        net = MSDNet()

    # Load the model on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('The code is run by {}'.format(device))
    if torch.cuda.device_count() > 1:
        print("using ", torch.cuda.device_count(), " GPUs!")
        net = nn.DataParallel(net)
        
    net.to(device)

    # Load the datasets
    data_files = cfg.DATA_FILE
    data_load = scipy.io.loadmat(data_files)

    nsample = data_load.get('MC')[0,0]  #10000
    ntrain = cfg.WINDOW_HEIGHT #50, 50; training time-series length
    Ttrain = cfg.T_START   #200; start time from data
    Ttarg = Ttrain+ntrain  #50
    samp = np.transpose(data_load.get('samp_u'), (1,0,2))
    train_set = torch.from_numpy(samp[:nsample, :, Ttrain:Ttrain+ntrain])
    train_set = train_set[:, None, :,:]
    target_set = torch.from_numpy(samp[:nsample, :, Ttarg:Ttarg+ntrain])
    target_set = target_set[:, None, :,:]
    if cfg.precision == 'double':
        train_loader = (train_set, target_set)
    elif cfg.precision == 'float':
        train_loader = (train_set.float(), target_set.float())
        
    del(samp, train_set, target_set, data_load)
    torch.cuda.empty_cache()
    # Design the optimizer
    # Change LR for Adam optimizer
    base_lr = 0.0005
    params_dict = dict(net.named_parameters())
    params = []
    for key, value in params_dict.items():
        params += [{'params':[value],'lr': base_lr}]

    # Change to Adam optimizer
    optimizer = optim.SGD(net.parameters(), lr=base_lr)
    # We define the scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [500, 750, 900], gamma=0.5)

    continue_epoch = 1
    if pretrained:
        # load the pretrained model
        net.load_state_dict(torch.load(cfg.exp_dir  + cfg.model_final)['model_state_dict'])
        if 'epoch' in cfg.model_final:
            continue_epoch = int(cfg.model_final.split('_')[-2]) + 1

    if task == 'train':
        # Train model
        train(net, train_loader, optimizer, 1000, scheduler, device, continue_epoch=continue_epoch)
    elif task == 'test':
        # Run tests
        accuracy, acc_norm, all_preds, all_gts, all_inputs = test(net, all=True, batch_size=500, device=device)

    if save:
        if task == 'test':
            np.savez(cfg.exp_dir + cfg.PRED_PATH + cfg.DATASET+
                 '_pred'+cfg.TEST_FILE[-6:-4]+cfg.LABELS[0].format(cfg.N_LAYERS, cfg.FIL_SIZE[0],cfg.DIL_M[0],int(10), 1000), 
                 accuracy=np.asarray(accuracy), all_preds=np.asarray(all_preds), all_gts=np.asarray(all_gts),
                 all_inputs=np.asarray(all_inputs), acc_norm=np.asarray(acc_norm) )
       
def get_config():
    return cfg

if __name__ == '__main__':
    gflags.DEFINE_boolean('write_data', True, 'Save the outputs produced by the model')
    gflags.DEFINE_boolean('pretrained', False, 'Use pretrained model')

    gflags.DEFINE_string('exp_dir', './experiment/', 'Path to experiment dump directory')
    gflags.DEFINE_string('cfg', './experiment/cfg.yml', 'Path to experiment configuration file')

    gflags.DEFINE_boolean('train', True, 'Train the network')

    gflags.FLAGS(sys.argv)
    cfg.init_paths(gflags.FLAGS.cfg, gflags.FLAGS.exp_dir)

    task = 'test'
    if gflags.FLAGS.train is True:
        task = 'train'

    main(save=gflags.FLAGS.write_data, pretrained=gflags.FLAGS.pretrained, task=task)
