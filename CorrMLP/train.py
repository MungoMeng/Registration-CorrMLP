import os
import glob
import sys
import random
import time
import torch
import numpy as np
import scipy.ndimage
from argparse import ArgumentParser

# project imports
import datagenerators
import networks
import losses


def Dice(vol1, vol2, labels=None, nargout=1):
    
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)
    
    
def NJD(displacement):

    D_y = (displacement[1:,:-1,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_x = (displacement[:-1,1:,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_z = (displacement[:-1,:-1,1:,:] - displacement[:-1,:-1,:-1,:])

    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])
    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])
    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
    Ja_value = D1-D2+D3
    
    return np.sum(Ja_value<0)


def train(train_dir,
          train_pairs,
          valid_dir, 
          valid_pairs,
          model_dir,
          load_model,
          device,
          initial_epoch,
          epochs,
          steps_per_epoch,
          batch_size):

    # preparation
    train_pairs = np.load(train_dir+train_pairs, allow_pickle=True)
    valid_pairs = np.load(valid_dir+valid_pairs, allow_pickle=True)

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # device handling
    if 'gpu' in device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'

    # prepare the model
    model = networks.CorrMLP()
    model.to(device)
    if load_model != './':
        print('loading', load_model)
        state_dict = torch.load(load_model, map_location=device)
        model.load_state_dict(state_dict)
    
    # transfer model
    SpatialTransformer = networks.SpatialTransformer_block(mode='nearest')
    SpatialTransformer.to(device)
    SpatialTransformer.eval()
    
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # prepare losses
    Losses = [losses.NCC(win=9).loss, losses.Grad('l2').loss]
    Weights = [1.0, 1.0]
            
    # data generator
    train_gen_pairs = datagenerators.gen_pairs(train_dir, train_pairs, batch_size=batch_size)
    train_gen = datagenerators.gen_s2s(train_gen_pairs, batch_size=batch_size)

    # training/validate loops
    for epoch in range(initial_epoch, epochs):
        start_time = time.time()
        
        # training
        model.train()
        train_losses = []
        train_total_loss = []
        for step in range(steps_per_epoch):
            
            # generate inputs (and true outputs) and convert them to tensors
            inputs, labels = next(train_gen)
            inputs = [torch.from_numpy(d).to(device).float() for d in inputs]
            labels = [torch.from_numpy(d).to(device).float() for d in labels]

            # run inputs through the model to produce a warped image and flow field
            pred = model(*inputs)

            # calculate total loss
            loss = 0
            loss_list = []
            for i, Loss in enumerate(Losses):
                curr_loss = Loss(labels[i], pred[i]) * Weights[i]
                loss_list.append(curr_loss.item())
                loss += curr_loss
            train_losses.append(loss_list)
            train_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # validation
        model.eval()
        valid_Dice = []
        valid_NJD = []
        for valid_pair in valid_pairs:
            
            # generate inputs (and true outputs) and convert them to tensors
            fixed_vol, fixed_seg = datagenerators.load_by_name(valid_dir, valid_pair[0])
            fixed_vol = torch.from_numpy(fixed_vol).to(device).float()
            fixed_seg = torch.from_numpy(fixed_seg).to(device).float()
            
            moving_vol, moving_seg = datagenerators.load_by_name(valid_dir, valid_pair[1])
            moving_vol = torch.from_numpy(moving_vol).to(device).float()
            moving_seg = torch.from_numpy(moving_seg).to(device).float()

            # run inputs through the model to produce a warped image and flow field
            with torch.no_grad():
                pred = model(fixed_vol, moving_vol)
                warped_seg = SpatialTransformer(moving_seg, pred[1])
                
            warped_seg = warped_seg.detach().cpu().numpy().squeeze()
            fixed_seg = fixed_seg.detach().cpu().numpy().squeeze()
            Dice_val = Dice(warped_seg, fixed_seg)
            valid_Dice.append(Dice_val)
            
            flow = pred[1].detach().cpu().permute(0, 2, 3, 4, 1).numpy().squeeze()
            NJD_val = NJD(flow)
            valid_NJD.append(NJD_val)
        
        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, epochs)
        time_info = 'Total %.2f sec' % (time.time() - start_time)
        train_losses = ', '.join(['%.4f' % f for f in np.mean(train_losses, axis=0)])
        train_loss_info = 'Train loss: %.4f  (%s)' % (np.mean(train_total_loss), train_losses)
        valid_Dice_info = 'Valid DSC: %.4f' % (np.mean(valid_Dice))
        valid_NJD_info = 'Valid NJD: %.2f' % (np.mean(valid_NJD))
        print(' - '.join((epoch_info, time_info, train_loss_info, valid_Dice_info, valid_NJD_info)), flush=True)
    
        # save model checkpoint
        torch.save(model.state_dict(), os.path.join(model_dir, '%02d.pt' % (epoch+1)))
    

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--train_dir", type=str,
                        dest="train_dir", default='./',
                        help="training folder")
    parser.add_argument("--train_pairs", type=str,
                        dest="train_pairs", default='train_pairs.npy',
                        help="training pairs(.npy)")
    parser.add_argument("--valid_dir", type=str,
                        dest="valid_dir", default='./',
                        help="validation folder")
    parser.add_argument("--valid_pairs", type=str,
                        dest="valid_pairs", default='valid_pairs.npy',
                        help="validation pairs(.npy)")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='./models/',
                        help="models folder")
    parser.add_argument("--load_model", type=str,
                        dest="load_model", default='./',
                        help="load model file to initialize with")
    parser.add_argument("--device", type=str, default='gpu0',
                        dest="device", help="cpu or gpuN")
    parser.add_argument("--initial_epoch", type=int,
                        dest="initial_epoch", default=0,
                        help="initial epoch")
    parser.add_argument("--epochs", type=int,
                        dest="epochs", default=100,
                        help="number of epoch")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=1000,
                        help="iterations of each epoch")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=1,
                        help="batch size")

    args = parser.parse_args()
    train(**vars(args))
