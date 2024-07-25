import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

def adjust_learning_rate(optimizer, epoch): # From Notes (07_pytorch_tutorial)
    '''Lowers learning rate for passed optimizer relative to passed epoch'''
    
    # Start with default value
    lr = 0.001
    
    # Check current epoch and assign learning rate accordingly
    if (epoch > 5):     
        lr = 0.0001
    if (epoch >= 10):
        lr = 0.00001
    if (epoch > 20):
        lr = 0.000001
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # Sets learning rate to new value inside optimizer

def save_checkpoint(ckp_path, model, epoch, optimizer, global_step):
    '''Saves a checkpoint to specified path (ckp_path) containing the model state dictionary, the current global step
        and epoch number, and the optimizer state dictionary'''

    ## save checkpoint to ckp_path:
    ckp_path = ckp_path + 'checkpoint.pt' 
    epoch += 1
    checkpoint = {'epoch': epoch,
                  'global_step': global_step,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, ckp_path)
    
def load_checkpoint(ckp_path, model, optimizer):
    '''Loads checkpoint for specified model and optimizer and returns the epoch and global step count of the checkpoint to
        ensure that the model picks up where it left off'''

    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoches = checkpoint['epoch']
    steps = checkpoint['global_step']
    
    return (epoches, steps)