import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchsummary import summary
from Load_Data import load_data
from Model import NonImageMLP as diag_model
from Utilities import *

 
 ### Functions ###
    
def train_model(input_names, model, label_name, train_dataloader, batch_size, num_epoches, loss_func, optimizer,
                writer, device, output_name, start_step=0, start_epoch=0, patience=3, print_every=10):
    
    step = 0
    prev_acc = 0
    lack_of_improvement = 0
    ckp_path = "./checkpoint/"
    
    model.train() 
    for epoch in range(num_epoches):
        adjust_learning_rate(optimizer, epoch)
        
        for batch_id, data in enumerate(train_dataloader): 
            step += 1

            # Inputs for minibatch and training model
            x = Variable(data[input_names[0]]).to(device).to(torch.float32)
            labels = Variable(data[label_name]).to(device).to(torch.float32)
            
            # Feed into Conversion model
            outputs = model(x) # Get the predictions
            
            # Feed into combo model and get loss
            loss = loss_func(outputs, labels)

            # Back propogation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics to print while training
            predictions = torch.round(outputs.data)
            train_acc = (predictions==labels).sum().item() / batch_size
            
            if step % print_every == 0:    # print every specified number of minibatches
                print('epoch: {}, step: {}, loss: {}, train_acc: {}'.format(epoch+1, step, loss, train_acc))
                writer.add_scalar("Loss/train global step", loss, step)
                writer.add_scalar("Acc/train global step", train_acc, step)
            
        # Early Stopping and checkpoint saving
        if train_acc - prev_acc < 0: # Keep track of if we need to early stop
            lack_of_improvement += 1
        else:
            lack_of_improvement = 0
            
        if lack_of_improvement == 0: # Only save model if it is current best
            save_checkpoint(ckp_path, model, epoch, optimizer, step)
            
        if lack_of_improvement >= patience:  # Early Stop
            break
        
        prev_acc = train_acc
        
    # Finish training
    writer.flush()
    print()
    print('Finished Training')
    path_file_path = './Path files/' + output_name + '.pth'
    torch.save(model.state_dict(), path_file_path)

def test_model(input_names, model, label_name, test_dataloader, device, output=True, output_name='recentmodel1'):
    '''Run passed model against passed test dataloader per the passed label and print test statistics (if output=True).
        Also needs the device (GPU(s) or CPU). Requires input_names list with desired input name at 0th index'''
        
    model.eval()
    
    true_pos = 0
    true_neg = 0
    fal_pos = 0
    fal_neg = 0
    
    with torch.no_grad(): # Freezes model parameters   
        for data in test_dataloader:
            x = Variable(data[input_names[0]]).to(device).to(torch.float32)
            labels = Variable(data[label_name]).to(device).to(torch.float32) 
            
            #x = img_net(images)
            outputs = model(x) # Get the predictions
            predicted = torch.round(outputs.data) # Collect the most likely class per score
                
            for idx in range(len(predicted)):
                if labels[idx] == predicted[idx]:
                    if labels[idx] == 0:
                        true_neg += 1
                    else:
                        true_pos += 1
                else:
                    if labels[idx] == 0:
                        fal_neg += 1
                    else:
                        fal_pos += 1

        # Calculate test statistics
        acc = 100 * (true_pos + true_neg) / (true_pos + true_neg + fal_pos + fal_neg)
        prec = 100 * true_pos / (true_pos + fal_pos)
        rec = 100 * true_pos / (true_pos + fal_neg)
        spec = 100 * true_neg / (true_neg + fal_pos)
        npv = 100 * true_neg / (fal_neg + true_neg)
        F1 = (2 * prec * rec) / (prec + rec)

        # Print test statistics and confusion matrix
        if output:
            lines = ['',
                     '-------------------------------------',
                     '          Testing Accuracy: {0:.2f} % '.format(acc),
                     '     Testing Precision/PPV: {0:.2f} % '.format(prec),
                     'Testing Recall/Sensitivity: {0:.2f} % '.format(rec),
                     '       Testing Specificity: {0:.2f} % '.format(spec),
                     '                       NPV: {0:.2f} % '.format(npv),
                     '                Testing F1: {0:.2f} % '.format(F1),
                     '-------------------------------------',
                     '',
                     str([true_pos, fal_pos]),
                     str([fal_neg, true_neg])]
            
            name1 = output_name + '.txt'
            with open(name1, 'w') as f:
                for line in lines:
                    print(line)
                    f.write(line)
                    f.write('\n')
        
### Main Program ###        
def full_process(input_names, label_name, num_epochs, learning_rate, batch_size, img_size, device, load_chckpnt, output_name='recentmodel1'):
    '''Runs the training process and then performs the test set evaluation. Passed a model object, the model inputs as a 
        list with the desired input name as the 0th index element, the label to train and test against, the hyperparameters 
        of number of epochs, the base learning rate, and the batch size, the size of images for image models, 
        and the device (CPU or GPU)'''

    import os

    # Create Model
    model = diag_model()
    model.to(device)

    # Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    loss_func = nn.BCELoss()

    # Other set up for training
    writer = SummaryWriter()
    train_dataloader, test_dataloader = load_data(batch_size, 224)
    start_step = 0
    used_epochs = 0

    # Load checkpoint if needed
    if load_chckpnt and os.path.isfile("./checkpoint/checkpoint.pt"):
        print("Loaded checkpoint")
        used_epochs, start_step = load_checkpoint("./checkpoint/checkpoint.pt", model, optimizer)
        
    # Train the model
    train_model(input_names, model, label_name, train_dataloader, batch_size, num_epochs, loss_func, 
                    optimizer, writer, device, output_name, start_step, start_epoch=used_epochs, print_every=1)
    writer.close()

    # Test the model
    test_model(input_names, model, label_name, test_dataloader, device, output_name=output_name)