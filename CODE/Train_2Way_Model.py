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
from Model import ResNet18Model as img_model
from Model import CombinerMLP as combo_model
from Utilities import *
 
### Functions ###
    
def train_model(input_names, label_name, img_net, combo_net, train_dataloader, batch_size, num_epoches, loss_func, optimizer,
                writer, device, output_name, start_step=0, start_epoch=0, patience=3, print_every=10):
    '''Train passed combo model against passed training dataloader per the passed label and hyperparameters (batch size, number of epochs, 
        and learning rate). Trains per passed loss and optimization functions. Outputs training results to tensorboard via writer object,
        prints loss function result and training accuracy every passed number of steps (print every), and stops training early if no_grad
        improvement in training accuracy is seen after three epochs. Also needs the device (GPU(s) or CPU) and pretrained, fixed image
        model (even if not used). Trains based on passed input names, which should be a list of length 2.'''
    
    step = start_step
    prev_acc = 0
    lack_of_improvement = 0
    ckp_path = "./checkpoint/"
    
    combo_net.train()
    for epoch in range(start_epoch, num_epoches):
        adjust_learning_rate(optimizer, epoch)
        
        for batch_id, data in enumerate(train_dataloader): 
            step += 1

            # Inputs for minibatch and training model
            labels = Variable(data[label_name]).to(device).to(torch.float32)
            
            if 'Image' in input_names:
                images = Variable(data['Image']).to(device)
                input_one = img_net(images)

                if "pH" in input_names:
                    input_two = Variable(data['pH']).to(device).to(torch.float32)
                    
                elif "ClueCell" in input_names:
                    input_two = Variable(data['ClueCell']).to(device).to(torch.float32)
                    
                else:
                    input_two = Variable(data['Whiff']).to(device).to(torch.float32)
            
            elif 'ClueCell' in input_names:
                input_one = Variable(data['ClueCell']).to(device).to(torch.float32)

                if "pH" in input_names:
                    input_two = Variable(data['pH']).to(device).to(torch.float32)
                else:
                    input_two = Variable(data['Whiff']).to(device).to(torch.float32)
            
            else:
                input_one = Variable(data['pH']).to(device).to(torch.float32)
                input_two = Variable(data['Whiff']).to(device).to(torch.float32)
            
            # Feed into combo model and get loss
            outputs = combo_net(input_one, input_two)
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
            save_checkpoint(ckp_path, combo_net, epoch, optimizer, step)
            
        if lack_of_improvement >= patience:  # Early Stop
            break
        
        prev_acc = train_acc
        
    # Finish training
    writer.flush()
    print()
    print('Finished Training')
    path_file_path = './Path files/' + output_name + '.pth'
    torch.save(combo_net.state_dict(), path_file_path)

def test_model(input_names, label_name, img_net, combo_net, test_dataloader, device, output=True, output_name='recentmodel1'):
    '''Run passed combo model against passed test dataloader per the passed label and print test statistics (if output=True).
        Also needs the device (GPU(s) or CPU) and pretrained, fixed image model (even if not used)'''
    
    combo_net.eval()
    
    true_pos = 0
    true_neg = 0
    fal_pos = 0
    fal_neg = 0
    
    with torch.no_grad(): # Freezes model parameters   
        for data in test_dataloader:
        
            # Inputs for minibatch and training model
            labels = Variable(data[label_name]).to(device).to(torch.float32)
            
            if 'Image' in input_names:
                images = Variable(data['Image']).to(device)
                input_one = img_net(images)

                if "pH" in input_names:
                    input_two = Variable(data['pH']).to(device).to(torch.float32)
                    
                elif "ClueCell" in input_names:
                    input_two = Variable(data['ClueCell']).to(device).to(torch.float32)
                    
                else:
                    input_two = Variable(data['Whiff']).to(device).to(torch.float32)
            
            elif 'ClueCell' in input_names:
                input_one = Variable(data['ClueCell']).to(device).to(torch.float32)

                if "pH" in input_names:
                    input_two = Variable(data['pH']).to(device).to(torch.float32)
                else:
                    input_two = Variable(data['Whiff']).to(device).to(torch.float32)
            
            else:
                input_one = Variable(data['pH']).to(device).to(torch.float32)
                input_two = Variable(data['Whiff']).to(device).to(torch.float32)

            outputs = combo_net(input_one, input_two) # Get the test predictions
            predicted = torch.round(outputs.data) # Collect the most likely class per score
                
            # Get categories of true positive, true negative, false positive, false negative
            for idx in range(len(predicted)):
                if labels[idx] == predicted[idx]:
                    if labels[idx] == 0:
                        true_neg += 1
                    else:
                        true_pos += 1
                else:
                    if labels[idx] == 0:
                        fal_pos += 1
                    else:
                        fal_neg += 1

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

def full_process(input_names, label_name, num_epochs, learning_rate, batch_size, img_size, device, load_chckpnt=False, path_file=None, output_name='recentmodel1'):
    '''Runs the training process and then performs the test set evaluation. Passed the label as a str and the input names as a
        list of strs to train and test against, the hyperparameters of number of epochs, the base learning rate, and the batch size, 
        the size of images for image models, and the device (CPU or GPU)'''

    import os

    # Create (fixed) Image Model
    img_net = img_model()
    img_net.to(device)
    if 'Image' in input_names:
        img_net.load_state_dict(torch.load(path_file))

    # Create Combo Model
    combo_net = combo_model()
    combo_net.to(device)

    # Optimizer and Loss Function
    optimizer = optim.Adam(combo_net.parameters(),lr=learning_rate)
    loss_func = nn.BCELoss()

    # Other set up for training
    writer = SummaryWriter()
    train_dataloader, test_dataloader = load_data(batch_size, 224)
    start_step = 0
    used_epochs = 0

    # Load checkpoint if needed
    if load_chckpnt and os.path.isfile("./checkpoint/checkpoint.pt"):
        print("Loaded checkpoint")
        used_epochs, start_step = load_checkpoint("./checkpoint/checkpoint.pt", combo_net, optimizer)
        
    # Train the model
    train_model(input_names, label_name, img_net, combo_net, train_dataloader, batch_size, num_epochs, loss_func, 
                optimizer, writer, device, output_name, start_step, used_epochs, print_every=1)
    writer.close()

    # Test the model
    test_model(input_names, label_name, img_net, combo_net, test_dataloader, device, output_name=output_name)