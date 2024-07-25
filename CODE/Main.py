import torch
import argparse
from Data_Split import *


def str2bool(v): # From HW2 utils. Makes typing False or True return the associated boolean object
    # codes from : stackover

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Parser Object
parser = argparse.ArgumentParser(description='Parameters to Train a Specific Type of Model and To Build Dataset (Or do Neither)')

# Dataset Building Parameters
parser.add_argument('--build_dataset', default=False, type=str2bool, help='Bool to build dataset or not')
parser.add_argument('--split_by_pt', default=False, type=str2bool, help='Bool to control if dataset is built randomly or by specific patients')
parser.add_argument('--test_pts', default=[], action='append', help='List of patients integers to use as testing set')
parser.add_argument('--preset_test_set', default=None, type=int, help='List of patients integers to use as testing set')
parser.add_argument('--exclude_pts', default=[6,15,21], action='append', help='List of patient integers to exclude from training and testing set')

# Model Training Parameters
parser.add_argument('--input_names', default=[], action='append', help='List of strs to use to retreive inputs for model from CSVs')
parser.add_argument('--label_name', default='Diagnosis', type=str, help='String to retreive for model from CSVs as label for training and testing')
parser.add_argument('--load_chckpnt', default=False, type=str2bool, help='Bool to load dataset or not during training')
parser.add_argument('--Train_a_Model', default=False, type=str2bool, help='Bool to train a model or not')
parser.add_argument('--Test_a_Model', default=False, type=str2bool, help='Bool to test a model or not')
parser.add_argument('--image_size', default=224, type=int, help='size to resize images to for model training and testing')
parser.add_argument('--Model_Type', default='Image', type=str, help='String of which model type to use')
parser.add_argument('--path_file', default=None, type=str, help='String of path file to use as fixed image model for mixed input model training')
parser.add_argument('--output_name', default='recentmodel', type=str, help='String to name path file of trained model')

# Hyperparameters
parser.add_argument('--num_epoches', default=10, type=int, help='integer number of epoches to train for')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate for model training')
parser.add_argument('--batch_size', default=32, type=int, help='integer of number of items to use in a batch')

# Pull arguments into namespace
args = parser.parse_args()

# Device info
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    torch.cuda.manual_seed(72)
    print("using cuda")

# Assign samples to training and testing folders per specifications
presets = [[1,2,10,14,19,24],[3,8,13,27,29,30],[7,12,13,14,18,30],[2,8,9,11,16,24],[5,7,9,12,22,27],
            [7,13,16,17,25,27],[3,4,20,22,23,28],[1,2,4,16,20,23],[1,10,11,17,26,28],[4,12,14,19,26,29],
            [3,5,11,18,24,26],[4,8,11,16,22,26],[2,3,5,11,20,23],[10,12,17,19,22,27],[9,12,13,24,26,27],
            [3,4,12,25,26,28],[4,7,16,18,23,28],[10,11,20,23,24,25],[1,7,19,27,28,30],[7,8,16,19,24,25]]

if args.build_dataset:
    print("Building dataset...")
        
    # Separate specified paitents into test set and leave out other specified patients from training set
    if args.split_by_pt:
        if args.preset_test_set is None:
            Pt_Data_Split(args.test_pts, args.exclude_pts)
        else:
            test_sets = presets[args.preset_test_set]
            Pt_Data_Split(test_sets, args.exclude_pts)
  
    # Initiate random 20% split of data with 50-50 postive-negative split per specified binary label
    # and excluding specified paitients from both datasets
    else:
        length = 0
        for row in open(".\Data\All_Cropped\Labels.csv"):
            length += 1
        test_set_size = int(length * 0.2)
        
        Rand_Data_Split(test_set_size, args.label_name, args.exclude_pts)
            
# Skip assigning samples to training and testing folders (if already done)
else:
    print("Maintaining current data orientation")
    
# Run Training and testing process for given model
if args.Train_a_Model or args.Test_a_Model:
    if args.Model_Type == 'Image':
        from Train_Image_Model import *
        full_process(args.label_name, args.num_epoches, args.learning_rate, args.batch_size, 
                        args.image_size, device, args.load_chckpnt, args.output_name)
    
    elif args.Model_Type == "CDD2MD":
        from Train_CCDtoMD import *
        pth_fle = 'C:/Users/Daniel/Desktop/BV/Path files/' + args.path_file + '.pth'
        full_process(args.label_name, args.num_epoches, args.learning_rate, args.batch_size, 
                        args.image_size, device, args.load_chckpnt, pth_fle, args.Train_a_Model, args.Test_a_Model, args.output_name)
    
    elif args.Model_Type == "NonImage":
        from Train_NonImage import *
        full_process(args.input_names, args.label_name, args.num_epoches, args.learning_rate, args.batch_size, args.image_size, device, args.load_chckpnt, args.output_name)
    
    elif args.Model_Type == '2Way':
        from Train_2Way_Model import *
        pth_fle = 'C:/Users/Daniel/Desktop/BV/Path files/' + args.path_file + '.pth'
        full_process(args.input_names, args.label_name, args.num_epoches, args.learning_rate, 
                        args.batch_size, args.image_size, device, args.load_chckpnt, pth_fle, args.output_name)
                        
    elif args.Model_Type == "3Way":
        from Train_3Way_Model import *
        pth_fle = 'C:/Users/Daniel/Desktop/BV/Path files/' + args.path_file + '.pth'
        full_process(args.label_name, args.num_epoches, args.learning_rate, args.batch_size, 
                        args.image_size, device, args.load_chckpnt, pth_fle, args.Train_a_Model, args.Test_a_Model, args.output_name)
