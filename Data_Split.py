import random
import pandas as pd
import os
import shutil
import sys
import scipy
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def Rand_Data_Split(input_test_size, label_name='Diagnosis', exclude_list=[]):
    '''Assign training and testing data disregarding which patient they come from. Input test set size, what the variable of
        interest is (to have a 50-50 postive-negative split. Not intended for labels beyond Diagnosis and ClueCell), and what
        patients if any to exclude from the training and testing data'''

    data_link = pd.read_csv("./Data/All_Images/Labels.csv")
    test_size = input_test_size
    extra = (test_size % 2 == 1)

    # Get column names 
    cols = []
    for col in data_link.columns:
        cols.append(col)

    # Calculate 50-50 postive-negative split
    num_healthy, num_sick = test_size // 2, test_size // 2
    if extra:
        give_to = random.randint(0,1)
        if give_to == 0:
            num_healthy += 1
        else:
            num_sick += 1
           
    # Remove excluded patients from dataframe with all patients
    exclude = ""
    for pt in exclude_list:
        pt_str = "pt" + str(pt) + " "
        if len(exclude) != 0:
            exclude += "|"
        exclude += pt_str
    if exclude != "":   
        data_link = data_link[~data_link['Image_Path'].str.contains(exclude)]

    # Create dataframe for test data with 50-50 postive-negative split
    healthy_link = data_link[data_link[label_name] == 0].sample(n=num_healthy).reset_index(drop=True)
    sick_link = data_link[data_link[label_name] == 1].sample(n=num_sick).reset_index(drop=True)
    test_data = healthy_link.merge(sick_link, how='outer')

    # Create dataframe for training data
    training_data = data_link.merge(test_data, how='left', indicator=True)
    training_data = training_data[training_data['_merge'] == 'left_only']
    #training_data = training_data[cols]

    # Populate folders for load_data
    data_fill(training_data, test_data)
    
def Pt_Data_Split(test_list, exclude_list=[]):
    '''Assign training and testing data by which patient they come from. Input which patients to use for test set and what
        patients if any to exclude from the training data'''
    
    data_link = pd.read_csv("./Data/All_Images/Labels.csv")

    # Get column names
    cols = []
    for col in data_link.columns:
        cols.append(col)

    # Get substrings for split generation
    include = ""
    for pt in test_list:
        pt_str = "pt" + str(pt) + " "
        if len(include) != 0:
            include += "|"
        include += pt_str
    
    exclude = ""
    for pt in exclude_list:
        pt_str = "pt" + str(pt) + " "
        if len(exclude) != 0:
            exclude += "|"
        exclude += pt_str
    
    # Generate datasets per substring
    test_data = data_link[data_link['Image_Path'].str.contains(include)]
    if exclude != "":
        include = include + "|" + exclude
    training_data = data_link[~data_link['Image_Path'].str.contains(include)]

    # Populate folders for load_data
    data_fill(training_data, test_data)
    
def clear_folder(PATH):
    '''Removes all files from a folder passed to the function'''
    folder = PATH
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
                
def copy_images(df, PATH):
    '''Moves specified images in passed dataframe to specified folder'''
    imgs = list(df['Image_Path'])
    for img in imgs:
        shutil.copy(img, PATH) 
              
def data_fill(training_data, test_data):
    '''Utillizes the training and testing dataframes to create CSV files to inform Load_Data and moves images to associated folders'''
    # Prepare training and testing folders
    clear_folder("./Data/testing")
    clear_folder("./Data/training")

    # Create Label CSVs
    test_data.to_csv("./Data/testing/Labels.csv", index=False)
    training_data.to_csv("./Data/training/Labels.csv", index=False)

    # Copy images to training and testing folders
    print("Importing testing images...")
    copy_images(test_data, "./Data/testing")
    print("Importing training images...")
    copy_images(training_data, "./Data/training")