# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:32:11 2024

@author: Brendan
"""

import cv2
import os
from glob import glob
import numpy as np
from random import shuffle
from datetime import datetime
import gc
from sklearn.model_selection import train_test_split
import joblib


class dataset_data():
    
    def __init__(self, dir):
        self.src_dir = dir
        self.bus = []           # List of Bus images
        self.car = []           # List of Car images
        self.motorcycles = []   # List of Motorcycle images
        self.person = []        # List of Pedestrian images
        self.Dataset = []
        
    def get_dataset_2D(self):
        list_dirs = os.listdir(self.src_dir)    # = [Bus, Car, Motorcycles, Person]
        for class_dir in list_dirs:                                             # Parse the list of the 4 classes
            images_list = (glob(f'{self.src_dir}/{class_dir}/*.jpg') +
                           glob(f'{self.src_dir}/{class_dir}/*.jpeg') +
                           glob(f'{self.src_dir}/{class_dir}/*.png'))   # List of Images in each specific class
            print(len(images_list))
            for image in images_list:   # Analyze each individual image
                # print(image)
                im = cv2.imread(image, cv2.COLOR_BGR2RGB)
                if im is None:
                    print(f"Dataset Error: Unable to read image {image}")
                    continue
                if len(im.shape) != 3:
                    print(f"Dataset Error: Image {image} has unexpected dimensions")
                    continue
                
                im2 = im[:-180, :, :]
                rows, columns, channel = im.shape
                if rows == 0 or columns == 0:
                    print (f"Dataset Error: Image {image} has invalid dimensions after preprocessing")
                    continue
                
                im2 = im[int(rows*0.17):rows, 0:columns]
                im2 = cv2.resize(im2, (244, 244), interpolation = cv2.INTER_AREA)
                
                if class_dir == "Bus":
                    self.bus.append(im2)
                elif class_dir == "Car":
                    self.car.append(im2)
                elif class_dir == "Motorcycles":
                    self.motorcycles.append(im2)
                elif class_dir == "Person":
                    self.person.append(im2)
                else:
                    break
                
        print(len(self.bus), len(self.car), len(self.motorcycles), len(self.person))
        shuffle(self.bus)
        shuffle(self.car)
        shuffle(self.motorcycles)
        shuffle(self.person)
        
    def augment(self, save_dir):
        dir_list = os.listdir(self.src_dir)
        '''
        image_bus_class = 3213/2
        image_car_class = 3216/2
        image_motorcycles_class = 3280/2
        image_person_class = 3263/2
        '''
        
        image_bus_class = 50
        image_car_class = 50
        image_motorcycles_class = 50
        image_person_class = 50
        
        print('Augment: Augmenting Bus')
        n = 0
        if not os.path.exists(save_dir + "/" + dir_list[0]):
            os.makedirs(save_dir + "/" + dir_list[0])
            
        for image in self.bus:
            self.process(image, 0, save_dir + "/" + dir_list[0], n)
            n += 1
            if n > image_bus_class:
                break
        
        del self.bus
        gc.collect()
        
        
        print('Augment: Augmenting Car')
        n = 0
        if not os.path.exists(save_dir + "/" + dir_list[1]):
            os.makedirs(save_dir + "/" + dir_list[1])
            
        for image in self.car:
            self.process(image, 1, save_dir + "/" + dir_list[1], n)
            n += 1
            if n > image_car_class:
                break
        
        del self.car
        gc.collect()
        
        
        print('Augment: Augmenting Motorcycles')
        n = 0
        if not os.path.exists(save_dir + "/" + dir_list[2]):
            os.makedirs(save_dir + "/" + dir_list[2])
            
        for image in self.motorcycles:
            self.process(image, 2, save_dir + "/" + dir_list[2], n)
            n += 1
            if n > image_motorcycles_class:
                break
        
        del self.motorcycles
        gc.collect()
        
        
        print('Augment: Augmenting Person')
        n = 0
        if not os.path.exists(save_dir + "/" + dir_list[3]):
            os.makedirs(save_dir + "/" + dir_list[3])
            
        for image in self.person:
            self.process(image, 3, save_dir + "/" + dir_list[3], n)
            n += 1
            if n > image_person_class:
                break
        
        del self.person
        gc.collect()
        
        
    def process(self, image, label, output, n):
        image_name = f'image_{label}_{n}'
        New_image_Name = os.path.join(output, image_name)
        cv2.imwrite(New_image_Name + ".png", image)
        self.Dataset.append([image,label])
        fliped_image = cv2.flip(image, 1)
        cv2.imwrite(New_image_Name + "_flipped.png", fliped_image)
        self.Dataset.append([fliped_image,label])
        for angle in range(180, 360, 180):
            rows, columns, channel = image.shape
            center = ((columns-1)/2, (rows-1)/2) #center of rotation
            Modifcation = cv2.getRotationMatrix2D(center, angle, 1.0)
            New_Image = cv2.warpAffine(image, Modifcation, (columns, rows))
            cv2.imwrite(New_image_Name + f"_{angle}.png", New_Image)
            self.Dataset.append([New_Image,label])
            # C_New_fliped_image = cv2.flip(New_Image, 1)
            # cv2.imwrite(New_image_Name + f"_flipped_{angle}.png", C_New_fliped_image)
            # self.Dataset.append([C_New_fliped_image,label])
            
            
    def packing(self, compact_dir):
        if not os.path.exists(compact_dir):
            os.makedirs(compact_dir)
        
        print('Packing: Shuffling Dataset')
        shuffle(self.Dataset)
        images = np.empty([len(self.Dataset), 244, 244, 3])
        labels = np.empty([len(self.Dataset), 1], dtype = int)
        
        print('Packing: Splitting Images/Labels')
        for Count, data in enumerate(self.Dataset):
            images[Count] = np.array(data[0])
            labels[Count] = np.array(data[1])
        del self.Dataset
        gc.collect()
        
        print('Packing: Splitting Dataset')
        Image_train, Image_test, Label_train, Label_test = train_test_split(images, labels, test_size = 0.2, train_size = 0.8, random_state = 1, shuffle = True)
        
        print('Packing: Image_train.sav')
        joblib.dump(Image_train, compact_dir + "/Image_train.sav")
        del Image_train
        gc.collect()
        
        print('Packing: Image_test.sav')
        joblib.dump(Image_test, compact_dir + "/Image_test.sav")
        del Image_test
        gc.collect()
        
        print('Packing: Label_train.sav')
        joblib.dump(Label_train, compact_dir + "/Label_train.sav")
        del Label_train
        gc.collect()
        
        print('Packing: Label_test.sav')
        joblib.dump(Label_test, compact_dir + "/Label_test.sav")
        del Label_test
        gc.collect()
        

# E:\Brendan_2023_24\Dataset\Processed

def main():
    startTime = datetime.now()
    input_data_dir = f"E:/Brendan_2023_24/Dataset/Preprocessed"
    output_data_dir = f"E:/Brendan_2023_24/Dataset/Processed"
    compacted_data_dir = f"E:/Brendan_2023_24/Dataset/Packed"
    dataset = dataset_data(input_data_dir)
    dataset.get_dataset_2D()
    dataset.augment(output_data_dir)
    dataset.packing(compacted_data_dir)
    print(datetime.now() - startTime)


if __name__ == "__main__":
    main()