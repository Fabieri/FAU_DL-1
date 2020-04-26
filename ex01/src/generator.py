import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.files = np.array([np.load(file_path + file) for file in os.listdir(file_path)])
        self.file_names = np.array([file.split('.')[0] for file in os.listdir(file_path)])
        with open(label_path) as js_file:
            self.labels = json.load(js_file)
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.shuffle = shuffle
        self.mirroring = mirroring
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

    def next(self):
        # This function creates a batch of images and corresponding labels and returns it.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method

        if self.shuffle is True:
            self.files, self.file_names = sklearn.utils.shuffle(self.files, self.file_names)
            # shuffler = np.random.permutation(len(self.file_names))
            # self.files = self.files[shuffler]
            # self.file_names = self.file_names[shuffler]

        num_files = 0
        while num_files < len(self.files):
        # for idx, file in enumerate(self.files):
            batch = None
            labels = np.array([])
            batch_files = 0
            while batch_files < self.batch_size:
                img = self.files[(num_files + batch_files) % (len(self.files))]
                img = np.resize(img, (self.image_size[0], self.image_size[1], self.image_size[2]))
                if self.mirroring and random.random() < 0.3:
                    img = np.flip(img, axis=0)
                if self.rotation:
                    rnd = random.random()
                    if rnd < 0.33:
                        img = np.rot90(img)
                    elif rnd < 0.66:
                        img = np.rot90(np.rot90(img))
                    else:
                        img = np.rot90(np.rot90(np.rot90(img)))
                if batch is None:
                    batch = np.array([img])
                else:
                    batch = np.append(batch, [img], axis=0)
                labels = np.append(labels, self.labels[str(self.file_names[(num_files + batch_files) % (len(self.files))])])
                print(labels)
                batch_files += 1

            num_files += batch_files
            return batch, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        pass

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        pass

