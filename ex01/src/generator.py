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
        if self.shuffle is True:
            self.files, self.file_names = sklearn.utils.shuffle(self.files, self.file_names)
        self.mirroring = mirroring
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.next_files = 0

    def next(self):
        # This function creates a batch of images and corresponding labels and returns it.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases

        while self.next_files < len(self.files):
            batch = None
            labels = None
            batch_files = 0
            while batch_files < self.batch_size:
                idx = (self.next_files + batch_files) % (len(self.files))
                img = self.files[idx]
                img = np.resize(img, (self.image_size[0], self.image_size[1], self.image_size[2]))
                if self.mirroring and random.random() < 0.5:
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
                if labels is None:
                    labels = np.array([self.labels[str(self.file_names[idx])]])
                else:
                    labels = np.append(labels, [self.labels[str(self.file_names[idx])]])
                batch_files += 1

            self.next_files += self.batch_size
            return batch, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        rnd = random.random()
        rnd2 = random.random()

        if rnd < 0.25:
            img = np.rot90(img)
        elif rnd < 0.5:
            img = np.rot90(np.rot90(img))
        elif rnd < 0.75:
            img = np.rot90(np.rot90(np.rot90(img)))

        if rnd2 < .5:
            img = np.flip(img, axis=0)

        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.

        # fig = plt.figure(1)
        #
        # one = fig.add_subplot(4,3,1)
        # one.title.set_text(str(self.class_name(labels[0])))
        # plt.imshow(plots[0])
        #
        # two = fig.add_subplot(4,3,2)
        # two.title.set_text(str(self.class_name(labels[1])))
        # plt.imshow(plots[1])
        #
        # three = fig.add_subplot(4,3,3)
        # three.title.set_text(str(self.class_name(labels[2])))
        # plt.imshow(plots[2])
        #
        # four = fig.add_subplot(4,3,4)
        # four.title.set_text(str(self.class_name(labels[3])))
        # plt.imshow(plots[3])
        #
        # five = fig.add_subplot(4,3,5)
        # five.title.set_text(str(self.class_name(labels[4])))
        # plt.imshow(plots[4])
        #
        # six = fig.add_subplot(4,3,6)
        # six.title.set_text(str(self.class_name(labels[5])))
        # plt.imshow(plots[5])
        #
        # seven = fig.add_subplot(4,3,7)
        # seven.title.set_text(str(self.class_name(labels[6])))
        # plt.imshow(plots[6])
        #
        # eight = fig.add_subplot(4,3,8)
        # eight.title.set_text(str(self.class_name(labels[7])))
        # plt.imshow(plots[7])
        #
        # nine = fig.add_subplot(4,3,9)
        # nine.title.set_text(str(self.class_name(labels[8])))
        # plt.imshow(plots[8])
        #
        # ten = fig.add_subplot(4,3,10)
        # ten.title.set_text(str(self.class_name(labels[9])))
        # plt.imshow(plots[9])
        #
        # eleven = fig.add_subplot(4,3,11)
        # eleven.title.set_text(str(self.class_name(labels[10])))
        # plt.imshow(plots[10])
        #
        # twelve = fig.add_subplot(4,3,12)
        # twelve.title.set_text(str(self.class_name(labels[11])))
        # plt.imshow(plots[11])

        plots = self.next()[0]
        labels = self.next()[1]

        fig2 = plt.figure(2)
        for i in range(12):
            zw = fig2.add_subplot(4, 3, i+1)
            zw.title.set_text(str(self.class_name(labels[i])))
            plt.axis('off')
            plt.imshow(plots[i])

        plt.show()

