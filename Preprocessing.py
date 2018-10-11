# -*- coding: utf-8 -*-

"""
@author: Ming JIN
"""
"""
Read images from the dataset; 

it has been revoked by Step1-Training.ipynb;
"""

import os
import numpy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from PIL import Image
import util as util
import cv2
import random
import collections

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def read_data_sets(data_dir,
                   one_hot=True,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=0,  # no validation needed
                   seed=None):

    TRAIN = os.path.join(data_dir, "train", "train.txt") # read the training description
    TEST = os.path.join(data_dir, "test", "test.txt") # read the testing description

    train_images, train_labels = process_images(TRAIN, one_hot = one_hot)
    test_images, test_labels = process_images(TEST, one_hot = one_hot) # testing set

    validation_images = train_images[:validation_size] # validation set
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:] # training set
    train_labels = train_labels[validation_size:]
    
    cut = numpy.empty((128,24,24,3))
    cut_test = numpy.empty((1280,24,24,3))
    
    new_part_cut = numpy.empty((128,24,24,3))
    new_part_cut_test = numpy.empty((1280,24,24,3))
    rest_part_cut = numpy.empty((128,24,24,3))
    rest_part_cut_test = numpy.empty((1280,24,24,3))
    
    train = DataSet(train_images, train_labels, cut, cut_test, new_part_cut, rest_part_cut, new_part_cut_test, rest_part_cut_test, dtype=dtype, reshape=reshape, seed=seed)
    validation = DataSet(validation_images,validation_labels,cut,cut_test, new_part_cut, rest_part_cut, new_part_cut_test, rest_part_cut_test, dtype=dtype,reshape=reshape,seed=seed)
    test = DataSet(test_images, test_labels, cut, cut_test, new_part_cut, rest_part_cut, new_part_cut_test, rest_part_cut_test, dtype=dtype, reshape=reshape, seed=seed)

    return Datasets(train = train, validation = validation, test = test)


def process_images(label_file, one_hot, num_classes=10):
    if util.getFileName(label_file) == 'train.txt':
        images = numpy.empty((50000, 3072)) 
        labels = numpy.empty(50000)
    if util.getFileName(label_file) == 'test.txt':
        images = numpy.empty((10000, 3072))
        labels = numpy.empty(10000)
        
    lines = util.readLines(label_file)
    label_record = util.getLabel(lines)
    file_name_length = len(util.getFileName(label_file))
    image_dir = label_file[:-1 * file_name_length]
    print(len(label_record))
    index = 0
    for name in label_record:
        image = Image.open(image_dir + str(label_record[name]) + '/' + name)
        if index % 100 == 0:
            print("processing %d: " % index + image_dir + str(label_record[name]) + '/' + name)
        img_ndarray = numpy.asarray(image, dtype='float32')
        images[index] = numpy.ndarray.flatten(img_ndarray)
        labels[index] = numpy.int(label_record[name])
        index = index + 1    
    print("done: %d" % index)
    
    num_images = index
    rows = 32
    cols = 32
    
    if one_hot:
      return images.reshape(num_images, rows, cols, 3), dense_to_one_hot(numpy.array(labels, dtype=numpy.uint8), num_classes)
  
    return images.reshape(num_images, rows, cols, 3), numpy.array(labels, dtype=numpy.uint8)


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


"""DataSet Structure"""
class DataSet(object):
  
  def __init__(self,
               images,
               labels,
               cut_images,
               cut_test_images,
               images_new_part_cut,
               images_rest_part_cut,
               images_new_part_cut_test,
               images_rest_part_cut_test,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=False,
               seed=None):
    
    seed1, seed2 = random_seed.get_seed(seed)
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot 
    else:  
      assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      if reshape:
        assert images.shape[3] == 3
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2], images.shape[3])
      
      if dtype == dtypes.float32:
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    
    self._cut_images = cut_images
    self._cut_test_images = cut_test_images
    self._images_new_part_cut = images_new_part_cut
    self._images_rest_part_cut = images_rest_part_cut
    self._images_new_part_cut_test = images_new_part_cut_test
    self._images_rest_part_cut_test = images_rest_part_cut_test   
    
  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  """This method returns the next batch from this data set."""
  def next_batch(self, batch_size, shuffle, flip, whiten, noise, crop, crop_test):
        
    start = self._index_in_epoch
    
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    
    # Go to the next epoch
    if start + batch_size > self._num_examples:

      # Finished epoch
      self._epochs_completed += 1   
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
        
      # Start next epoch      
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
 
      if crop:
         images_crop_1 = images_new_part
         self._images_new_part_cut = self._image_crop(images_crop_1)
         
         images_crop_2 = images_rest_part
         self._images_rest_part_cut = self._image_crop(images_crop_2)
         
      if crop_test:
         images_crop_1 = images_new_part
         self._images_new_part_cut_test = self._image_test_crop(images_crop_1)
         
         images_crop_2 = images_rest_part
         self._images_rest_part_cut_test = self._image_test_crop(images_crop_2) 
         
      if flip:
          images_flip_1 = self._images_new_part_cut
          self._images_new_part_cut = self._image_flip(images_flip_1)
          images_flip_2 = self._images_rest_part_cut
          self._images_rest_part_cut = self._image_flip(images_flip_2)
      
      if whiten:
          if crop:
             images_whiten_1 = self._images_new_part_cut
             self._images_new_part_cut = self._image_whitening(images_whiten_1)
             images_whiten_2 = self._images_rest_part_cut
             self._images_rest_part_cut = self._image_whitening(images_whiten_2)
             
          if crop_test:
             images_whiten_1 = self._images_new_part_cut_test
             self._images_new_part_cut_test = self._image_whitening(images_whiten_1)
             images_whiten_2 = self._images_rest_part_cut_test
             self._images_rest_part_cut_test = self._image_whitening(images_whiten_2)     
      
      if noise:
          images_noise_1 = self._images_new_part_cut
          self._images_new_part_cut = self._image_noise(images_noise_1)
          images_noise_2 = self._images_rest_part_cut
          self._images_rest_part_cut = self._image_noise(images_noise_2)
          
      if crop:
         return numpy.concatenate((self._images_rest_part_cut, self._images_new_part_cut), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
      elif crop_test:   
         return numpy.concatenate((self._images_rest_part_cut_test, self._images_new_part_cut_test), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      
      if crop:
         images_crop = self._images[start:end]
         self._cut_images = self._image_crop(images_crop)

      if crop_test:
         images_crop = self._images[start:end]
         self._cut_test_images = self._image_test_crop(images_crop)
          
      if flip:
          images_flip = self._cut_images
          self._cut_images = self._image_flip(images_flip)
      
      if whiten:
          if crop:
             images_whiten = self._cut_images
             self._cut_images = self._image_whitening(images_whiten)
          if crop_test:
             images_whiten = self._cut_test_images
             self._cut_test_images = self._image_whitening(images_whiten)     
      
      if noise:
          images_noise = self._cut_images
          self._cut_images = self._image_noise(images_noise)
      
      if crop:
         return self._cut_images, self._labels[start:end]
      elif crop_test:
         return self._cut_test_images, self._labels[start:end]     

######################## Data Enhancements Functions ############################

  def _image_crop(self, images, crop_shape=(24,24,3)):
        # image cutting 1
        new_images = numpy.empty((images.shape[0],24,24,3))
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            left = numpy.random.randint(old_image.shape[0] - crop_shape[0] + 1)
            top = numpy.random.randint(old_image.shape[1] - crop_shape[1] + 1)
            new_image = old_image[left:left+crop_shape[0],top:top+crop_shape[1], :]
            new_images[i,:,:,:] = new_image
            
        return new_images

  def _image_test_crop(self, images, crop_shape=(24,24,3)):
        # image cutting 2
        new_images = numpy.empty((images.shape[0],24,24,3))
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            left = int((old_image.shape[0] - crop_shape[0])/2)
            top = int((old_image.shape[1] - crop_shape[1])/2)
            new_image = old_image[left:left+crop_shape[0],top:top+crop_shape[1], :]
            new_images[i,:,:,:] = new_image
            
        return new_images

  def _image_whitening(self, images):
        # function for image whitening
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            new_image = (old_image - numpy.mean(old_image)) / numpy.std(old_image)
            images[i,:,:,:] = new_image
        
        return images
  
  def _image_flip(self, images):
        # function for image flipping
        for i in range(images.shape[0]):
            
            old_image = images[i,:,:,:]
            
            if numpy.random.random() < 0.5:
                new_image = cv2.flip(old_image, 1)
            else:
                new_image = old_image
                
            images[i,:,:,:] = new_image
        
        return images

  def _image_noise(self, images, mean=0, std=0.01):
        # functions for introducing noise
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            new_image = old_image
            for i in range(old_image.shape[0]):
                for j in range(old_image.shape[1]):
                    for k in range(old_image.shape[2]):
                        new_image[i, j, k] += random.gauss(mean, std)
            images[i,:,:,:] = new_image
        
        return images