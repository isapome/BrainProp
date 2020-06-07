# -*- coding: utf-8 -*-

import sys,os
from PIL import Image
import numpy as np
from tqdm import tqdm

class HiddenPrints:
  def __enter__(self):
    self._original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
  def __exit__(self, exc_type, exc_val, exc_tb):
    sys.stdout.close()
    sys.stdout = self._original_stdout
    
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def get_annotations_map(dataset, path):
  if dataset == "validation":
    string = "val"
  elif dataset == "test":
    string = "test"
  else:
    raise Exception("Unknown subdataset for Tiny ImageNet.")
  val_annotations_path = os.path.join(path,'tiny-imagenet-200',string,'val_annotations.txt')
  with open(val_annotations_path, 'r') as file:
    content = file.read()
  val_annotations = {}
  for line in content.splitlines():
    pieces = line.strip().split()
    val_annotations[pieces[0]] = pieces[1]
  return val_annotations
  

def prepare_tinyimagenet(num_classes=None, path=None):
  print('Fetching Tiny ImageNet..')
  if path:
    path0 = path
  else:
    path0 = ''

  if not num_classes:
    num_classes = 200
  x_train=np.zeros([num_classes*500,3,64,64],dtype='uint8')
  y_train=np.zeros([num_classes*500], dtype='uint8')
  train_path=os.path.join(path0,'tiny-imagenet-200/train')
  print('loading training images belonging to {} classes...'.format(num_classes));
  i=0
  label=0
  annotations={}
  filelist = [x for x in os.listdir(train_path) if not x.startswith('.')]
  for class_folder in tqdm(iterable=listdir_nohidden(train_path), total=len(filelist)):
#  for class_folder in listdir_nohidden(train_path):
      images_folder = os.path.join(os.path.join(train_path,class_folder),'images')
      annotations[class_folder]=label
      for image in listdir_nohidden(images_folder):
          X=np.array(Image.open(os.path.join(images_folder,image)))
          if len(np.shape(X))==2:
              x_train[i]=np.array([X,X,X])
          else:
              x_train[i]=np.transpose(X,(2,0,1))
          y_train[i]=label
          i+=1
      label+=1

  print('finished loading {} training images'.format(len(filelist)))
  
  val_annotations_map = get_annotations_map("validation", path0)
  x_val = np.zeros([num_classes*50,3,64,64],dtype='uint8')
  y_val = np.zeros([num_classes*50], dtype='uint8')
  print('loading validation images...')
  i = 0
  validation_path=os.path.join(path0,'tiny-imagenet-200/val/images')
#  for image in listdir_nohidden(validation_path):
  filelist = [x for x in os.listdir(validation_path) if not x.startswith('.')]
  for image in tqdm(iterable=listdir_nohidden(validation_path), total=len(filelist)):
      if val_annotations_map[image] in annotations.keys():
          image_path = os.path.join(validation_path, image)
          X=np.array(Image.open(image_path))
          if len(np.shape(X))==2:
              x_val[i]=np.array([X,X,X])
          else:
              x_val[i]=np.transpose(X,(2,0,1))
          y_val[i]=annotations[val_annotations_map[image]]
          i+=1
      else:
          pass
  print('finished loading validation images')

  x_train=np.transpose(x_train,(0,2,3,1))
  x_val=np.transpose(x_val,(0,2,3,1))
  return x_train, y_train, x_val, y_val


def main():
    print('Preparing Tiny ImageNet dataset in the current directory')
    foldername = "tiny-imagenet-200"
    filename = "tiny-imagenet-200.zip"
    if os.path.isdir(foldername):
        print("The folder \'{}\' already exists.".format(foldername))
    else:
        if os.path.isfile(filename):
            from zipfile import ZipFile
            with ZipFile(filename, 'r') as zip_file:
                for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
                    zip_file.extract(member=file)
                print("Tiny ImageNet successfully extracted to the current directory.")
        else:
            print("File \'{}\' not found. Please download it to the current directory.".format(filename))
    
#    train_images, train_labels, test_images, test_labels = prepare_tinyimagenet()
    
if __name__ == "__main__":
    main()

