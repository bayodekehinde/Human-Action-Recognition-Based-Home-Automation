#libraries needed for running load_zip_file
import zipfile

#libraries needed View random images
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

#check the model prediction using confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


#function to load zipped files to a drive, 

def load_zip_file(filename: str):

  """This functions reads a zipped file and extract all the files to the present working directory
  Params: 
  filename  (str): the path of the zipped file

  Returns:
  A file/folder in the present working directory

  N.B: You must import the zipfile module before calling this function

  import zipfile
  """

  zip_ref=zipfile.ZipFile(filename, "r")
  
  return zip_ref.extractall()   #extract the files in the directory


  


def view_random_image(*,target_dir : str, class_names : list) :

  """The function helps to view random images from different classes
  
  Params:
    target_dir (str): The directory of the images to be viewed

    class_names (List) : The list of the name of the classes in the dataset
  
  Returns:
  
  An image plot
  
  """
  # Setup target directory 
  target_class=random.sample(list(class_names), 1) #take one sample from the list of classes
  
  target_folder = target_dir+target_class[0]       #get the class name as a string type target_class[0]

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class[0])
  plt.axis("off");

  return (f"Image shape: {img.shape}") # show the shape of the image



#plot the loss and accuracy curves

import matplotlib.pyplot as plt

# Plot the validation and training data separately
def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();





def plot_confusion_matrix(trained_model, test_data, class_names : list, architecture_name: str):

  """
  This function plots confusion matrix

  Params:

  traine_model : The model that has been trained or a model loaded after training
  test_data: A batchTensor of the test data

  class_names (list) : The names of distict class in the data

  #N.B Ensure that when loading the test data , the shuffle should be False
  
  """

  model=trained_model   #trained model
  test_data =test_data  # test_data 
  test_label = np.concatenate([y for x, y in test_data], axis=0)  # Get the labels from the test_data

  # Make predictions on the test dataset
  predicted_labels=model.predict(test_data)

  predicted_labels = np.argmax(predicted_labels, axis=1)
  test_label=np.argmax(test_label, axis=1)

  # Calculate confusion matrix
  confusion_mat = confusion_matrix(test_label, predicted_labels)

  # Visualize the confusion matrix
  class_names  

  plt.figure(figsize=(8, 6))
  sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
  plt.title(f'Confusion Matrix {architecture_name}')
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.show()


#test model on some images using 

import cv2
import tensorflow as tf
import numpy as np

def test_model(image_path : str, trained_model, class_names : list):
  """
    This function helps to test trained models on images

    Params:

    image_path (str) : The path of the image to make predictions on

    trained_model :  A model that has been trained or a saved model that has been loaded

    class_names (list) : The class labels of the data, the class_names should be passed exactly the same way it
            was passed when training was done
            class_names=['clapping', 'lying', 'sitting', 'standing', 'waving']


    Returns:
      A predicted label
    
    """
  # Load and preprocess the input image
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
  image = cv2.resize(image, (224, 224))  # Resize to match the input size expected by the model
 
  image = np.expand_dims(image, axis=0)  # Add batch dimension

  # Perform prediction
  predictions = trained_model.predict(image)

  # Get the predicted class
  predicted_class = class_names[np.argmax(predictions)]

  # Print the predicted class or use it for further processing
  print("Predicted class:", predicted_class)
