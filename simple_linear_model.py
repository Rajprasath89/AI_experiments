###
# Sricharan Chiruvolu -- Learning MNIST, Simple Linear models on Tensorflow.
###
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# from sklearn.metrics import confusion_matrix

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("MNIST_data/", one_hot=True)


# MNIST dataset
print ("Size of: ")
print ("- Training Set:\t\t{}".format(len(data.train.labels)))
print ("- Test Set:\t\t{}".format(len(data.test.labels)))
print ("- Validation Set:\t{}".format(len(data.validation.labels)))
print


# The dataset has been loaded as one-hot encoding.
print ("One-hot encoded labels for the first 5 images")
print (data.test.labels[0:5, :])
print 

print ("One-hot encoded to numbers conversion")
data.test.cls = np.array([label.argmax() for label in data.test.labels])
print (data.test.cls[0:5])
print 

# data dimensions
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10 # one for each digit


# Function to plot images 
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

# Ploting a few images from test-set
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]

print ("Plotting a few images from test-set")
plot_images(images=images, cls_true=cls_true)
plt.show()



