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
print


# Creating Tensors.


# Placeholder variables used to change the input to the graph.
x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

# Optimize the variables! -- Make model variables -- Model variables that are going to be 
# optimized so as to make the model perform better.
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))


# The model which is essentially just a mathematical function that calculates some 
# output given the input in the placeholder variables and the model variables.


#  [num_images, img_size_flat] x [img_size_flat, num_classes] = [num_images, num_classes] 
#  The logit is here! -- element of the i'th row and j'th column is an estimate of how 
# likely the i'th input image is to be of the j'th class.
logits = tf.matmul(x, weights) + biases

# Normalize them using a softmax function.
y_pred = tf.nn.softmax(logits)

# Find the largest element in each row, to be the predicted class.
y_pred_cls = tf.argmax(y_pred, dimension=1)


# Some performace measures to understand how well we trained the model.

#  Minimun cross_entorpy ==> Match.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_true)

cost = tf.reduce_mean(cross_entropy)


# Target is to decrease the cost. Writing optimizers.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

# Another performance measure.
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Tensorflow Run

# Creating tf session and init variables.
session = tf.Session()
session.run(tf.global_variables_initializer())


# Perform iterations.
batch_size = 100
def optimize(num_iterations):
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)


# Dict with test data -- Input to tf graph
feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: data.test.cls}


def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))


# Print mis classified images.
def plot_example_errors():
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct, cls_pred = session.run([correct_prediction, y_pred_cls],
                                    feed_dict=feed_dict_test)

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


# function to plot weights.
def plot_weights():
    # Get the values for the weights from the TensorFlow variable.
    w = session.run(weights)
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])










