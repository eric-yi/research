#!/usr/bin/env python
# -*- coding:utf-8 -*-

from header import *
import gzip
import cv2


train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

data_sources = {
    'training_images': os.path.join('dataset/mnist_lecun', 'train-images-idx3-ubyte.gz'),  # 60,000 training images.
    'test_images':  os.path.join('dataset/mnist_lecun','t10k-images-idx3-ubyte.gz'),  # 10,000 test images.
    'training_labels':  os.path.join('dataset/mnist_lecun','train-labels-idx1-ubyte.gz'),  # 60,000 training labels.
    'test_labels':  os.path.join('dataset/mnist_lecun','t10k-labels-idx1-ubyte.gz'),  # 10,000 test labels.
}

def read_gzip(path):
    with gzip.open(path, 'rb') as fd:
        return fd.read()

def load_images(path):
    return np.frombuffer(read_gzip(path), np.uint8, offset=16).reshape(-1, img_size)

def load_labels(path):
    return np.frombuffer(read_gzip(path), np.uint8, offset=8)
    
mnist_dataset = {
    'training_images': load_images(data_sources['training_images']),
    'test_images': load_images(data_sources['test_images']),
    'training_labels': load_labels(data_sources['training_labels']),
    'test_labels': load_labels(data_sources['test_labels']),
}

logger.debug(mnist_dataset['training_images'].shape)

def show_image(img, title='img'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    
# show_image(mnist_dataset['training_images'][0].reshape(img_dim[1], img_dim[2]))


seed = 147197952744
rng = np.random.default_rng(seed)    

def show(dataset):
    num_examples = 5
    images_list = []
    for sample in rng.choice(dataset, size=num_examples, replace=False):
        images_list.append(sample.reshape(img_dim[1], img_dim[2]))
    show_image(np.hstack(images_list))

def one_hot_encoding(labels, dimension=10):
    one_hot_labels = labels[..., None] == np.arange(dimension)[None]
    return one_hot_labels.astype(np.float64)

def relu(x):
    return (x >= 0) * x

def relu2deriv(output):
    return output >= 0


    
def prehandle():
    x_train, y_train, x_test, y_test = (
        mnist_dataset['training_images'],
        mnist_dataset['training_labels'],
        mnist_dataset['test_images'],
        mnist_dataset['test_labels'],
    )
    training_sample, test_sample = 1000, 1000
    training_images = x_train[0:training_sample] / 255
    test_images = x_test[0:test_sample] / 255
    training_labels = one_hot_encoding(y_train[:training_sample])
    test_labels = one_hot_encoding(y_test[:test_sample])
    print(training_labels[0])
    print(training_labels[1])
    print(training_labels[2])
    print(y_train[0])
    print(y_train[1])
    print(y_train[2])

    return (
        training_images,
        test_images,
        training_labels,
        test_labels,
    )

def train():
    training_images, test_images, training_labels, test_labels = prehandle()
    learning_rate = 0.001
    epochs = 100
    hidden_size = 500
    pixels_per_image = 784
    num_labels = 10
    
    weights = rng.random((pixels_per_image, hidden_size))
    weights_1 = 0.2 * rng.random((pixels_per_image, hidden_size)) - 0.1
    weights_2 = 0.2 * rng.random((hidden_size, num_labels)) - 0.1
    print(f'weights = {weights}')
    print(f'weights_1 = {weights_1}')
    print(f'weights_2 = {weights_2}')
    
    
    store_training_loss = []
    store_training_accurate_pred = []
    store_test_loss = []
    store_test_accurate_pred = []

    # This is a training loop.
    # Run the learning experiment for a defined number of epochs (iterations).
    for j in range(epochs):

    #################
    # Training step #
    #################

    # Set the initial loss/error and the number of accurate predictions to zero.
        training_loss = 0.0
        training_accurate_predictions = 0

    # For all images in the training set, perform a forward pass
    # and backpropagation and adjust the weights accordingly.
        for i in range(len(training_images)):
        # Forward propagation/forward pass:
        # 1. The input layer:
        #    Initialize the training image data as inputs.
            layer_0 = training_images[i]
        # 2. The hidden layer:
        #    Take in the training image data into the middle layer by
        #    matrix-multiplying it by randomly initialized weights.
            layer_1 = np.dot(layer_0, weights_1)
        # 3. Pass the hidden layer's output through the ReLU activation function.
            layer_1 = relu(layer_1)
        # 4. Define the dropout function for regularization.
            dropout_mask = rng.integers(low=0, high=2, size=layer_1.shape)
        # 5. Apply dropout to the hidden layer's output.
            layer_1 *= dropout_mask * 2
        # 6. The output layer:
        #    Ingest the output of the middle layer into the the final layer
        #    by matrix-multiplying it by randomly initialized weights.
        #    Produce a 10-dimension vector with 10 scores.
            layer_2 = np.dot(layer_1, weights_2)

        # Backpropagation/backward pass:
        # 1. Measure the training error (loss function) between the actual
        #    image labels (the truth) and the prediction by the model.
            training_loss += np.sum((training_labels[i] - layer_2) ** 2)
        # 2. Increment the accurate prediction count.
            training_accurate_predictions += int(np.argmax(layer_2) == np.argmax(training_labels[i]))
        # 3. Differentiate the loss function/error.
            layer_2_delta = training_labels[i] - layer_2
        # 4. Propagate the gradients of the loss function back through the hidden layer.
            layer_1_delta = np.dot(weights_2, layer_2_delta) * relu2deriv(layer_1)
        # 5. Apply the dropout to the gradients.
            layer_1_delta *= dropout_mask
        # 6. Update the weights for the middle and input layers
        #    by multiplying them by the learning rate and the gradients.
            weights_1 += learning_rate * np.outer(layer_0, layer_1_delta)
            weights_2 += learning_rate * np.outer(layer_1, layer_2_delta)

    # Store training set losses and accurate predictions.
            store_training_loss.append(training_loss)
            store_training_accurate_pred.append(training_accurate_predictions)

    ###################
    # Evaluation step #
    ###################

    # Evaluate model performance on the test set at each epoch.

    # Unlike the training step, the weights are not modified for each image
    # (or batch). Therefore the model can be applied to the test images in a
    # vectorized manner, eliminating the need to loop over each image
    # individually:

            results = relu(test_images @ weights_1) @ weights_2

    # Measure the error between the actual label (truth) and prediction values.
            test_loss = np.sum((test_labels - results) ** 2)

    # Measure prediction accuracy on test set
            test_accurate_predictions = np.sum(
                np.argmax(results, axis=1) == np.argmax(test_labels, axis=1))

    # Store test set losses and accurate predictions.
            store_test_loss.append(test_loss)
            store_test_accurate_pred.append(test_accurate_predictions)

    # Summarize error and accuracy metrics at each epoch
        print(
            "\n"
        + "Epoch: "
        + str(j)
        + " loss:" + str(training_loss)
        + " Training set error:"
        + str(training_loss / float(len(training_images)))[0:5]
        + " Training set accuracy:"
        + str(training_accurate_predictions / float(len(training_images)))
        + " Test set error:"
        + str(test_loss / float(len(test_images)))[0:5]
        + " Test set accuracy:"
        + str(test_accurate_predictions / float(len(test_images)))
        )


def lenet5():
    training_images, test_images, training_labels, test_labels = prehandle()
    logger.debug(training_images[0].shape)


class MnistLecunUnitTests(UnitTests):
    def __init__(self):
        super().__init__(__file__)

    @UnitTests.skip
    def show_images_test(self):
        show(mnist_dataset['training_images'])
        logger.debug(mnist_dataset['training_labels'][0])
        
    @UnitTests.skip
    def prehandle_test(self):
        prehandle()
        
        
    @UnitTests.skip
    def train_test(self):
        train()

    # @UnitTests.skip
    def lenet5_test(self):
        lenet5()
      
MnistLecunUnitTests().run()


