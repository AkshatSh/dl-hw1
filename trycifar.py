from uwnet import *

def softmax_model():
    l = [make_connected_layer(3072, 10, SOFTMAX)]
    return make_net(l)

def neural_net():
    l = [   make_connected_layer(3072, 32, LRELU),
            make_connected_layer(32, 10, SOFTMAX)]
    return make_net(l)

def conv_net():
    # How many operations are needed for a forard pass through this network?
    # Your answer: 
    '''
    Each layer computes a size * size * filter for each input in the input tensor

    l1: 32 * 32 * 3 * 8 * 3 * 3 = 221184
    l2: 32 / 2 * 32 / 2 * 8 * 3 * 3 = 18432
    l3: 16 * 16 * 8 * 16 * 3 * 3 = 294912
    l4: 16 /2 * 16 /2 * 16 * 3 * 3 = 9216
    l5: 8 * 8 * 16 * 32 * 3 * 3 = 294912
    l6: 8 / 2 * 8 / 2 * 32 * 3 * 3 = 4608
    l7: 4 * 4 * 32 * 64 * 3 * 3 = 294912
    l8: 4 /2 * 4 /2 * 64 * 3 * 3 = 1536
    l9: 256 * 10 = 2560

    1,142,272

    '''
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10, SOFTMAX)]
    return make_net(l)

def your_net():
    # Define your network architecture here. It should have 5 layers. How many operations does it need for a forward pass?
    # It doesn't have to be exactly the same as conv_net but it should be close.
    '''

    The number of operations the network takes depends on the batch size, 
    This network takes 893568 * batch_size.

    Operations per layer:

    l1: batch * 3072 * 256  = (batch , 256)
    l2: batch * 256 * 256 = (batch , 256)
    l3: batch * 256 * 128 = (batch, 128)
    l4: batch * 128 * 64 = (batch, 64)
    l5: batch * 64 * 10 = (batch, 10)

    Total Operations: 893568 * batch_size
    '''
    l = [   
        make_connected_layer(3072, 256 , LRELU),
        make_connected_layer(256, 256 , LRELU),
        make_connected_layer(256, 128 , LRELU),
        make_connected_layer(128, 64 , LRELU),
        make_connected_layer(64, 10, SOFTMAX)
    ]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test", "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = your_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#
'''
The conv net performs around 10% better. The conv net had a test accuracy of about 60%, where as the
feed forward network performs at about 50% accuracy on the test set.
'''

