
'''
*
* model
*
* Purpose: the module implements a MobileNetV2 stixels model
*
* Inputs:
*
*
* Outputs:
*
*
* Conventions: (x=0, y=0) is the upper left corner of the image
*
* Written by: Ran Zaslavsky 10-12-2018 (framework originates from excellent https://crosleythomas.github.io/blog/)
*
'''

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
import os

# params - defining the stixel dimensions for the entire toolchain (RAN)
H = 370
W = 36 # was 24
C = 3

' the next four cells are a modification of https://github.com/xiaochus/MobileNetV2/blob/master/mobilenet_v2.py '

"""
*   Convolution Block
*
*   Purpose: This function defines a 2D convolution operation with BN and relu6.
*
*   Arguments:
*        inputs: Tensor, input tensor of conv layer.
*        filters: Integer, the dimensionality of the output space.
*        kernel: An integer or tuple/list of 2 integers, specifying the
*            width and height of the 2D convolution window.
*        strides: An integer or tuple/list of 2 integers,

*            specifying the strides of the convolution along the width and height.
*            Can be a single integer to specify the same value for
*            all spatial dimensions.
*   Returns:
*        Output tensor.
"""

def _conv_block(inputs, filters, kernel, strides, is_training):

    x = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel,
        activation=None,
        strides=strides,
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=481),
        padding='same')

    x = tf.layers.batch_normalization(
        inputs=x,
        training=is_training)

    x = tf.nn.relu6(features=x)

    return x

"""
*   Bottleneck
*
*   Purpose: This function defines a basic bottleneck structure.
*
*   Arguments:
*        inputs: Tensor, input tensor of conv layer.
*        filters: Integer, the dimensionality of the output space.
*        kernel: An integer or tuple/list of 2 integers, specifying the
*            width and height of the 2D convolution window.
*        t: Integer, expansion factor.
*            t is always applied to the input size.
*        s: An integer or tuple/list of 2 integers,specifying the strides
*            of the convolution along the width and height.Can be a single
*            integer to specify the same value for all spatial dimensions.
*        r: Boolean, Whether to use the residuals.
*   Returns:
*        Output tensor.
"""

def _bottleneck(inputs, filters, kernel, t, s, is_training, r=False):

    num_filters_in = utils.last_dimension(inputs.get_shape(), min_rank=4)

    # "Expension layer" + BN + activation
    x = _conv_block(
        inputs=inputs,
        filters=num_filters_in * t,
        kernel=(1, 1),
        strides=(1, 1),
        is_training=is_training)

    # Depthwise convolution + BN + activation

    x = tf.contrib.layers.separable_conv2d(
        inputs=x,
        num_outputs=None,
        kernel_size=kernel,
        depth_multiplier=1,
        stride=(s, s),
        padding='SAME',
        activation_fn=None,  # tf.nn.relu6,
        weights_initializer=tf.contrib.layers.xavier_initializer(seed=481),
        normalizer_fn=None)

    x = tf.layers.batch_normalization(
        inputs=x,
        training=is_training)

    x = tf.nn.relu6(features=x)
    # x = tf.nn.leaky_relu(features=x) # RAN - trial

    # "Projection" layer + BN
    x = tf.layers.conv2d(
        inputs=x,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=481))

    x = tf.layers.batch_normalization(
        inputs=x,
        training=is_training)

    if r:
        x = tf.add(x, inputs)


    return x

"""
*   Inverted Residual Block
*
*   This function defines a sequence of 1 or more identical layers.
*
*   Arguments:
*        inputs: Tensor, input tensor of conv layer.
*        filters: Integer, the dimensionality of the output space.
*        kernel: An integer or tuple/list of 2 integers, specifying the
*            width and height of the 2D convolution window.
*        t: Integer, expansion factor.
*            t is always applied to the input size.
*        s: An integer or tuple/list of 2 integers,specifying the strides
*            of the convolution along the width and height.Can be a single
*            integer to specify the same value for all spatial dimensions.
*        n: Integer, layer repeat times.
*
*   Returns:
*        Output tensor.
"""

def _inverted_residual_block(inputs, filters, kernel, t, strides, n, is_training):

    x = _bottleneck(
        inputs=inputs,
        filters=filters,
        kernel=kernel,
        t=t,
        s=strides,
        is_training=is_training)

    for i in range(1, n):
        x = _bottleneck(
            inputs=x,
            filters=filters,
            kernel=kernel,
            t=t,
            s=1,
            is_training=is_training,
            r=True)

    return x


"""
*    MobileNetV2
* 
*    Purpose: This function defines a MobileNetV2 architectures.
*
*    Arguments:
*        inputs: A tensor of the input of shape [-1,W,H,C].
*        k: Integer, number of classes.
*        is_training: boolean indication training or prediction
*
*    Returns:
*        MobileNetV2 model
"""

def MobileNetV2(inputs, k, is_training):

    with tf.variable_scope('Conv-block'):
        x = _conv_block(inputs=inputs, filters=32, kernel=(3, 3), strides=(2, 2), is_training=is_training)

    with tf.variable_scope("IRblock-1"):
        x = _inverted_residual_block(inputs=x, filters=16, kernel=(3, 3), t=1, strides=1, n=1, is_training=is_training)
    with tf.variable_scope("IRblock-2"):
        x = _inverted_residual_block(inputs=x, filters=24, kernel=(3, 3), t=6, strides=2, n=2, is_training=is_training)
        x = tf.layers.dropout(inputs=x, rate=0.5, noise_shape=None, seed=None, training=is_training, name=None) # RAN - adding dropout
    with tf.variable_scope("IRblock-3"):
        x = _inverted_residual_block(inputs=x, filters=32, kernel=(3, 3), t=6, strides=2, n=3, is_training=is_training)
        x = tf.layers.dropout(inputs=x, rate=0.5, noise_shape=None, seed=None, training=is_training, name=None)  # RAN - adding dropout
    with tf.variable_scope("IRblock-4"):
        x = _inverted_residual_block(inputs=x, filters=64, kernel=(3, 3), t=6, strides=2, n=4, is_training=is_training)
    with tf.variable_scope("IRblock-5"):
        x = _inverted_residual_block(inputs=x, filters=96, kernel=(3, 3), t=6, strides=1, n=3, is_training=is_training)

    x = tf.layers.average_pooling2d(inputs=x, pool_size=(24, 2), strides=(1, 1))

    # Eventually this should be replaced with:
    x = tf.layers.flatten(x)

    # This is the last layer so it does not use an activation function.
    x = tf.layers.dense(
        inputs=x, name='layer_fc6',
        units=k,
        kernel_initializer=tf.contrib.layers.xavier_initializer())

    return x

'''
*
*   model_fn()
*
*   Purpose: Defines the model function passed into tf.estimator.
*   This function defines the computational logic for the model.
*
*   Implementation:
*    1. Define the model's computations with TensorFlow operations
*    2. Generate predictions and return a prediction EstimatorSpec
*    3. Define the loss function for training and evaluation
*    4. Define the training operation and optimizer
*    5. Return loss, train_op, eval_metric_ops in an EstimatorSpec
*
*    Inputs:
*        features: A dict containing the features passed to the model via input_fn
*        labels: A Tensor containing the labels passed to the model via input_fn
*        mode: One of the following tf.estimator.ModeKeys string values indicating
*               the context in which the model_fn was invoked
*                  - tf.estimator.ModeKeys.TRAIN ---> model.train()
*                  - tf.estimator.ModeKeys.EVAL, ---> model.evaluate()
*                  - tf.estimator.ModeKeys.PREDICT -> model.predict()
*
*    Outputs:
*        tf.EstimatorSpec that defines the model in different modes.
*
'''

##################################################################################################################################################################################################
### follow https://colab.research.google.com/github/amygdala/tensorflow-workshop/blob/master/workshop_sections/high_level_APIs/mnist_cnn_custom_estimator/cnn_mnist_tf.ipynb#scrollTo=Dl8XIZS3P7sE
##################################################################################################################################################################################################


def model_fn(features, labels, mode, params):

    # decide if training or not
    if mode == tf.estimator.ModeKeys.TRAIN:
        is_training = True
        print('Model training mode')
    else:
        is_training = False
        print('Model eval mode')


    # Reference to the tensor named "image" in the input-function.
    with tf.variable_scope('Reshape'):
        x = features["image"]
        # Reshape from 2-rank tensor to a 4-rank tensor expected by the convolutional layers
        x_image = tf.reshape(x, [-1, H, W, C])
        #inputs = tf.reshape(x, [-1, H, W, C])

    # 1. Define model structure
    with tf.variable_scope('mobilenetV2'):
        net = MobileNetV2(inputs=x_image, k=74, is_training=is_training) # RAN changed k to 74 (0-73) - temporary!!!!!
    #net = MobileNetV2(inputs=inputs, k=73, is_training=is_training)

    # Logits output of the neural network.
    logits = net

    # Softmax output of the neural network.
    #y_pred = tf.nn.softmax(logits=logits)

    # Classification output of the neural network.
    #y_pred_cls = tf.argmax(y_pred, axis=1)


    # Generate predictions (for PREDICT and EVAL mode) - consider if classes = argmax(logits) or argmax(softmax()) as previously !!!!!!!!!!!!!!!!
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"), # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`
    }

    prediction_output = tf.estimator.export.PredictOutput({
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")})

    # 2. Generate predictions
    #if mode == tf.estimator.ModeKeys.PREDICT:
    #    return tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred_cls)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_output})

    # 3. Define the loss functions

    with tf.variable_scope("xent"):
        # calculate the cross-entropy between the output of
        # the neural network and the true labels for the input data.
        # This gives the cross-entropy for each image in the batch.
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits)

        # Reduce the cross-entropy batch-tensor to a single number
        # which can be used in optimization of the neural network.
        tf.argmax(logits)

        loss = tf.reduce_mean(cross_entropy)

    # 3.1 Additional metrics for monitoring (in this case the classification accuracy)
    #with tf.variable_scope("accuracy"):
    #    metrics = {"accuracy": tf.metrics.accuracy(
    #        labels, y_pred_cls)}
    with tf.variable_scope("accuracy"):
        metrics = {"accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

    # 4. Define optimizer
    with tf.variable_scope("train"):
        lr = params.learning_rate
        #lr = 1e-4 # RAN - original StixelNET value
        step_rate = 20000
        #step_rate = 5000 # original StixelNET value
        decay = 0.95 # try to break the 0.23 accuracy barrier ...
        #decay = 0.7  # if this equals 1 the lr stays the same (original StixelNET value)
        learning_rate = tf.train.exponential_decay(
            lr, global_step=tf.train.get_or_create_global_step(),
            decay_steps=step_rate,
            decay_rate=decay,
            staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #print('Learning rate = {}'.format(learning_rate))

        # for learning parameters of batch normalization:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())

    # 5. Return training/evaluation EstimatorSpec
    spec = tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

    tf.summary.scalar("accuracy", metrics["accuracy"][1])
    tf.summary.scalar("loss", loss)
    '''
    with tf.Session() as sess:
        print('Loss & accuracy = ' )
        print(sess.run(loss))
        print(sess.run(metrics["accuracy"][1]))
    '''
    #print('Loss = {} accuracy = {}'.format(loss, metrics["accuracy"][1]))
    tf.summary.image('input', x_image, 1)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    return spec

#######################################
###   Defining the Parse Function   ###
#######################################

def parse(serialized):
    # Define the features to be parsed out of each example.
    #    You should recognize this from when we wrote the TFRecord files!
    features ={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    # Parse the features out of this one record we were passed
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)
    # Format the data
    image_raw = parsed_example['image']
    image = tf.image.decode_png(image_raw, channels=3, dtype=tf.uint8) # Decode the raw bytes so it becomes a tensor with type.
    image = tf.cast(image, tf.float32)  # The type is now uint8 but we need it to be float.

    label = parsed_example['label']
    return {'image': image}, label

'''
# Support added features set
def parse(serialized):
    # Define the features to be parsed out of each example.
    #    You should recognize this from when we wrote the TFRecord files!
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'frame_id': tf.FixedLenFeature([], tf.int64),
        'name': tf.FixedLenFeature([], tf.string),
    }

    # Parse the features out of this one record we were passed
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)
    # Format the data
    image_raw = parsed_example['image']
    image = tf.image.decode_png(image_raw, channels=3, dtype=tf.uint8) # Decode the raw bytes so it becomes a tensor with type.
    image = tf.cast(image, tf.float32)  # The type is now uint8 but we need it to be float.

    label = parsed_example['label']
    frame_id = parsed_example['frame_id']
    frame_name = parsed_example['frame_name']
    return {'image': image}, label, frame_id, frame_name
    '''


###################################
###   Define Hyper Parameters   ###
###################################


'''
params = tf.contrib.training.HParams(
    learning_rate=0.001,
    train_epochs=250,
    batch_size=32,
    image_height=370,
    image_width=W,
    image_depth=1
) # NOTE image_depth reduced to 1 to support BW
'''

# StixelNet initial params (note that params.learning_rate was not used)
params = tf.contrib.training.HParams(
    learning_rate=0.001,
    train_epochs=250,
    batch_size=32,
    image_height=370,
    image_width=W,
    image_depth=3
)

# Run Configuration
config = tf.estimator.RunConfig(
    tf_random_seed=0,
    save_checkpoints_steps=10000,
    save_checkpoints_secs=None,
    save_summary_steps=500,
)

'''
# Run Configuration
config = tf.estimator.RunConfig(
    tf_random_seed=0,
    save_checkpoints_steps=1000,
    save_checkpoints_secs=None,
    save_summary_steps=10,
)
'''
