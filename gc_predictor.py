

#from __future__ import absolute_import

import tensorflow as tf
import os
import numpy as np
#from TF_stixels.code.model import model_fn, params
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import glob
from PIL import Image
import sys
import time
from gc_image2TFRec import Frame2StxTfrecords
#import csv

# params
H = 370
C = 3

# Init CRF parameters
'''
N = 1
T = 20 #20
W_trans = 10
'''

N = 0
T = 20 #20
W_trans = 20

#######################################
###   Creating a dataset_input_fn   ###
#######################################

'''
    dataset_input_fn - Constructs a Dataset, Iterator, and returns handles that will be called when
    Estimator requires a new batch of data. This function will be passed into 
    Estimator as the input_fn argument.

    Inputs:
        mode: string specifying whether to take the inputs from training or validation data
    Outputs:
        features: the columns of feature input returned from a dataset iterator
        labels: the columns of labels for training return from a dataset iterator
'''


def dataset_input_fn(mode, data_files):
    # Function that does the heavy lifting for constructing a Dataset
    #    depending on the current mode of execution
    dataset = load_dataset(mode,data_files)
    # Making an iterator that runs from start to finish once
    #    (the preferred type for Estimators)
    iterator = dataset.make_one_shot_iterator()
    # Consuming the next batch of data from our iterator

    '''
    features, label, frame_id, frame_name = iterator.get_next()
    return features, label, frame_id, frame_name
    '''

    features, frame_id = iterator.get_next()
    return features, frame_id


####################################
###   Constructing the Dataset   ###
####################################

'''
    load_dataset() - Loads and does all processing for a portion of the dataset specified by 'mode'.

    Inputs:
        mode: string specifying whether to take the inputs from training or validation data

    Outputs:
        dataset: the Dataset object constructed for this mode and pre-processed
'''

def load_dataset(mode, data_files):
    # Taking either the train or validation files from the dictionary we constructed above
    files = data_files[mode]
    # Created a Dataset from our list of TFRecord files
    dataset = tf.data.TFRecordDataset(files)
    # Apply any processing we need on each example in our dataset.  We
    #    will define parse next.  num_parallel_calls decides how many records
    #    to apply the parse function to at a time (change this based on your
    #    machine).
    dataset = dataset.map(parse, num_parallel_calls=2)
    # Shuffle the data if training, for validation it is not necessary
    # buffer_size determines how large the buffer of records we will shuffle
    #    is, (larger is better!) but be wary of your memory capacity.
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=1000)
    # Batch the data - you can pick a batch size, maybe 32, and later
    #    we will include this in a dictionary of other hyper parameters.
    dataset = dataset.batch(params.batch_size)
    return dataset


#######################################
###   Defining the Parse Function   ###
#######################################


def parse(serialized):
    # Define the features to be parsed out of each example.
    #    You should recognize this from when we wrote the TFRecord files!
    '''
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'frame_id': tf.FixedLenFeature([], tf.int64),
        'name': tf.FixedLenFeature([], tf.string),
    }
    '''

    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'frame_id': tf.FixedLenFeature([], tf.int64),
    }

    # Parse the features out of this one record we were passed
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)
    # Format the data
    image_raw = parsed_example['image']
    image = tf.image.decode_png(image_raw, channels=3,
                                dtype=tf.uint8)  # Decode the raw bytes so it becomes a tensor with type.
    image = tf.cast(image, tf.float32)  # The type is now uint8 but we need it to be float.

    '''
    label = parsed_example['label']
    frame_id = parsed_example['frame_id']
    frame_name = parsed_example['name']    
    return {'image': image}, label, frame_id, frame_name
    '''

    frame_id = parsed_example['frame_id']
    return {'image': image}, frame_id

class image_predictor:
    def __init__(self, image_in, out_folder, image_width, model_dir, debug_image, show_images):

        # load the correct model
        if os.path.exists(model_dir + '/model_for_CRF.py'):
            from model_for_CRF import model_fn, params
            print('impotrting model function')
        else:
            print('No model file within directory - exiting!!!!')

        # Init class internal params
        self.out_folder = out_folder
        self.model_name = os.path.basename(model_dir)
        self.model_dir = model_dir
        self.image_folder = os.path.dirname(out_folder)
        self.plot_border_width = 2.0
        im = Image.open(image_in)
        self.image_size = im.size
        self.image_width = image_width
        self.debug_image = debug_image
        self.show_images = show_images
        print('image size =  {}'.format(self.image_size))

        self.W = params.image_width
        self.model_fn = model_fn
        self.params = params

        # Reset tf graph
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.INFO)  # possible values - DEBUG / INFO / WARN / ERROR / FATAL

        # Prepare data structure for softmax predictions
        self.grid_x_width = int((self.image_width - 36) / 5) + 1
        self.grid_y_width = 74

        # Create data grid
        x = np.linspace(0, self.grid_x_width - 1, self.grid_x_width) * 5 + 18 - 3  # reduce 3 to center the probability points
        y = np.linspace(0, self.grid_y_width - 1, self.grid_y_width) * 5 - 2
        self.X, self.Y = np.meshgrid(x, y)
        #print(np.shape(self.X), np.shape(self.Y))

        # Create session
        self.sess =  tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Load the model
        self.estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir, params=params)

    def close_session(self):
        print('Closing the session')
        self.sess.close()

    #############################################################
    ###   Visualizing predictions and creating output video   ###
    #############################################################

    def visualize_pred(self, image_in, tfrecord_file, predictions_list):

        # Pipeline of dataset and iterator
        dataset = tf.data.TFRecordDataset([tfrecord_file])
        dataset = dataset.map(parse)
        iterator = dataset.make_one_shot_iterator()
        next_image_data = iterator.get_next()

        num_of_stixels = len(predictions_list)
        print('Number of stixels to be proceesed  {}'.format(num_of_stixels))

        # Go through the TFRecord and reconstruct the images + predictions

        # Init new image
        new_im = Image.new('RGB', (self.image_width, 370))
        grid = np.zeros((self.grid_y_width, self.grid_x_width))
        x_offset = 0
        first_time = True
        fig, ax = plt.subplots()

        # Go through all the stixels in the tfrecord file
        for i in range(num_of_stixels):
            image_data = self.sess.run(next_image_data)
            image = image_data[0]['image']
            im = Image.fromarray(np.uint8(image))
            frame_id = image_data[1]
            prediction = predictions_list[i]['classes']
            prediction_softmax = predictions_list[i]['probabilities']

            #####################################################################################
            ## Collect all image predictions into a new array tp be filtered by CRF++/CRFSuite ##
            #####################################################################################

            # Collect and visualize stixels
            new_im.paste(im, (frame_id * 5, 0))
            x_offset += 5
            if self.debug_image:
                plt.plot(int(params.image_width / 2) + 5 * (frame_id), prediction * 5, marker='o', markersize=4, color="red")
            # visualize probabilities
            grid[:,frame_id] = prediction_softmax
            plt.draw()

        # Use CRF to find the best path
        from gc_crf import viterbi
        best_path = viterbi(grid.T, N, T, W_trans)

        # Plot the CRF border line
        best_path_points = []
        for index, path in enumerate(best_path):
            best_path_points.append([int(params.image_width / 2) + index*5, path*5 + 2])
        plt.plot(np.array(best_path_points)[:,0], np.array(best_path_points)[:,1], color="blue", linewidth=self.plot_border_width)

        if self.debug_image:
            #In debug mode plot the softmax probabilities
            grid = np.ma.masked_array(grid, grid < .0001)
            plt.pcolormesh(self.X, self.Y, grid, norm=colors.LogNorm(), alpha = 0.75)
        plt.imshow(new_im, cmap='gray', alpha=1.0, interpolation='none')

        name = ' {} N{}_T{}_Tr{}.jpg'.format(self.model_name, N, T, W_trans)
        if self.debug_image:
            name.replace('.jpg',' debug.jpg')
            name = ' {} N{}_T{}_Tr{} debug.jpg'.format(self.model_name, N, T, W_trans)
            print('replacing name to indicate debug !!!!!!')
            print(name)

        image_out_name = os.path.basename(image_in)
        image_out_name = image_out_name.replace('.jpg', name)
        image_out_name = os.path.basename(image_out_name)
        image_file_name = os.path.join(self.out_folder, image_out_name)
        plt.savefig(image_file_name, format='jpg')
        print('saving fig to ', image_file_name)

        if self.show_images:
            plt.show()
        plt.close()


    ######################################
    ### translate image ro a TFRecord ###
    #####################################

    def image_2_tfrec(self, image_in, tfrec_filename, model_stixel_width):

        start_time = time.time()
        # Create TFRec writer
        os.chdir(os.path.dirname(image_in))
        print('TFRec output file = ', tfrec_filename)
        writer = tf.python_io.TFRecordWriter(tfrec_filename)

        # parse the image and save it to TFrecord
        f_to_stx = Frame2StxTfrecords(image_in, writer, model_stixel_width)
        f_to_stx.create_stx(False)
        writer.close()

        duration = time.time() - start_time
        print('TFRec creation took {} sec'.format(int(duration)))

    ######################################
    ###      start a new prediction   ###
    #####################################

    def predict(self, image_in):

        # Translate the image to a TF record
        tfrec_filename = image_in.replace('.jpg', '_W' + str(self.W) + '.tfrecord')
        self.image_2_tfrec(image_in, tfrec_filename, self.W)
        data_files = {'test': tfrec_filename}
        predictions = self.estimator.predict(input_fn=lambda: dataset_input_fn('test', data_files))

        # Predict!
        predictions_list = list(predictions)

        # Visualize predictions based on single test TFrecord
        self.visualize_pred(image_in, tfrec_filename, predictions_list)

if __name__ == '__main__':

    # Determine input image
    cwd = os.getcwd()
    image_in = os.path.join(cwd, 'frame_000136.jpg')
    image_width = 476

    if os.path.exists(image_in):
        image_out_dir = os.path.dirname(image_in)
        print(image_out_dir)

        # Determine the model
        #model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2019-02-14_20-17-53_EP_250'
        model_dir = cwd + '/model'
        model_name = os.path.basename(model_dir)

        os.chdir(model_dir)
        sys.path.insert(0, os.getcwd())
        if os.path.exists(model_dir + '/model_for_CRF.py'):
            from model_for_CRF import model_fn, params

            # Create image_predictor object
            predictor = image_predictor(image_in, image_out_dir, image_width, model_dir, debug_image=True, show_images=False)

            # Predict
            predictor.predict(image_in)

            # Close the session
            predictor.close_session()
        else:
            print('No model file within directory - exiting!!!!')

    else:
        print('no such file')










