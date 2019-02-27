'''
*
* folder2TFRec()
*
* Purpose: the module receives a directory name, creates train/valid/test/control/meta_data
*          folders, and scans it's "annotated" folder, translating annotated images into
*          stixels TFrecords. Note that lowest bound limit stixels diluted by a factor of 2
*          to reduce classification bias
*
*
* Inputs:
*   frame_path - annotated image path
*
* Outputs:
*   Stixles tfrecord files are saved into train/valid/test folders
*   meta_data CSV file saved to meta_data folder
*
*  Method:
*   1) Gather all the data (e.g. a list of images and corresponding labels)
*   2) Create a TFRecordWriter
*   3) Create an Example out of Features for each datapoint
*   4) Serialize the Examples
*   5) Write the Examples to your TFRecord
*
* Conventions: (x=0, y=0) is the upper left corner of the image
*
* Written by: Ran Zaslavsky 10-12-2018
'''


# imports

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import io


TRAIN_IMAGES_RATIO = 80
VALID_IMAGES_RATIO = 15

class Frame2StxTfrecords:
    #def __init__(self, frame_path, GT_file_path, writer, control_dir, stixel_width, frame_type):
    def __init__(self, frame_path, writer, stixel_width):
        ' Define stixels dimensions'
        self.stx_w = stixel_width  # stixel width
        self.stx_half_w = int(self.stx_w/2)
        self.stx_h = 370  # stixel height
        self.bin_pixels = 5  # stixel bins height
        self.stride = 5  # stride used to split to the frame

        ' init internal data '
        self.frame_path = frame_path  # frame path
        self.writer = writer
        self.labels = []


    def int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def float_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))

    def string_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.strip().encode("ascii")]))

    ## NEW: supporting frame_id for CRF prediction - may not be used during training
    def create_tf_example_2(self, img_raw, label):
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image': self.bytes_feature(img_raw),
            'frame_id': self.int64_feature(label),
        }))
        return tf_example

    def create_stx(self, printCtrlImage):
        ' read current frame '
        img = mpimg.imread(self.frame_path)  # img is np.array of the frame
        height, width, c = img.shape
        #print('image dimensions: h={} w={} c={}'.format(height, width, c))

        print(width)

        x_start = 0
        x_stop = width-4 # EMBEDDED - handle properly !!!!!!!!!

        annotated_w = x_stop - x_start + 1
        #print('start={} stop={} w={}'.format(x_start, x_stop, annotated_w))
        num_stixels = int(((annotated_w - self.stx_w) / self.stride) + 1)
        #print('stixel width = {}, number of stixles to be generated {}'.format(self.stx_w, num_stixels))
        frame_name = os.path.basename(self.frame_path)

        ' display the current frame'
        fig, ax = plt.subplots()
        ax.imshow(img)

        for stixel in range(num_stixels):

            imName = '_'.join([os.path.splitext(frame_name)[0], 'stx', str(stixel).zfill(3)]) # RAN - 13-12
            #print('image name = ' + imName)
            #imName = '_'.join([self.output_prefix, os.path.splitext(frame_name)[0], 'stx', str(stixel).zfill(3)])

            i = self.stx_half_w + (stixel * 5) + x_start
            #i = 12 + (stixel * 5) + x_start
            #print('\nstixel {} center = {}'.format(stixel, i))
            ' cut the lower image part (high y values)'
            if img.shape[0] == self.stx_h:
                s = img[:, i - self.stx_half_w:i + self.stx_half_w, :]  # that's the stixel
                print('diff_h not defined !!!!!!!!')
            else:
                diff_h = img.shape[0] - self.stx_h
                s = img[diff_h:, i - self.stx_half_w:i + self.stx_half_w, :]  # that's the stixel

            ' save a tfrecord file'
            img_for_tfrec = Image.fromarray(s)
            with io.BytesIO() as output:
                img_for_tfrec.save(output, format="PNG")
                contents = output.getvalue()
            tf_example = self.create_tf_example_2(contents, stixel)  ### NEW - save the frame_id, rather than stixel label
            self.writer.write(tf_example.SerializeToString())

        plt.close('all')

        return self.labels


def main(data_dir, stixel_width, isControl = True):

    print('main: Do Nothing!')


if __name__ == '__main__':

    print('Do nothing!')



