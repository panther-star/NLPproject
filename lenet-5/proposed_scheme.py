from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time
import math
from sklearn.cluster import KMeans

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#original = __import__('original_lenet-5_on_mnist')


# CVDF mirror of http://yann.lecun.com/exdb/mnist/
SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
IMAGE_SIZE_FOR_LENET = 32
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 50
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.

#PRUNING_RATIO=0.98
#REMAINED_COLUMN = math.floor(48000*(1-PRUNING_RATIO)/400)
#PRUNING_RATIO_REAL = (48000-400*REMAINED_COLUMN)/48000
#NUM_OF_PRUNING_STEP=5
#PRUNING_RATIO_PER_ITERATION=PRUNING_RATIO/NUM_OF_PRUNING_STEP
#NUM_OF_PRUNING_COLUMN = 120-REMAINED_COLUMN
column_num = 1
quality_parameter = 1.7
analysis = False



FLAGS = None


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels


def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = np.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=np.float32)
  labels = np.zeros(shape=(num_images,), dtype=np.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == labels) /
      predictions.shape[0])


def get_min_order_index_compare(w, axis_1):
    original_index_list=[]
    abs_w=np.abs(w)
    reshape=abs_w.reshape(-1)
    index_array=np.argsort(reshape)
    for i in range(index_array.size):
        row=index_array[i]//axis_1
        col=index_array[i]%axis_1
        original_index_list.append([row,col])
    return original_index_list


def get_min_order_index(w, threshold, axis_1):
        original_index_list=[]
        abs_w=np.abs(w)
        reshape=abs_w.reshape(-1)

        reshape_sort = np.sort(reshape)
        the_number_of_under_threshold = 0
        for i in range(reshape_sort.size):
                if reshape_sort[i] <= threshold:
                        the_number_of_under_threshold+=1
                else:
                        break

        index_array=np.argsort(reshape)
        for i in range(index_array.size):
                row=index_array[i]//axis_1
                col=index_array[i]%axis_1
                original_index_list.append([row,col])
        return original_index_list, the_number_of_under_threshold


#def get_min_order_index(w):
#  original_index_list=[]
#  abs_w=np.abs(w)
#  reshape=abs_w.reshape(-1)
#  index_list=np.argsort(reshape)
#  for i in range(len(index_list)):
#    row=index_list[i]//120
#    col=index_list[i]%120
#    original_index_list.append([row,col])
#  return original_index_list


def prune_weights(num_of_pruning, array, min_order_index_list, retrain_iteration):
#       num_of_pruning=math.floor(len(min_order_index_list)*PRUNING_RATIO_PER_ITERATION*pruning_step)
        num_of_pruning_per_iteration_list = []
        reset_weight_list = []
        num_of_pruning_regular = num_of_pruning//retrain_iteration

        reset_array = np.ones((array.shape[0], array.shape[1]), dtype = np.float32)


        for i in range(retrain_iteration):
                if i != (retrain_iteration-1):
                        num_of_pruning_per_iteration_list.append(num_of_pruning_regular * (i+1))
                else:
                        num_of_pruning_per_iteration_list.append(num_of_pruning)

        for i in range(retrain_iteration):
                for j in range(num_of_pruning_per_iteration_list[i]):
                        row=min_order_index_list[j][0]
                        col=min_order_index_list[j][1]
                        reset_array[row][col]=0.0
                reset_weight_list.append(reset_array)
        return reset_weight_list


#def prune_weights(pruning_step, fc1_weights, min_order_index_list):
#  num_of_pruning=math.floor(len(min_order_index_list)*PRUNING_RATIO_PER_ITERATION*pruning_step)
#  for i in range(num_of_pruning):
#    row=min_order_index_list[i][0]
#    col=min_order_index_list[i][1]
#    fc1_weights[row][col]=0
#  return fc1_weights


def reset_weights(pruning_step, fc1_weights, min_order_index_list):
  num_of_pruning=math.floor(len(min_order_index_list)*PRUNING_RATIO_PER_ITERATION*pruning_step)
  for i in range(num_of_pruning):
    row=min_order_index_list[i][0]
    col=min_order_index_list[i][1]
    fc1_weights[row][col]=0
  return fc1_weights


def calculate_l1_norm_of_each_neuron(fc2_weights):
  value_list = []
  abs_w = np.abs(fc2_weights)
  for i in range (fc2_weights.shape[0]):
    l1_norm = 0
    for j in range (fc2_weights.shape[1]):
      l1_norm += abs_w[i][j]
    value_list.append(l1_norm)
  value_array = np.array(value_list)
  index_array = np.argsort(value_array)
  return index_array


def delete_column_of_fc1_weights(fc1_weights, index_array):
  delete_column_list = []
  for i in range (NUM_OF_PRUNING_COLUMN):
#  for i in range (n_column):
    delete_column_list.append(index_array[i])
  fc1_weights_del = np.delete(fc1_weights, delete_column_list, 1)
  return fc1_weights_del


def delete_row_of_fc2_weights(fc2_weights, index_array):
  delete_row_list = []
  for i in range (NUM_OF_PRUNING_COLUMN):
#  for i in range (n_column):
    delete_row_list.append(index_array[i])
  fc2_weights_del = np.delete(fc2_weights, delete_row_list, 0)
  return fc2_weights_del


def delete_element_of_fc1_biases(fc1_biases, index_array):
  delete_column_list = []
  for i in range (NUM_OF_PRUNING_COLUMN):
#  for i in range (n_column):  
    delete_column_list.append(index_array[i])
  fc1_biases_del = np.delete(fc1_biases, delete_column_list, None)
  return fc1_biases_del


def merge_matrix(pruned_weights, pruned_biases, next_weights, cluster_list, centroid_array, n_clusters):

    wfc1_non_zero_element_before_merge = []
    wfc1_non_zero_element_after_merge = []
    wfc2_non_zero_element_before_merge = []
    wfc2_non_zero_element_after_merge = []
    reduction_nz_wfc1 = []
    reduction_nz_wfc2 = []
    reduction_col = []
    slope_wfc1 = []
    slope_wfc2 = []
    delete_list = []
    pruned_weights_copy = np.zeros((pruned_weights.shape[0], pruned_weights.shape[1]))
    next_weights_copy = np.zeros((next_weights.shape[0], next_weights.shape[1]))


    for i in range(pruned_weights.shape[0]):
        for j in range (pruned_weights.shape[1]):
            pruned_weights_copy[i][j] = pruned_weights[i][j]

    for i in range(next_weights.shape[0]):
        for j in range (next_weights.shape[1]):
            next_weights_copy[i][j] = next_weights[i][j]

    


    for i in range(n_clusters):
        count = 0
        for j in range(len(cluster_list[i])):
            for k in range(pruned_weights.shape[0]):
                if pruned_weights[k][cluster_list[i][j]] != 0:
                    count += 1
        wfc1_non_zero_element_before_merge.append(count)

        count = 0
        for j in range(len(cluster_list[i])):
            for k in range(next_weights.shape[1]):
                if next_weights[cluster_list[i][j]][k] != 0:
                    count += 1
        wfc2_non_zero_element_before_merge.append(count)
#    print(N_CLUSTERS)
#    print(fc2_weights.shape)
#    print(fc1_weights.shape)

#    count=0
#    for i in range (N_CLUSTERS):
#        if len(cluster_list[i]) == 0:
#            count+=1
#    print(count)

    for i in range (n_clusters):
        for j in range (next_weights.shape[1]):
            next_weights[cluster_list[i][0]][j] = centroid_array[i][j]

    for i in range (n_clusters):
        if len(cluster_list[i]) == 1:
            continue
        else:
            for j in range (1, len(cluster_list[i])):
                delete_list.append(cluster_list[i][j])

    next_weights_del = np.delete(next_weights, delete_list, 0)


    for i in range(n_clusters):
        count = 0
        for j in range(next_weights.shape[1]):
            if centroid_array[i][j] != 0:
                count += 1
        wfc2_non_zero_element_after_merge.append(count)


#    for i in range (fc1_weights.shape[1]):
#      for j in range (fc1_weights.shape[0]):
#        fc1_weights[j][i] = fc1_weights[j][i]*scale_list[i]
#      fc1_biases[i] = fc1_biases[i]*scale_list[i]


    for i in range (n_clusters):
        if len(cluster_list[i]) == 1:
            continue
        else:
            for j in range (1, len(cluster_list[i])):
                for k in range (pruned_weights.shape[0]):
                    pruned_weights[k][cluster_list[i][0]] += pruned_weights[k][cluster_list[i][j]]
                pruned_biases[cluster_list[i][0]] += pruned_biases[cluster_list[i][j]]

    for i in range(n_clusters):
        count = 0
        for j in range(pruned_weights.shape[0]):
            if pruned_weights[j][cluster_list[i][0]] != 0:
                count += 1
        wfc1_non_zero_element_after_merge.append(count)

    for i in range(n_clusters):
        reduction_nz_wfc1.append(wfc1_non_zero_element_after_merge[i]-float(wfc1_non_zero_element_before_merge[i]))
        reduction_nz_wfc2.append(wfc2_non_zero_element_after_merge[i]-float(wfc2_non_zero_element_before_merge[i]))

        if len(cluster_list[i]) !=1:
            reduction_col.append(len(cluster_list[i])-1)
        else:
            reduction_col.append(1)

        slope_wfc1.append(reduction_nz_wfc1[i]/reduction_col[i])
        slope_wfc2.append(reduction_nz_wfc2[i]/reduction_col[i])

    print("merge analysis")
    print("cluster index, # of element, element, wfc1 before, wfc1 after, wfc2 before, wfc2 after")
    for i in range(n_clusters):
        print('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(i, len(cluster_list[i]), reduction_col[i], cluster_list[i], wfc1_non_zero_element_before_merge[i], wfc1_non_zero_element_after_merge[i], reduction_nz_wfc1[i], wfc2_non_zero_element_before_merge[i], wfc2_non_zero_element_after_merge[i], reduction_nz_wfc2[i], slope_wfc1[i], slope_wfc2[i]))

    avg_wfc1=0
    avg_wfc2=0
    for i in range(n_clusters):
        avg_wfc1 += float(wfc1_non_zero_element_after_merge[i] / wfc1_non_zero_element_before_merge[i])
        avg_wfc2 += float(wfc2_non_zero_element_after_merge[i] / wfc2_non_zero_element_before_merge[i])
    avg_wfc1 = avg_wfc1/n_clusters
    avg_wfc2 = avg_wfc2/n_clusters

    print('average reduction rate')
    print('wfc1:{}, wfc2:{}'.format(avg_wfc1, avg_wfc2))

    for i in range(n_clusters):
        if cluster_list[i][0] == 23:
            print('{}-------{}'.format(cluster_list[i], cluster_list[i][0]))
            col = 0
            for j in range(pruned_weights.shape[0]):
                nz = 0
                print(j, end=" : ")
                for k in range(len(cluster_list[i])):
                    print(pruned_weights_copy[j][cluster_list[i][k]], end=" ")
                    if pruned_weights_copy[j][cluster_list[i][k]] != 0:
                        nz += 1
                print('------ {}'.format(pruned_weights[j][cluster_list[i][0]]))
                if nz >= 2:
                    col += 1
            print('col:{}'.format(col))
            break

    for i in range (10):
        print("")
    for i in range(n_clusters):
        if cluster_list[i][0] == 87:
            print('{}-------{}'.format(cluster_list[i], cluster_list[i][0]))
            col = 0
            for j in range(pruned_weights.shape[0]):
                nz = 0
                print(j, end=" : ")
                for k in range(len(cluster_list[i])):
                    print(pruned_weights_copy[j][cluster_list[i][k]], end=" ")
                    if pruned_weights_copy[j][cluster_list[i][k]] != 0:
                        nz += 1
                print('------ {}'.format(pruned_weights[j][cluster_list[i][0]]))
                if nz >= 2:
                    col += 1
            print('col:{}'.format(col))
            break


    for i in range (10):
        print("")
    for i in range(n_clusters):
#        if len(cluster_list[i]) == 2:
            print('{}-------{}'.format(cluster_list[i], cluster_list[i][0]))
            col = 0
            for j in range(next_weights.shape[1]):
                nz = 0
                print(j, end=" : ")
                for k in range(len(cluster_list[i])):
                    print(next_weights_copy[cluster_list[i][k]][j], end=" ")
                    if next_weights_copy[cluster_list[i][k]][j] != 0:
                        nz += 1
                print('------ {}'.format(centroid_array[i][j]))
#                if nz >= 2:
#                    col += 1
#            print('col:{}'.format(col))


    for i in range (10):
        print("")
    for i in range(n_clusters):
        if len(cluster_list[i]) == 10:
            print('{}-------{}'.format(cluster_list[i], cluster_list[i][0]))
            col = 0
            for j in range(next_weights.shape[1]):
                nz = 0
                print(j, end=" : ")
                for k in range(len(cluster_list[i])):
                    print(next_weights_copy[cluster_list[i][k]][j], end=" ")
                    if next_weights_copy[cluster_list[i][k]][j] != 0:
                        nz += 1
                print('------ {}'.format(centroid_array[i][j]))
#                if nz >= 2:
#                    col += 1
#            print('col:{}'.format(col))

    for i in range (10):
        print("")
    for i in range(n_clusters):
        if len(cluster_list[i]) == 10:
            print('{}-------{}'.format(cluster_list[i], cluster_list[i][0]))
            col = 0
            for j in range(pruned_weights.shape[0]):
                nz = 0
                print(j, end=" : ")
                for k in range(len(cluster_list[i])):
                    print(pruned_weights_copy[j][cluster_list[i][k]], end=" ")
                    if pruned_weights_copy[j][cluster_list[i][k]] != 0:
                        nz += 1
                print('------ {}'.format(pruned_weights[j][cluster_list[i][0]]))
                if nz >= 2:
                    col += 1
            print('col:{}'.format(col))
            break            

    pruned_weights_del = np.delete(pruned_weights, delete_list, 1)
    pruned_biases_del = np.delete(pruned_biases, delete_list, None)

    return pruned_weights_del, pruned_biases_del, next_weights_del



def make_reset_array(weight_array):
    reset_array = np.ones((weight_array.shape[0], weight_array.shape[1]), dtype = np.float32)
    for i in range(weight_array.shape[0]):
        for j in range(weight_array.shape[1]):
            if weight_array[i][j] == 0.0:
               reset_array[i][j] = 0.0

    return reset_array





def main(_):

  retrain_path_pr0 = '/home/h-iwasaki/lenet-5/ckpt/retrain_with_relu/fc1/qp_' + str(quality_parameter)
  if not os.path.exists(retrain_path_pr0):
      os.makedirs(retrain_path_pr0)
  kmeans_path_pr0 = '/home/h-iwasaki/lenet-5/ckpt/kmeans/fc1/analysis/qp_' + str(quality_parameter) + '/' + str(column_num)
  if not os.path.exists(kmeans_path_pr0):
      os.makedirs(kmeans_path_pr0)

  if FLAGS.self_test:
    print('Running self-test.')
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
  else:
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into np arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS
  train_size = train_labels.shape[0]

  # Pad images with 0s
  train_data = np.pad(train_data, ((0,0),(2,2),(2,2),(0,0)), 'constant')
  validation_data = np.pad(validation_data, ((0,0),(2,2),(2,2),(0,0)), 'constant')
  test_data = np.pad(test_data, ((0,0),(2,2),(2,2),(0,0)), 'constant')






  with tf.Session() as sess:

    
    saver = tf.train.import_meta_graph(os.path.join(retrain_path_pr0, 'lenet-5.ckpt.meta'))
    saver.restore(sess, tf.train.latest_checkpoint(retrain_path_pr0))
#    saver.restore(sess, 'ckpt/4/my_model_4')

    graph = tf.get_default_graph()
#    train_data_node = graph.get_tensor_by_name("train_data_node:0")
#    train_labels_node = graph.get_tensor_by_name("train_labels_node:0")
#    eval_data = graph.get_tensor_by_name("eval_data:0")
    conv1_weights = graph.get_tensor_by_name("conv1_weights_un0:0")
    conv1_biases = graph.get_tensor_by_name("conv1_biases_un0:0")
    conv2_weights = graph.get_tensor_by_name("conv2_weights_un0:0")
    conv2_biases = graph.get_tensor_by_name("conv2_biases_un0:0")
    fc1_weights = graph.get_tensor_by_name("fc1_weights_un0:0")
    fc1_biases = graph.get_tensor_by_name("fc1_biases_un0:0")
    fc2_weights = graph.get_tensor_by_name("fc2_weights_un0:0")
    fc2_biases = graph.get_tensor_by_name("fc2_biases_un0:0")
    fc3_weights = graph.get_tensor_by_name("fc3_weights_un0:0")
    fc3_biases = graph.get_tensor_by_name("fc3_biases_un0:0")
#    masked_fc1 = graph.get_tensor_by_name("masked_fc1:0")
#    optimizer = graph.get_tensor_by_name("optimizer:0")
#    predictions = graph.get_tensor_by_name("predictions:0")
#    eval_prediction = graph.get_tensor_by_name("eval_prediction:0")
#    accuracy_test = graph.get_tensor_by_name("accuracy_test:0")
#    assign_mask = graph.get_tensor_by_name("assign_mask:0")

#    prediction_test = eval_in_batches(test_data, sess)

#    accuracy_test_percent = sess.run(accuracy_test, feed_dict={predictions:prediction_test})*100
#    print('test accuracy before pruning  %.8f%%' % accuracy_test_percent)

#    print("finish")

    conv1_weights_val = sess.run(conv1_weights)
    conv1_biases_val = sess.run(conv1_biases)
    conv2_weights_val = sess.run(conv2_weights)
    conv2_biases_val = sess.run(conv2_biases)
    fc1_weights_val = sess.run(fc1_weights)
    fc1_biases_val = sess.run(fc1_biases)
    fc2_weights_val = sess.run(fc2_weights)
    fc2_biases_val = sess.run(fc2_biases)
    fc3_weights_val = sess.run(fc3_weights)
    fc3_biases_val = sess.run(fc3_biases)




  if analysis == True:
      shape_list = []
      non_zero_element = []
      memory_size = []

      shape_list.append(conv1_weights_val.shape)
      shape_list.append(conv1_biases_val.shape)
      shape_list.append(conv2_weights_val.shape)
      shape_list.append(conv2_biases_val.shape)
      shape_list.append(fc1_weights_val.shape)
      shape_list.append(fc1_biases_val.shape)
      shape_list.append(fc2_weights_val.shape)
      shape_list.append(fc2_biases_val.shape)
      shape_list.append(fc3_weights_val.shape)
      shape_list.append(fc3_biases_val.shape)

      non_zero_element.append(conv1_weights_val.size)
      non_zero_element.append(conv1_biases_val.size)
      non_zero_element.append(conv2_weights_val.size)
      non_zero_element.append(conv2_biases_val.size)
      non_zero_element.append(np.count_nonzero(fc1_weights_val))
      non_zero_element.append(fc1_biases_val.size)
      non_zero_element.append(np.count_nonzero(fc2_weights_val))
      non_zero_element.append(fc2_biases_val.size)
      non_zero_element.append(np.count_nonzero(fc3_weights_val))
      non_zero_element.append(fc3_biases_val.size)

      memory_size.append(conv1_weights_val.size*4)
      memory_size.append(conv1_biases_val.size*4)
      memory_size.append(conv2_weights_val.size*4)
      memory_size.append(conv2_biases_val.size*4)
      memory_size.append(min(2*np.count_nonzero(fc1_weights_val)+fc1_weights_val.shape[1]+1, fc1_weights_val.size)*4)
      memory_size.append(fc1_biases_val.size*4)
      memory_size.append(min(2*np.count_nonzero(fc2_weights_val)+fc2_weights_val.shape[1]+1, fc2_weights_val.size)*4)
      memory_size.append(fc2_biases_val.size*4)
      memory_size.append(min(2*np.count_nonzero(fc3_weights_val)+fc3_weights_val.shape[1]+1, fc3_weights_val.size)*4)
      memory_size.append(fc3_biases_val.size*4)

      cnn_memory_size = 0
      for i in range(4):
          cnn_memory_size += memory_size[i]

      fc_memory_size = 0
      for i in range(4, 10):
          fc_memory_size += memory_size[i]

      total_memory_size=0
      for i in range(len(memory_size)):
          total_memory_size += memory_size[i]


      print("before")
      print("shape")
      print("wfc1:{}, bfc1:{}, wfc2:{}, bfc2:{}, wfc3:{}, bfc3:{}".format(shape_list[4], shape_list[5], shape_list[6], shape_list[7], shape_list[8], shape_list[9]))
      print("the number of non zero element")
      print("wfc1:{}, bfc1:{}, wfc2:{}, bfc2:{}, wfc3:{}, bfc3:{}".format(non_zero_element[4], non_zero_element[5], non_zero_element[6], non_zero_element[7], non_zero_element[8], non_zero_element[9]))
      print("memory size[MB]")
      print("wfc1:{}, bfc1:{}, wfc2:{}, bfc2:{}, wfc3:{}, bfc3:{}".format(memory_size[4]/float(1000000), memory_size[5]/float(1000000), memory_size[6]/float(1000000), memory_size[7]/float(1000000), memory_size[8]/float(1000000), memory_size[9]/float(1000000)))
      print("cnn memory size[MB]")
      print(cnn_memory_size/float(1000000))
      print("fc memory size[MB]")
      print(fc_memory_size/float(1000000))
      print("total memory size[MB]")
      print(total_memory_size/float(1000000))


  else:
      n_clusters = column_num
      column_num_reduction_ratio = (float(fc1_weights_val.shape[1]-column_num) / fc1_weights_val.shape[1])*100

      label_list = []
      cluster_centers_list = []
      distance_list = []
      for i in range(10):
          kmeans = KMeans(n_clusters=n_clusters).fit(fc2_weights_val)
          label_list.append(kmeans.labels_)
          cluster_centers_list.append(kmeans.cluster_centers_)
          distance_list.append(kmeans.inertia_)
          print("kmeans {} done".format(i))

      print("distance_list:{}".format(distance_list))
      index = distance_list.index(min(distance_list))
      print("index:{}".format(index))

      labels = label_list[index]
      cluster_centers = cluster_centers_list[index]

#    kmeans = KMeans(n_clusters=n_clusters, random_state=10).fit(wfc2_val)

      cluster_list = []
      for i in range (n_clusters):
          cluster_list.append([])
      for i in range (fc2_weights_val.shape[0]):
#        cluster_list[kmeans.labels_[i]].append(i)
          cluster_list[labels[i]].append(i)


      small_wfc1, small_bfc1, small_wfc2 = merge_matrix(fc1_weights_val, fc1_biases_val, fc2_weights_val, cluster_list, cluster_centers, n_clusters)


      wfc1_reset_array = make_reset_array(small_wfc1)
      wfc2_reset_array = make_reset_array(small_wfc2)
      wfc3_reset_array = make_reset_array(fc3_weights_val)


  size = column_num








  pruning_model_pr0 = tf.Graph()
  with pruning_model_pr0.as_default():
    train_data_node_pr0 = tf.placeholder(
      data_type(),
      shape=(BATCH_SIZE, IMAGE_SIZE_FOR_LENET, IMAGE_SIZE_FOR_LENET, NUM_CHANNELS), name="train_data_node_pr0")
    train_labels_node_pr0 = tf.placeholder(tf.int64, shape=(BATCH_SIZE,), name="train_labels_node_pr0")
    eval_data_pr0 = tf.placeholder(
      data_type(),
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE_FOR_LENET, IMAGE_SIZE_FOR_LENET, NUM_CHANNELS), name="eval_data_pr0")

#    a_val = tf.Variable(0)
#    b_val = tf.Variable(1)
#    addition = tf.add(a_val, b_val)
#    if retrain:
#       addition = tf.add(addition, 3)
#    addition = tf.add(addition, 3)


    # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.global_variables_initializer().run()}
    conv1_weights_pr0 = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, 6],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED, dtype=data_type()), name='conv1_weights_pr0', trainable=False)
    conv1_biases_pr0 = tf.Variable(tf.zeros([6], dtype=data_type()), name='conv1_biases_pr0', trainable=False)
    conv2_weights_pr0 = tf.Variable(tf.truncated_normal(
      [5, 5, 6, 16], stddev=0.1,
      seed=SEED, dtype=data_type()), name='conv2_weights_pr0', trainable=False)
    conv2_biases_pr0 = tf.Variable(tf.constant(0.1, shape=[16], dtype=data_type()), name='conv2_biases_pr0', trainable=False)
    fc1_weights_pr0 = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal([400, size],
                          stddev=0.1,
                          seed=SEED,
                          dtype=data_type()), name='fc1_weights_pr0', trainable=True)
  
#  fc1_weights_when_pruning = tf.Variable
#      tf.truncated_normal([400, num_neuron],
#                          stddev=0.1,
#                          seed=SEED,
#                          dtype=data_type()), name='fc1_weights_when_pruning')
  
    fc1_biases_pr0 = tf.Variable(tf.constant(0.1, shape=[size], dtype=data_type()), name='fc1_biases_pr0', trainable=True)
    fc2_weights_pr0 = tf.Variable(tf.truncated_normal([size, 84],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()), name='fc2_weights_pr0', trainable=True)
  
#  fc2_weights_when_pruning = tf.Variable
#      tf.truncated_normal([num_neuron, 84],
#                          stddev=0.1,
#                          seed=SEED,
#                          dtype=data_type()), name='fc2_weights_when_pruning')
    fc2_biases_pr0 = tf.Variable(tf.constant(
      0.1, shape=[84], dtype=data_type()), name='fc2_biases_pr0', trainable=True)
    fc3_weights_pr0 = tf.Variable(tf.truncated_normal([84, NUM_LABELS],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()), name='fc3_weights_pr0', trainable=True)
    fc3_biases_pr0 = tf.Variable(tf.constant(
      0.1, shape=[NUM_LABELS], dtype=data_type()), name='fc3_biases_pr0', trainable=True)



    masked_conv1_weights_pr0 = tf.placeholder(tf.float32, [5, 5, NUM_CHANNELS, 6], name="masked_conv1_weights_pr0")
    masked_conv1_biases_pr0 = tf.placeholder(tf.float32, [6], name="masked_conv1_biases_pr0")
    masked_conv2_weights_pr0 = tf.placeholder(tf.float32, [5, 5, 6, 16], name="masked_conv2_weights_pr0")
    masked_conv2_biases_pr0 = tf.placeholder(tf.float32, [16], name="masked_conv2_biases_pr0")
    masked_fc1_weights_pr0 = tf.placeholder(tf.float32, [400, size], name="masked_fc1_weights_pr0")
    masked_fc1_biases_pr0 = tf.placeholder(tf.float32, [size], name="masked_fc1_biases_pr0")
    masked_fc2_weights_pr0 = tf.placeholder(tf.float32, [size, 84], name="masked_fc2_weights_pr0")
    masked_fc2_biases_pr0 = tf.placeholder(tf.float32, [84], name="masked_fc2_biases_pr0")
    masked_fc3_weights_pr0 = tf.placeholder(tf.float32, [84, NUM_LABELS], name="masked_fc3_weights_pr0")
    masked_fc3_biases_pr0 = tf.placeholder(tf.float32, [NUM_LABELS], name="masked_fc3_biases_pr0")

    wfc1_reset_array_pr0 = tf.Variable(tf.truncated_normal([400,size], stddev=0.01), dtype=tf.float32, trainable=False, name="wfc1_reset_array_pr0")
    wfc2_reset_array_pr0 = tf.Variable(tf.truncated_normal([size,84], stddev=0.01), dtype=tf.float32, trainable=False, name="wfc2_reset_array_pr0")
    wfc3_reset_array_pr0 = tf.Variable(tf.truncated_normal([84,10], stddev=0.01), dtype=tf.float32, trainable=False, name="wfc3_reset_array_pr0")


    reset_masked_wfc1_pr0 = tf.placeholder(tf.float32, [400,size], name="reset_masked_wfc1_pr0")
    reset_masked_wfc2_pr0 = tf.placeholder(tf.float32, [size,84], name="reset_masked_wfc2_pr0")
    reset_masked_wfc3_pr0 = tf.placeholder(tf.float32, [84,10], name="reset_masked_wfc3_pr0")

    initial_wfc1_pr0 = tf.Variable(tf.truncated_normal([400,size], stddev=0.01), dtype=tf.float32, trainable=False, name="initial_wfc1_pr0")
    initial_bfc1_pr0 = tf.Variable(tf.truncated_normal([size], stddev=0.01), dtype=tf.float32, trainable=False, name="initial_bfc1_pr0")
    initial_wfc2_pr0 = tf.Variable(tf.truncated_normal([size,84], stddev=0.01), dtype=tf.float32, trainable=False, name="initial_wfc2_pr0")

    initial_wfc1_ph_pr0 = tf.placeholder(tf.float32, [400,size], name="initial_wfc1_ph_pr0")
    initial_bfc1_ph_pr0 = tf.placeholder(tf.float32, [size], name="initial_bfc1_ph_pr0")
    initial_wfc2_ph_pr0 = tf.placeholder(tf.float32, [size,84], name="initial_wfc2_ph_pr0")



    assign_masked_conv1_weights_pr0 = tf.assign(conv1_weights_pr0, masked_conv1_weights_pr0)
    assign_masked_conv1_biases_pr0 = tf.assign(conv1_biases_pr0, masked_conv1_biases_pr0)
    assign_masked_conv2_weights_pr0 = tf.assign(conv2_weights_pr0, masked_conv2_weights_pr0)
    assign_masked_conv2_biases_pr0 = tf.assign(conv2_biases_pr0, masked_conv2_biases_pr0)
    assign_masked_fc1_weights_pr0 = tf.assign(fc1_weights_pr0, masked_fc1_weights_pr0)
    assign_masked_fc1_biases_pr0 = tf.assign(fc1_biases_pr0, masked_fc1_biases_pr0)
    assign_masked_fc2_weights_pr0 = tf.assign(fc2_weights_pr0, masked_fc2_weights_pr0)
    assign_masked_fc2_biases_pr0 = tf.assign(fc2_biases_pr0, masked_fc2_biases_pr0)
    assign_masked_fc3_weights_pr0 = tf.assign(fc3_weights_pr0, masked_fc3_weights_pr0)
    assign_masked_fc3_biases_pr0 = tf.assign(fc3_biases_pr0, masked_fc3_biases_pr0)

    assign_reset_wfc1_pr0 = tf.assign(wfc1_reset_array_pr0, reset_masked_wfc1_pr0)
    assign_reset_wfc2_pr0 = tf.assign(wfc2_reset_array_pr0, reset_masked_wfc2_pr0)
    assign_reset_wfc3_pr0 = tf.assign(wfc3_reset_array_pr0, reset_masked_wfc3_pr0)

    assign_initial_wfc1_pr0 = tf.assign(initial_wfc1_pr0, initial_wfc1_ph_pr0)
    assign_initial_bfc1_pr0 = tf.assign(initial_bfc1_pr0, initial_bfc1_ph_pr0)
    assign_initial_wfc2_pr0 = tf.assign(initial_wfc2_pr0, initial_wfc2_ph_pr0)


    make_reset_wfc1_pr0 = tf.multiply(fc1_weights_pr0, wfc1_reset_array_pr0)
    reset_wfc1_pr0 = tf.assign(fc1_weights_pr0, make_reset_wfc1_pr0)

    make_reset_wfc2_pr0 = tf.multiply(fc2_weights_pr0, wfc2_reset_array_pr0)
    reset_wfc2_pr0 = tf.assign(fc2_weights_pr0, make_reset_wfc2_pr0)

    make_reset_wfc3_pr0 = tf.multiply(fc3_weights_pr0, wfc3_reset_array_pr0)
    reset_wfc3_pr0 = tf.assign(fc3_weights_pr0, make_reset_wfc3_pr0)
#    masked_fc1 = tf.placeholder(tf.float32, [400, 120], name="masked_fc1")

  
#  num_neuron = tf.placeholder(tf.float32, name='num_neuron')

#    if_pruning = tf.placeholder(dtype=tf.bool)

    # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
#    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
      conv_pr0 = tf.nn.conv2d(data,
                        conv1_weights_pr0,
                        strides=[1, 1, 1, 1],
                        padding='VALID')
    # Bias and rectified linear non-linearity.
      relu_pr0 = tf.nn.relu(tf.nn.bias_add(conv_pr0, conv1_biases_pr0))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
      pool_pr0 = tf.nn.max_pool(relu_pr0,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='VALID')
      conv_pr0 = tf.nn.conv2d(pool_pr0,
                        conv2_weights_pr0,
                        strides=[1, 1, 1, 1],
                        padding='VALID')
      relu_pr0 = tf.nn.relu(tf.nn.bias_add(conv_pr0, conv2_biases_pr0))
      pool_pr0 = tf.nn.max_pool(relu_pr0,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='VALID')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
      pool_shape_pr0 = pool_pr0.get_shape().as_list()
      reshape_pr0 = tf.reshape(
        pool_pr0,
        [pool_shape_pr0[0], pool_shape_pr0[1] * pool_shape_pr0[2] * pool_shape_pr0[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
      if train:
        reshape_pr0 = tf.nn.dropout(reshape_pr0, 0.5, seed=SEED)

#      if pruning:
      fc1_pr0 = tf.matmul(reshape_pr0, fc1_weights_pr0) + fc1_biases_pr0
#      else:
#        fc1_pr0 = tf.matmul(reshape, fc1_weights) + fc1_biases
#      if retrain==False:
#        fc1_pr0 = tf.nn.relu(fc1_pr0)

      if train:
        fc1_pr0 = tf.nn.dropout(fc1_pr0, 0.5, seed=SEED)

    #  if pruning:
      fc2_pr0 = tf.nn.relu(tf.matmul(fc1_pr0, fc2_weights_pr0) + fc2_biases_pr0)
    #  else:
    #    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
    
#    fc2 = tf.matmul(fc1, fc2_weights) + fc2_biases
#    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)

      if train:
        fc2_pr0 = tf.nn.dropout(fc2_pr0, 0.5, seed=SEED)

      return tf.matmul(fc2_pr0, fc3_weights_pr0) + fc3_biases_pr0

#    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
#    if train:
#      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
#    return tf.matmul(hidden, fc2_weights) + fc2_biases

  # Training computation: logits + cross-entropy loss.
#    def logits_before_pruning():
#      return model(train_data_node_pr0, True, False)
  
#    def logits_after_pruning():
#      return model(train_data_node_pr0, True, True)

    logits_pr0 = model(train_data_node_pr0, True)
#    logits_pr0 = tf.cond(if_pruning, logits_after_pruning, logits_before_pruning)
#  logits = model(train_data_node, True, False)
    loss_pr0 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=train_labels_node_pr0, logits=logits_pr0))

  # L2 regularization for the fully connected parameters.
    regularizers_pr0 = (tf.nn.l2_loss(fc1_weights_pr0) + tf.nn.l2_loss(fc1_biases_pr0) +
                  tf.nn.l2_loss(fc2_weights_pr0) + tf.nn.l2_loss(fc2_biases_pr0) +
                  tf.nn.l2_loss(fc3_weights_pr0) + tf.nn.l2_loss(fc3_biases_pr0))
  # Add the regularization term to the loss.
    loss_pr0 += 5e-4 * regularizers_pr0

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
    batch_pr0 = tf.Variable(0, dtype=data_type())
    step_reset = tf.assign(batch_pr0, 0)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate_pr0 = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch_pr0 * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
    optimizer_pr0 = tf.train.MomentumOptimizer(learning_rate_pr0,
                                         0.9).minimize(loss_pr0,
                                                       global_step=batch_pr0, name="optimizer_pr0")

  # Predictions for the current training minibatch.
    train_prediction_pr0 = tf.nn.softmax(logits_pr0)

  # Predictions for the test and validation, which we'll compute less often.
    eval_prediction_pr0 = tf.nn.softmax(model(eval_data_pr0), name="eval_prediction")

    correct_train_pr0 = tf.equal(tf.argmax(train_prediction_pr0, 1), train_labels_node_pr0)

    accuracy_train_pr0 = tf.reduce_mean(tf.cast(correct_train_pr0, tf.float32))

    predictions_pr0 = tf.placeholder(tf.float32, [None, NUM_LABELS], name="predictions")

    correct_eval_pr0 = tf.equal(tf.argmax(predictions_pr0, 1), validation_labels)

    accuracy_eval_pr0 = tf.reduce_mean(tf.cast(correct_eval_pr0, tf.float32))

    correct_test_pr0 = tf.equal(tf.argmax(predictions_pr0, 1), test_labels)

    accuracy_test_pr0 = tf.reduce_mean(tf.cast(correct_test_pr0, tf.float32), name="accuracy_test")


    saver_pr0 = tf.train.Saver()

#    assign_mask = tf.assign(fc1_weights, masked_fc1, name="assign_mask")


    # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(data, sess):
#    """Get all predictions for a dataset by running it in small batches."""
      size = data.shape[0]
      if size < EVAL_BATCH_SIZE:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
      predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
      for begin in xrange(0, size, EVAL_BATCH_SIZE):
        end = begin + EVAL_BATCH_SIZE
        if end <= size:
          predictions[begin:end, :] = sess.run(
            eval_prediction_pr0,
            feed_dict={eval_data_pr0: data[begin:end, ...]})
        else:
          batch_predictions = sess.run(
            eval_prediction_pr0,
            feed_dict={eval_data_pr0: data[-EVAL_BATCH_SIZE:, ...]})
          predictions[begin:, :] = batch_predictions[begin - size:, :]
      return predictions



  with tf.Session(graph=pruning_model_pr0) as sess:
    if analysis:
        saver_pr0.restore(sess, os.path.join(kmeans_path_pr0, 'lenet-5.ckpt'))

        wcnn1_after = sess.run(conv1_weights_pr0)
        bcnn1_after = sess.run(conv1_biases_pr0)
        wcnn2_after = sess.run(conv2_weights_pr0)
        bcnn2_after = sess.run(conv2_biases_pr0)
        wfc1_after = sess.run(fc1_weights_pr0)
        bfc1_after = sess.run(fc1_biases_pr0)
        wfc2_after = sess.run(fc2_weights_pr0)
        bfc2_after = sess.run(fc2_biases_pr0)
        wfc3_after = sess.run(fc3_weights_pr0)
        bfc3_after = sess.run(fc3_biases_pr0)

        shape_list_after = []
        non_zero_element_after = []
        memory_size_after = []

        shape_list_after.append(wcnn1_after.shape)
        shape_list_after.append(bcnn1_after.shape)
        shape_list_after.append(wcnn2_after.shape)
        shape_list_after.append(bcnn2_after.shape)
        shape_list_after.append(wfc1_after.shape)
        shape_list_after.append(bfc1_after.shape)
        shape_list_after.append(wfc2_after.shape)
        shape_list_after.append(bfc2_after.shape)
        shape_list_after.append(wfc3_after.shape)
        shape_list_after.append(bfc3_after.shape)

        non_zero_element_after.append(wcnn1_after.size)
        non_zero_element_after.append(bcnn1_after.size)
        non_zero_element_after.append(wcnn2_after.size)
        non_zero_element_after.append(bcnn2_after.size)
        non_zero_element_after.append(np.count_nonzero(wfc1_after))
        non_zero_element_after.append(bfc1_after.size)
        non_zero_element_after.append(np.count_nonzero(wfc2_after))
        non_zero_element_after.append(bfc2_after.size)
        non_zero_element_after.append(np.count_nonzero(wfc3_after))
        non_zero_element_after.append(bfc3_after.size)

        memory_size_after.append(wcnn1_after.size*4)
        memory_size_after.append(bcnn1_after.size*4)
        memory_size_after.append(wcnn2_after.size*4)
        memory_size_after.append(bcnn2_after.size*4)
        memory_size_after.append(min(2*np.count_nonzero(wfc1_after)+wfc1_after.shape[1]+1, wfc1_after.size)*4)
        memory_size_after.append(bfc1_after.size*4)
        memory_size_after.append(min(2*np.count_nonzero(wfc2_after)+wfc2_after.shape[0]+1, 2*np.count_nonzero(wfc2_after)+wfc2_after.shape[1]+1, wfc2_after.size)*4)
        memory_size_after.append(bfc2_after.size*4)
        memory_size_after.append(min(2*np.count_nonzero(wfc3_after)+wfc3_after.shape[1]+1, wfc3_after.size)*4)
        memory_size_after.append(bfc3_after.size*4)


        cnn_memory_size_after = 0
        for i in range(4):
            cnn_memory_size_after += memory_size_after[i]

        fc_memory_size_after = 0
        for i in range(4, 10):
            fc_memory_size_after += memory_size_after[i]

        total_memory_size_after=0
        for i in range(len(memory_size_after)):
            total_memory_size_after += memory_size_after[i]


        wfc1_bfc1 = np.vstack((wfc1_after, bfc1_after))
        for i in range(wfc1_bfc1.shape[1]):
            if wfc1_bfc1[wfc1_bfc1.shape[0]-1][i] != bfc1_after[i]:
                print("error")
        wfc1_bfc1_wfc2 = np.dot(wfc1_bfc1, wfc2_after)

        shape_list_after.append(wfc1_bfc1_wfc2.shape)
        non_zero_element_after.append(np.count_nonzero(wfc1_bfc1_wfc2))
        memory_size_after.append(min(2*np.count_nonzero(wfc1_bfc1_wfc2)+wfc1_bfc1_wfc2.shape[1]+1, wfc1_bfc1_wfc2.size)*4)

        fc_memory_size_after_merge = memory_size_after[7]+memory_size_after[8]+memory_size_after[9]+memory_size_after[10]
        add_one = 8
        total_memory_size_after_merge = cnn_memory_size_after+add_one+fc_memory_size_after_merge

        print("")
        print("after")
        print("shape")
        print("wfc1:{}, bfc1:{}, wfc2:{}, bfc2:{}, wfc3:{}, bfc3:{}".format(shape_list_after[4], shape_list_after[5], shape_list_after[6], shape_list_after[7], shape_list_after[8], shape_list_after[9]))
        print("the number of non zero element")
        print("wfc1:{}, bfc1:{}, wfc2:{}, bfc2:{}, wfc3:{}, bfc3:{}".format(non_zero_element_after[4], non_zero_element_after[5], non_zero_element_after[6], non_zero_element_after[7], non_zero_element_after[8], non_zero_element_after[9]))
        print("memory size[MB]")
        print("wfc1:{}, bfc1:{}, wfc2:{}, bfc2:{}, wfc3:{}, bfc3:{}".format(memory_size_after[4]/float(1000000), memory_size_after[5]/float(1000000), memory_size_after[6]/float(1000000), memory_size_after[7]/float(1000000), memory_size_after[8]/float(1000000), memory_size_after[9]/float(1000000)))
        print("cnn memory size[MB]")
        print(cnn_memory_size_after/float(1000000))
        print("fc memory size[MB]")
        print(fc_memory_size_after/float(1000000))
        print("total memory size[MB]")
        print(total_memory_size_after/float(1000000))
        print("")

        print("merge wfc1 bfc1 wfc2")
        print("shape")
        print("wfc1_bfc1_wfc2:{}".format(shape_list_after[10]))
        print("the number of non zero element")
        print("wfc1_bfc1_wfc2:{}".format(non_zero_element_after[10]))
        print("memory size[MB]")
        print("wfc1_bfc1_wfc2:{}".format(memory_size_after[10]/float(1000000)))
        print("cnn memory size[MB]")
        print(cnn_memory_size_after/float(1000000))
        print("fc memory size[MB]")
        print(fc_memory_size_after_merge/float(1000000))
        print("total memory size[MB]")
        print(total_memory_size_after_merge/float(1000000))

        sys.exit()

    else:
       
        tf.global_variables_initializer().run()

        sess.run(assign_masked_conv1_weights_pr0, feed_dict={masked_conv1_weights_pr0 : conv1_weights_val})
        sess.run(assign_masked_conv1_biases_pr0, feed_dict={masked_conv1_biases_pr0 : conv1_biases_val})
        sess.run(assign_masked_conv2_weights_pr0, feed_dict={masked_conv2_weights_pr0 : conv2_weights_val})
        sess.run(assign_masked_conv2_biases_pr0, feed_dict={masked_conv2_biases_pr0 : conv2_biases_val})
        sess.run(assign_masked_fc1_weights_pr0, feed_dict={masked_fc1_weights_pr0 : small_wfc1})
        sess.run(assign_masked_fc1_biases_pr0, feed_dict={masked_fc1_biases_pr0 : small_bfc1})
        sess.run(assign_masked_fc2_weights_pr0, feed_dict={masked_fc2_weights_pr0 : small_wfc2})
        sess.run(assign_masked_fc2_biases_pr0, feed_dict={masked_fc2_biases_pr0 : fc2_biases_val})
        sess.run(assign_masked_fc3_weights_pr0, feed_dict={masked_fc3_weights_pr0 : fc3_weights_val})
        sess.run(assign_masked_fc3_biases_pr0, feed_dict={masked_fc3_biases_pr0 : fc3_biases_val})

        sess.run(assign_reset_wfc1_pr0, feed_dict = {reset_masked_wfc1_pr0 : wfc1_reset_array})
        sess.run(assign_reset_wfc2_pr0, feed_dict = {reset_masked_wfc2_pr0 : wfc2_reset_array})
        sess.run(assign_reset_wfc3_pr0, feed_dict = {reset_masked_wfc3_pr0 : wfc3_reset_array})

        sess.run(assign_initial_wfc1_pr0, feed_dict = {initial_wfc1_ph_pr0 : small_wfc1})
        sess.run(assign_initial_bfc1_pr0, feed_dict = {initial_bfc1_ph_pr0 : small_bfc1})
        sess.run(assign_initial_wfc2_pr0, feed_dict = {initial_wfc2_ph_pr0 : small_wfc2})




#    print('addition=%d' % sess.run(addition))
    start_time = time.time()
    for step in range(int(num_epochs * train_size) // BATCH_SIZE):
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]


        feed_dict = {train_data_node_pr0: batch_data,
                   train_labels_node_pr0: batch_labels}

        sess.run(optimizer_pr0, feed_dict=feed_dict)

        sess.run(reset_wfc1_pr0)
        sess.run(reset_wfc2_pr0)
        sess.run(reset_wfc3_pr0)

        if step % EVAL_FREQUENCY == 0:
        # fetch some extra nodes' data
            l, lr = sess.run([loss_pr0, learning_rate_pr0],
                                      feed_dict={train_data_node_pr0: batch_data,
                   train_labels_node_pr0: batch_labels})

            accuracy_train_percent = sess.run(accuracy_train_pr0, feed_dict={train_data_node_pr0: batch_data,
                   train_labels_node_pr0: batch_labels})*100
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
            print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
            print('Minibatch accuracy: %.3f%%' % accuracy_train_percent)
#        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))


            prediction_eval = eval_in_batches(validation_data, sess)
#        prediction = prediction.tolist()
            accuracy_eval_percent = sess.run(accuracy_eval_pr0, feed_dict={predictions_pr0:prediction_eval})*100
            print('Validation accuracy %.3f%%' % accuracy_eval_percent)

            sys.stdout.flush()


#    print("conv1_weights_shape" + str(sess.run(conv1_weights_pr0).shape))
#    print("conv1_biases_shape" + str(sess.run(conv1_biases_pr0).shape))
#    print("conv2_weights_shape" + str(sess.run(conv2_weights_pr0).shape))
#    print("conv2_biases_shape" + str(sess.run(conv2_biases_pr0).shape))
#    print("fc1_shape" + str(sess.run(fc1_weights_pr0).shape))
#    print("fc1_biases_shape" + str(sess.run(fc1_biases_pr0).shape))
#    print("fc2_shape" + str(sess.run(fc2_weights_pr0).shape))
#    print("fc2_biases_shape" + str(sess.run(fc2_biases_pr0).shape))
#    print("fc3_shape" + str(sess.run(fc3_weights_pr0).shape))
#    print("fc3_biases_shape" + str(sess.run(fc3_biases_pr0).shape))

    timer_start=time.time()
    prediction_test = eval_in_batches(test_data, sess)
    accuracy_test_percent = sess.run(accuracy_test_pr0, feed_dict={predictions_pr0:prediction_test})*100
    timer_stop=time.time()-timer_start
#    print("pruning ratio:" + str(PRUNING_RATIO_REAL))
    print('test accuracy %.8f%%' % accuracy_test_percent)
    print('evaluation time:' + str(1000*timer_stop) + 'ms')


    save_path = saver_pr0.save(sess, os.path.join(kmeans_path_pr0, 'lenet-5.ckpt'))
    print('Variables saved in file: %s' % save_path)




    







 


    
"""
    ckpt_state = tf.train.get_checkpoint_state('ckpt/4/')
    if ckpt_state:
      last_model = ckpt_state.model_checkpoint_path
      saver.restore(sess, 'ckpt/4/' + last_model)
      print("model was loaded:", last_model)
    else:
      sess.run(init)
      print("initialized")

"""

#    test_error = original.error_rate(original.eval_in_batches(test_data, sess), test_labels)
#    print('Test error: %.3f%%' % test_error)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_fp16',
      default=False,
      help='Use half floats instead of full floats if True.',
      action='store_true')
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.')

  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
