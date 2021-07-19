
import sys
import os.path
import time
#from models import alexnet_nonrelu_un0
import tensorflow as tf
import util_pr0 as tu
import numpy as np
import threading
import random
from sklearn.cluster import KMeans
from scipy.linalg import norm
import copy
import csv

FC1_LEN=37748736
FC1_ROW=9216
FC1_COL=4096
PRUNING_RATIO=0.99
NUM_OF_PRUNING_STEP=1
ORIGINAL_DROPOUT_RATIO = 0.5
#REMAINED_COLUMN = math.floor(48000*(1-PRUNING_RATIO)/400)
#PRUNING_RATIO_REAL = (48000-400*REMAINED_COLUMN)/48000
#PRUNING_RATIO_PER_ITERATION=PRUNING_RATIO_REAL/NUM_OF_PRUNING_STEP


def pruning_number_list(total_len, row, unstructured):
    pruning_num_list = []
    delete_element = total_len* PRUNING_RATIO
    column_num = delete_element // row
    delete_element_true = row * column_num
    pruning_ratio_true = (delete_element_true / total_len) * 100

    if unstructured:
        pruning_number_regular = delete_element_true // NUM_OF_PRUNING_STEP
        pruning_number_last = pruning_number_regular + (delete_element_true % NUM_OF_PRUNING_STEP)
        for i in range(NUM_OF_PRUNING_STEP):
            if i != (NUM_OF_PRUNING_STEP-1):
                pruning_num_list.append((i+1)*pruning_number_regular)
            else:
                pruning_num_list.append(delete_element_true)
#        pruning_num_list.append(pruning_number_regular)
#        pruning_num_list.append(pruning_number_last)
    else:
        column_num_regular = column_num // NUM_OF_PRUNING_STEP
        column_num_last = column_num_regular + (column_num % NUM_OF_PRUNING_STEP)
        pruning_number_regular = row * column_num_regular
        pruning_number_last = row * column_num_last
        for i in range(NUM_OF_PRUNING_STEP):
            if i != (NUM_OF_PRUNING_STEP-1):
                pruning_num_list.append((i+1)*column_num_regular)
            else:
                pruning_num_list.append(column_num)
#        pruning_num_list.append(pruning_number_regular)
#        pruning_num_list.append(pruning_number_last)
    
    return pruning_num_list, pruning_ratio_true





def get_min_order_index(w, axis_1):
	original_index_list=[]
	abs_w=np.abs(w)
	reshape=abs_w.reshape(-1)
	index_list=np.argsort(reshape)
	for i in range(len(index_list)):
		row=index_list[i]//axis_1
		col=index_list[i]%axis_1
		original_index_list.append([row,col])
	return original_index_list


def prune_weights(num_of_pruning, target_array, min_order_index_list):
#	num_of_pruning=math.floor(len(min_order_index_list)*PRUNING_RATIO_PER_ITERATION*pruning_step)
	for i in range(num_of_pruning):
		row=min_order_index_list[i][0]
		col=min_order_index_list[i][1]
		target_array[row][col]=0.0
	return target_array


#def make_reset_array(num_of_pruning, target_row, target_col, min_order_index_list):
#    target_array = np.ones((target_row, target_col), dtype = np.float32)
#    for i in range(num_of_pruning):
#        row=min_order_index_list[i][0]
#        col=min_order_index_list[i][1]
#        target_array[row][col]=0.0
#    return target_array


def make_reset_array(weight_array):
    reset_array = np.ones((weight_array.shape[0], weight_array.shape[1]), dtype = np.float32)
    for i in range(weight_array.shape[0]):
        for j in range(weight_array.shape[1]):
            if weight_array[i][j] == 0.0:
               reset_array[i][j] = 0.0

    return reset_array


def calculate_l1_norm_of_each_neuron(target_array):
    value_list = []
    abs_w = np.abs(target_array)
    for i in range (target_array.shape[0]):
        l1_norm = 0
        for j in range (target_array.shape[1]):
            l1_norm += abs_w[i][j]
        value_list.append(l1_norm)
    value_array = np.array(value_list)
    index_array = np.argsort(value_array)
    return index_array


def shrink_weight_array(num_of_pruning, target_array, index_array, axis):
    delete_index_list = []
    for i in range (num_of_pruning):
        delete_index_list.append(index_array[i])
    shrinked_array = np.delete(target_array, delete_index_list, axis)
    return shrinked_array


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
#    print(N_CLUSTERS)
#    print(fc2_weights.shape)
#    print(fc1_weights.shape)

#    count=0
#    for i in range (N_CLUSTERS):
#        if len(cluster_list[i]) == 0:
#            count+=1
#    print(count)

    for i in range(pruned_weights.shape[0]):
        for j in range (pruned_weights.shape[1]):
            pruned_weights_copy[i][j] = pruned_weights[i][j]

    for i in range(next_weights.shape[0]):
        for j in range (next_weights.shape[1]):
            next_weights_copy[i][j] = next_weights[i][j]
    with open('pcpcluster_list.csv', 'w') as f:
        writer = csv.writer(f)
        for i in range(len(cluster_list)):
            writer.writerow(cluster_list[i])



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


    for i in range (len(cluster_list)):
        for j in range(1,len(cluster_list[i])):
            next_weights[cluster_list[i][j]]=copy.deepcopy(next_weights_copy[cluster_list[i][0]])
            
    for i in range (n_clusters):
        if len(cluster_list[i]) == 1:
            continue
        else:
            for j in range (1, len(cluster_list[i])):
                delete_list.append(cluster_list[i][j])
    
    next_weights_del = np.delete(next_weights, delete_list, 0)
    np.savetxt('pcpnext.csv',delete_list,delimiter=',')
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
                    pruned_weights[k][cluster_list[i][0]] += pruned_weights_copy[k][cluster_list[i][j]]
                pruned_biases[cluster_list[i][0]] += pruned_biases[cluster_list[i][j]]
    np.savetxt('pcppcp_out.csv',pruned_weights,delimiter=',')

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

#    print("merge analysis")
#    print("cluster index, # of element, element, wfc1 before, wfc1 after, wfc2 before, wfc2 after")
#    for i in range(n_clusters):
#        print('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(i, len(cluster_list[i]), reduction_col[i], cluster_list[i], wfc1_non_zero_element_before_merge[i], wfc1_non_zero_element_after_merge[i], reduction_nz_wfc1[i], wfc2_non_zero_element_before_merge[i], wfc2_non_zero_element_after_merge[i], reduction_nz_wfc2[i], slope_wfc1[i], slope_wfc2[i]))

#    sys.exit()

    pruned_weights_del = np.delete(pruned_weights, delete_list, 1)
    pruned_biases_del = np.delete(pruned_biases, delete_list, None)

    return pruned_weights_del, pruned_biases_del, next_weights_del




def proposed_scheme(
threads_numbers,
		epochs, 
		batch_size, 
		learning_rate, 
		dropout, 
		lmbda,
		momentum, 
		analysis,
		resume, 
        test,
        top_k,
		column_num,
		imagenet_path, 
		display_step, 
		test_step, 
		ckpt_path_un0,
        ckpt_path_un1, 
		summary_path,
		save_path_pr0,
        save_path_un1):

#    pruning_stage = -1
    train_img_path = os.path.join(imagenet_path, 'ILSVRC2012_img_train')
    ts_size = tu.imagenet_size(train_img_path)
    print ("ts_size : {}".format(ts_size))
    num_batches = int(float(ts_size) / batch_size)
#    num_batches = 20

    wnid_labels, _ = tu.load_imagenet_meta(os.path.join(imagenet_path, 'ILSVRC2012_devkit_t12/ILSVRC2012_devkit_t12/data/meta.mat'))
#    pruning_num_list, pruning_ratio_true = pruning_number_list(FC1_LEN, FC1_ROW, False)


#    mask = tf.placeholder(tf.float32, [9216, 4096], name="masked_fc1")
#    assign_mask = tf.assign(wfc1, mask)

#    global_step = tf.Variable(0, trainable=False)
#    epoch = tf.div(global_step, num_batches)
    
#    saver = tf.train.Saver()
#    coord = tf.train.Coordinator()
#    init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto()) as sess:
#        sess.run(init)
#        saver.restore(sess, os.path.join(ckpt_path, 'alexnet-cnn.ckpt'))
        saver = tf.train.import_meta_graph(os.path.join(ckpt_path_un0, 'alexnet-cnn.ckpt.meta'), clear_devices=True)
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path_un0))

        graph = tf.get_default_graph()

#        for op in graph.get_operations():
#            print(op.name)

#        graph = tf.get_default_graph()
        wcnn1 = graph.get_tensor_by_name("alexnetwork_un0/conv1_un0/wcnn1_un0:0")
        bcnn1 = graph.get_tensor_by_name("alexnetwork_un0/conv1_un0/bcnn1_un0:0")
        wcnn2 = graph.get_tensor_by_name("alexnetwork_un0/conv2_un0/wcnn2_un0:0")
        bcnn2 = graph.get_tensor_by_name("alexnetwork_un0/conv2_un0/bcnn2_un0:0")
        wcnn3 = graph.get_tensor_by_name("alexnetwork_un0/conv3_un0/wcnn3_un0:0")
        bcnn3 = graph.get_tensor_by_name("alexnetwork_un0/conv3_un0/bcnn3_un0:0")
        wcnn4 = graph.get_tensor_by_name("alexnetwork_un0/conv4_un0/wcnn4_un0:0")
        bcnn4 = graph.get_tensor_by_name("alexnetwork_un0/conv4_un0/bcnn4_un0:0")
        wcnn5 = graph.get_tensor_by_name("alexnetwork_un0/conv5_un0/wcnn5_un0:0")
        bcnn5 = graph.get_tensor_by_name("alexnetwork_un0/conv5_un0/bcnn5_un0:0")
        wfc1 = graph.get_tensor_by_name("classifier_un0/fullyconected1_un0/wfc1_un0:0")
        bfc1 = graph.get_tensor_by_name("classifier_un0/fullyconected1_un0/bfc1_un0:0")
        wfc2 = graph.get_tensor_by_name("classifier_un0/fullyconected2_un0/wfc2_un0:0")
        bfc2 = graph.get_tensor_by_name("classifier_un0/fullyconected2_un0/bfc2_un0:0")
        wfc3 = graph.get_tensor_by_name("classifier_un0/classifier_output_un0/wfc3_un0:0")
        bfc3 = graph.get_tensor_by_name("classifier_un0/classifier_output_un0/bfc3_un0:0")
#        enqueue_op = graph.get_operation_by_name("enqueue_op")
#        x = graph.get_tensor_by_name("x:0")
#        y = graph.get_tensor_by_name("y:0")
#        lr = graph.get_tensor_by_name("lr:0")
#        keep_prob = graph.get_tensor_by_name("keep_prob:0")
#        optimizer = graph.get_operation_by_name("optimizer/optimizer")
#        loss = graph.get_operation_by_name("loss/loss")
#        accuracy = graph.get_operation_by_name("accuracy/accuracy")

        wcnn1_val = sess.run(wcnn1)
        bcnn1_val = sess.run(bcnn1)
        wcnn2_val = sess.run(wcnn2)
        bcnn2_val = sess.run(bcnn2)
        wcnn3_val = sess.run(wcnn3)
        bcnn3_val = sess.run(bcnn3)
        wcnn4_val = sess.run(wcnn4)
        bcnn4_val = sess.run(bcnn4)
        wcnn5_val = sess.run(wcnn5)
        bcnn5_val = sess.run(bcnn5)
        wfc1_val = sess.run(wfc1)
        bfc1_val = sess.run(bfc1)
        wfc2_val = sess.run(wfc2)
        bfc2_val = sess.run(bfc2)
        wfc3_val = sess.run(wfc3)
        bfc3_val = sess.run(bfc3)



    if analysis == True:
       shape_list = []
       non_zero_element = []
       memory_size = []

       shape_list.append(wcnn1_val.shape)
       shape_list.append(bcnn1_val.shape)
       shape_list.append(wcnn2_val.shape)
       shape_list.append(bcnn2_val.shape)
       shape_list.append(wcnn3_val.shape)
       shape_list.append(bcnn3_val.shape)
       shape_list.append(wcnn4_val.shape)
       shape_list.append(bcnn4_val.shape)
       shape_list.append(wcnn5_val.shape)
       shape_list.append(bcnn5_val.shape)
       shape_list.append(wfc1_val.shape)
       shape_list.append(bfc1_val.shape)
       shape_list.append(wfc2_val.shape)
       shape_list.append(bfc2_val.shape)
       shape_list.append(wfc3_val.shape)
       shape_list.append(bfc3_val.shape)
       

       
       non_zero_element.append(wcnn1_val.size)
       non_zero_element.append(bcnn1_val.size)
       non_zero_element.append(wcnn2_val.size)
       non_zero_element.append(bcnn2_val.size)
       non_zero_element.append(wcnn3_val.size)
       non_zero_element.append(bcnn3_val.size)
       non_zero_element.append(wcnn4_val.size)
       non_zero_element.append(bcnn4_val.size)
       non_zero_element.append(wcnn5_val.size)
       non_zero_element.append(bcnn5_val.size)
       non_zero_element.append(np.count_nonzero(wfc1_val))
       non_zero_element.append(bfc1_val.size)
       non_zero_element.append(np.count_nonzero(wfc2_val))
       non_zero_element.append(bfc2_val.size)
       non_zero_element.append(np.count_nonzero(wfc3_val))
       non_zero_element.append(bfc3_val.size)


       memory_size.append(wcnn1_val.size*4)
       memory_size.append(bcnn1_val.size*4)
       memory_size.append(wcnn2_val.size*4)
       memory_size.append(bcnn2_val.size*4)
       memory_size.append(wcnn3_val.size*4)
       memory_size.append(bcnn3_val.size*4)
       memory_size.append(wcnn4_val.size*4)
       memory_size.append(bcnn4_val.size*4)
       memory_size.append(wcnn5_val.size*4)
       memory_size.append(bcnn5_val.size*4)
       memory_size.append(min(2*np.count_nonzero(wfc1_val)+wfc1_val.shape[1]+1, wfc1_val.size)*4)
       memory_size.append(bfc1_val.size*4)
       memory_size.append(min(2*np.count_nonzero(wfc2_val)+wfc2_val.shape[1]+1, wfc2_val.size)*4)
       memory_size.append(bfc2_val.size*4)
       memory_size.append(min(2*np.count_nonzero(wfc3_val)+wfc3_val.shape[1]+1, wfc3_val.size)*4)
       memory_size.append(bfc3_val.size*4)

       cnn_memory_size = 0
       for i in range(10):
           cnn_memory_size += memory_size[i]

       fc_memory_size = 0
       for i in range(10, 16):
           fc_memory_size += memory_size[i]

       total_memory_size=0
       for i in range(len(memory_size)):
           total_memory_size += memory_size[i]

#       count=0
#       for i in range(wfc1_val.shape[0]):
#           for j in range(wfc1_val.shape[1]):
#               if wfc1_val[i][j] != 0.0:
#                   count += 1

       print("before")
       print("shape")
       print("wfc1:{}, bfc1:{}, wfc2:{}, bfc2:{}, wfc3:{}, bfc3:{}".format(shape_list[10], shape_list[11], shape_list[12], shape_list[13], shape_list[14], shape_list[15]))
       print("the number of non zero element")
       print("wfc1:{}, bfc1:{}, wfc2:{}, bfc2:{}, wfc3:{}, bfc3:{}".format(non_zero_element[10], non_zero_element[11], non_zero_element[12], non_zero_element[13], non_zero_element[14], non_zero_element[15]))
       print("memory size[MB]")
       print("wswswfc1:{}, ddddbfc1:{}, wfc2:{}, bfc2:{}, wfc3:{}, bfc3:{}".format(memory_size[10]/float(1000000), memory_size[11]/float(1000000), memory_size[12]/float(1000000), memory_size[13]/float(1000000), memory_size[14]/float(1000000), memory_size[15]/float(1000000)))
       print("cnn memory size[MB]")
       print(cnn_memory_size/float(1000000))
       print("fc memory size[MB]")
       print(fc_memory_size/float(1000000))
       print("total memory size[MB]")
       print(total_memory_size/float(1000000))



#    pruning_stage += 1
#    n_clusters = FC1_COL - int(pruning_num_list[pruning_stage])

    if ((resume == False) and (analysis == False) and (test == False)):
        print("exit")
        n_clusters = column_num
        column_num_reduction_ratio = (float(wfc1_val.shape[1]-column_num) / wfc1_val.shape[1])*100

        label_list = []
        cluster_centers_list = []
        distance_list = []
                        
        for i in range(0,wfc1_val.shape[1]):
            wfc1_val[:,i]=wfc1_val[:,i]*(float(np.linalg.norm(wfc2_val[i],ord=2)))

#        np.savetxt('pcpbef_fc2_out.csv',wfc2_val,delimiter=',')

        for i in range (0,int(wfc2_val.shape[0])):
            wfc2_val[i]=wfc2_val[i]/(float(np.linalg.norm(wfc2_val[i],ord=2)))

        np.savetxt('pcpaft_fc2_out.csv',wfc2_val,delimiter=',')
            
        for i in range(5):
            kmeans = KMeans(n_clusters=n_clusters).fit(wfc2_val)
            label_list.append(kmeans.labels_)
            cluster_centers_list.append(kmeans.cluster_centers_)
            distance_list.append(kmeans.inertia_)
            print("kmeans##### {} done".format(i))

        print("distance_list:{}".format(distance_list))
        index = distance_list.index(min(distance_list))
        print("index:{}".format(index))

        labels = label_list[index]
        cluster_centers = cluster_centers_list[index] 
        print(cluster_centers)

        
#    kmeans = KMeans(n_clusters=n_clusters, random_state=10).fit(wfc2_val)

        cluster_list = []
        for i in range (n_clusters):
            cluster_list.append([])
        for i in range (wfc2_val.shape[0]):
#        cluster_list[kmeans.labels_[i]].append(i)
            cluster_list[labels[i]].append(i)

#
#        np.savetxt('label.csv',label_list)
#        print("-----------------------------------")
#        print(cluster_list)
#        print(cluster_centers)
#         print("-----------------------------------")
#        with open('pcpcluster_list.csv', 'w') as f:
#            writer = csv.writer(f)
#            for i in range(len(cluster_list)):
#                writer.writerow(cluster_list[i])

            
        


        small_wfc1, small_bfc1, small_wfc2 = merge_matrix(wfc1_val, bfc1_val,wfc2_val, cluster_list, cluster_centers, n_clusters)

        np.savetxt('pcpclus_fc2_out.csv',small_wfc2,delimiter=',')

        np.savetxt('pcpcenter_fc2_out.csv',cluster_centers,delimiter=',')
#        np.savetxt('pcp_n_fc2_out.csv',n_clusters,delimiter=',')
        wfc1_reset_array = make_reset_array(small_wfc1)
        wfc2_reset_array = make_reset_array(small_wfc2)
        wfc3_reset_array = make_reset_array(wfc3_val)

        np.savetxt('pcpreset_fc2_out.csv',wfc2_reset_array,delimiter=',')

#    saving = tf.train.Saver()
#    save_path = saving.save(sess, os.path.join(save_path_pr0, 'alexnet-cnn.ckpt'))
#    print('Variables saved in file: %s' % save_path)
    size = column_num



    pruning_model_pr0 = tf.Graph()
    with pruning_model_pr0.as_default():

        with tf.device('/gpu:0'):
            x_pr0 = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x_pr0')
            y_pr0 = tf.placeholder(tf.float32, [None, 1000], name='y_pr0')
            lr_pr0 = tf.placeholder(tf.float32, name='lr_pr0')
            keep_prob_pr0 = tf.placeholder(tf.float32, name='keep_prob_pr0')

            data_pr0 = tf.placeholder(tf.float32, [None, 224, 224, 3], name="data_pr0")
            label_pr0 = tf.placeholder(tf.float32, [None, 1000], name="label_pr0")

            masked_wcnn1_pr0 = tf.placeholder(tf.float32, [11,11,3,96], name="masked_wcnn1_pr0")
            masked_bcnn1_pr0 = tf.placeholder(tf.float32, [96], name="masked_bcnn1_pr0")
            masked_wcnn2_pr0 = tf.placeholder(tf.float32, [5,5,96,256], name="masked_wcnn2_pr0")
            masked_bcnn2_pr0 = tf.placeholder(tf.float32, [256], name="masked_bcnn2_pr0")
            masked_wcnn3_pr0 = tf.placeholder(tf.float32, [3,3,256,384], name="masked_wcnn3_pr0")
            masked_bcnn3_pr0 = tf.placeholder(tf.float32, [384], name="masked_bcnn3_pr0")
            masked_wcnn4_pr0 = tf.placeholder(tf.float32, [3,3,384,384], name="masked_wcnn4_pr0")
            masked_bcnn4_pr0 = tf.placeholder(tf.float32, [384], name="masked_bcnn4_pr0")
            masked_wcnn5_pr0 = tf.placeholder(tf.float32, [3,3,384,256], name="masked_wcnn5_pr0")
            masked_bcnn5_pr0 = tf.placeholder(tf.float32, [256], name="masked_bcnn5_pr0")
            masked_wfc1_pr0 = tf.placeholder(tf.float32, [6400,size], name="masked_wfc1_pr0")
            masked_bfc1_pr0 = tf.placeholder(tf.float32, [size], name="masked_bfc1_pr0")
            masked_wfc2_pr0 = tf.placeholder(tf.float32, [size,4096], name="masked_wfc2_pr0")
            masked_bfc2_pr0 = tf.placeholder(tf.float32, [4096], name="masked_bfc2_pr0")
            masked_wfc3_pr0 = tf.placeholder(tf.float32, [4096,1000], name="masked_wfc3_pr0")
            masked_bfc3_pr0 = tf.placeholder(tf.float32, [1000], name="masked_bfc3_pr0")

            wfc1_reset_array_pr0 = tf.Variable(tf.truncated_normal([6400,size], stddev=0.01), dtype=tf.float32, trainable=False, name="wfc1_reset_array_pr0")
            wfc2_reset_array_pr0 = tf.Variable(tf.truncated_normal([size,4096], stddev=0.01), dtype=tf.float32, trainable=False, name="wfc2_reset_array_pr0")
            wfc3_reset_array_pr0 = tf.Variable(tf.truncated_normal([4096,1000], stddev=0.01), dtype=tf.float32, trainable=False, name="wfc3_reset_array_pr0")


#            reset_array_init = tf.Variable(tf.constant(1.0, [9216, size]), trainable=False, name="reset_array_init")

            reset_masked_wfc1_pr0 = tf.placeholder(tf.float32, [6400,size], name="reset_masked_wfc1_pr0")
            reset_masked_wfc2_pr0 = tf.placeholder(tf.float32, [size,4096], name="reset_masked_wfc2_pr0")
            reset_masked_wfc3_pr0 = tf.placeholder(tf.float32, [4096,1000], name="reset_masked_wfc3_pr0")


            initial_wfc1_pr0 = tf.Variable(tf.truncated_normal([6400,size], stddev=0.01), dtype=tf.float32, trainable=False, name="initial_wfc1_pr0")
            initial_bfc1_pr0 = tf.Variable(tf.truncated_normal([size], stddev=0.01), dtype=tf.float32, trainable=False, name="initial_bfc1_pr0")
            initial_wfc2_pr0 = tf.Variable(tf.truncated_normal([size,4096], stddev=0.01), dtype=tf.float32, trainable=False, name="initial_wfc2_pr0")


            initial_wfc1_ph_pr0 = tf.placeholder(tf.float32, [6400,size], name="initial_wfc1_ph_pr0")
            initial_bfc1_ph_pr0 = tf.placeholder(tf.float32, [size], name="initial_bfc1_ph_pr0")
            initial_wfc2_ph_pr0 = tf.placeholder(tf.float32, [size,4096], name="initial_wfc2_ph_pr0")
        
        # queue of examples being filled on the cpu
        with tf.device('/cpu:0'):
            q_pr0 = tf.FIFOQueue(batch_size * 3, [tf.float32, tf.float32], shapes=[[224, 224, 3], [1000]])
            enqueue_op_pr0 = q_pr0.enqueue_many([x_pr0, y_pr0], name='enqueue_op_pr0')
            x_b_pr0, y_b_pr0 = q_pr0.dequeue_many(batch_size)

        

        with tf.name_scope('alexnetwork_pr0') as scope:
            with tf.name_scope('conv1_pr0') as inner_scope:
                wcnn1_pr0 = tu.weight([11, 11, 3, 96], name='wcnn1_pr0', train=False)
                bcnn1_pr0 = tu.bias(0.0, [96], name='bcnn1_pr0', train=False)
                conv1_pr0 = tf.add(tu.conv2d(data_pr0, wcnn1_pr0, stride=(4, 4), padding='VALID'), bcnn1_pr0)
                #conv1 = tu.batch_norm(conv1)
                conv1_pr0 = tu.relu(conv1_pr0)
#                pool1_pr0 = tu.max_pool2d(conv1_pr0, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')
                norm1_pr0 = tu.lrn(conv1_pr0, depth_radius=5, bias=2.0, alpha=1e-04, beta=0.75)
                pool1_pr0 = tu.max_pool2d(norm1_pr0, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')

            with tf.name_scope('conv2_pr0') as inner_scope:
                wcnn2_pr0 = tu.weight([5, 5, 96, 256], name='wcnn2_pr0', train=False)
                bcnn2_pr0 = tu.bias(0.1, [256], name='bcnn2_pr0', train=False)
                conv2_pr0 = tf.add(tu.conv2d(pool1_pr0, wcnn2_pr0, stride=(1, 1), padding='SAME'), bcnn2_pr0)
                #conv2 = tu.batch_norm(conv2)
                conv2_pr0 = tu.relu(conv2_pr0)
#                pool2_pr0 = tu.max_pool2d(conv2_pr0, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')
                norm2_pr0 = tu.lrn(conv2_pr0, depth_radius=5, bias=2.0, alpha=1e-04, beta=0.75)
                pool2_pr0 = tu.max_pool2d(norm2_pr0, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')

            with tf.name_scope('conv3_pr0') as inner_scope:
                wcnn3_pr0 = tu.weight([3, 3, 256, 384], name='wcnn3_pr0', train=False)
                bcnn3_pr0 = tu.bias(0.0, [384], name='bcnn3_pr0', train=False)
                conv3_pr0 = tf.add(tu.conv2d(pool2_pr0, wcnn3_pr0, stride=(1, 1), padding='SAME'), bcnn3_pr0)
                #conv3 = tu.batch_norm(conv3)
                conv3_pr0 = tu.relu(conv3_pr0)

            with tf.name_scope('conv4_pr0') as inner_scope:
                wcnn4_pr0 = tu.weight([3, 3, 384, 384], name='wcnn4_pr0', train=False)
                bcnn4_pr0 = tu.bias(0.1, [384], name='bcnn4_pr0', train=False)
                conv4_pr0 = tf.add(tu.conv2d(conv3_pr0, wcnn4_pr0, stride=(1, 1), padding='SAME'), bcnn4_pr0)
                #conv4 = tu.batch_norm(conv4)
                conv4_pr0 = tu.relu(conv4_pr0)

            with tf.name_scope('conv5_pr0') as inner_scope:
                wcnn5_pr0 = tu.weight([3, 3, 384, 256], name='wcnn5_pr0', train=False)
                bcnn5_pr0 = tu.bias(0.1, [256], name='bcnn5_pr0', train=False)
                conv5_pr0 = tf.add(tu.conv2d(conv4_pr0, wcnn5_pr0, stride=(1, 1), padding='SAME'), bcnn5_pr0)
                #conv5 = tu.batch_norm(conv5)
                conv5_pr0 = tu.relu(conv5_pr0)
                pool5_pr0 = tu.max_pool2d(conv5_pr0, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')
            

#        dim_un0 = pool5_un0.get_shape().as_list()
#        flat_dim_un0 = dim_un0[1] * dim_un0[2] * dim_un0[3] # 6 * 6 * 256
        flat_pr0 = tf.reshape(pool5_pr0, [-1, 6400])
        print("flat shape")
        print(flat_pr0.get_shape().as_list())


        with tf.name_scope('classifier_pr0') as scope:
            with tf.name_scope('fullyconected1_pr0') as inner_scope:
                wfc1_pr0 = tu.weight([6400, size], name='wfc1_pr0', train=True)
                bfc1_pr0 = tu.bias(0.1, [size], name='bfc1_pr0', train=True)
                fc1_pr0 = tf.add(tf.matmul(flat_pr0, wfc1_pr0), bfc1_pr0)
                #fc1 = tu.batch_norm(fc1)
#               fc1 = tu.relu(fc1)
                fc1_pr0 = tf.nn.dropout(fc1_pr0, keep_prob_pr0)
#               fc1 = tf.multiply(fc1, half)

            with tf.name_scope('fullyconected2_pr0') as inner_scope:
                wfc2_pr0 = tu.weight([size, 4096], name='wfc2_pr0', train=True)
                bfc2_pr0 = tu.bias(0.1, [4096], name='bfc2_pr0', train=True)
                fc2_pr0 = tf.add(tf.matmul(fc1_pr0, wfc2_pr0), bfc2_pr0)
                #fc2 = tu.batch_norm(fc2)
                fc2_pr0 = tu.relu(fc2_pr0)
                fc2_pr0 = tf.nn.dropout(fc2_pr0, keep_prob_pr0)
#               fc2 = tf.multiply(fc2, half)

            with tf.name_scope('classifier_output_pr0') as inner_scope:
                wfc3_pr0 = tu.weight([4096, 1000], name='wfc3_pr0', train=True)
                bfc3_pr0 = tu.bias(0.0, [1000], name='bfc3_pr0', train=True)
                fc3_pr0 = tf.add(tf.matmul(fc2_pr0, wfc3_pr0), bfc3_pr0)
                softmax_pr0 = tf.nn.softmax(fc3_pr0)
        













        

#        pred_un0, _ = alexnet_nonrelu_un0.classifier_un0(x_b_un0, keep_prob_un0, size)

        assign_masked_wcnn1_pr0 = tf.assign(wcnn1_pr0, masked_wcnn1_pr0)
        assign_masked_bcnn1_pr0 = tf.assign(bcnn1_pr0, masked_bcnn1_pr0)
        assign_masked_wcnn2_pr0 = tf.assign(wcnn2_pr0, masked_wcnn2_pr0)
        assign_masked_bcnn2_pr0 = tf.assign(bcnn2_pr0, masked_bcnn2_pr0)
        assign_masked_wcnn3_pr0 = tf.assign(wcnn3_pr0, masked_wcnn3_pr0)
        assign_masked_bcnn3_pr0 = tf.assign(bcnn3_pr0, masked_bcnn3_pr0)
        assign_masked_wcnn4_pr0 = tf.assign(wcnn4_pr0, masked_wcnn4_pr0)
        assign_masked_bcnn4_pr0 = tf.assign(bcnn4_pr0, masked_bcnn4_pr0)
        assign_masked_wcnn5_pr0 = tf.assign(wcnn5_pr0, masked_wcnn5_pr0)
        assign_masked_bcnn5_pr0 = tf.assign(bcnn5_pr0, masked_bcnn5_pr0)
        assign_masked_wfc1_pr0 = tf.assign(wfc1_pr0, masked_wfc1_pr0)
        assign_masked_bfc1_pr0 = tf.assign(bfc1_pr0, masked_bfc1_pr0)
        assign_masked_wfc2_pr0 = tf.assign(wfc2_pr0, masked_wfc2_pr0)
        assign_masked_bfc2_pr0 = tf.assign(bfc2_pr0, masked_bfc2_pr0)
        assign_masked_wfc3_pr0 = tf.assign(wfc3_pr0, masked_wfc3_pr0)
        assign_masked_bfc3_pr0 = tf.assign(bfc3_pr0, masked_bfc3_pr0)

        assign_reset_wfc1_pr0 = tf.assign(wfc1_reset_array_pr0, reset_masked_wfc1_pr0)
        assign_reset_wfc2_pr0 = tf.assign(wfc2_reset_array_pr0, reset_masked_wfc2_pr0)
        assign_reset_wfc3_pr0 = tf.assign(wfc3_reset_array_pr0, reset_masked_wfc3_pr0)

        assign_initial_wfc1_pr0 = tf.assign(initial_wfc1_pr0, initial_wfc1_ph_pr0)
        assign_initial_bfc1_pr0 = tf.assign(initial_bfc1_pr0, initial_bfc1_ph_pr0)
        assign_initial_wfc2_pr0 = tf.assign(initial_wfc2_pr0, initial_wfc2_ph_pr0)

        make_reset_wfc1_pr0 = tf.multiply(wfc1_pr0, wfc1_reset_array_pr0)
        reset_wfc1_pr0 = tf.assign(wfc1_pr0, make_reset_wfc1_pr0)

        make_reset_wfc2_pr0 = tf.multiply(wfc2_pr0, wfc2_reset_array_pr0)
        reset_wfc2_pr0 = tf.assign(wfc2_pr0, make_reset_wfc2_pr0)

        make_reset_wfc3_pr0 = tf.multiply(wfc3_pr0, wfc3_reset_array_pr0)
        reset_wfc3_pr0 = tf.assign(wfc3_pr0, make_reset_wfc3_pr0)


        with tf.device('/gpu:0'):
            # cross-entropy and weight decay
            with tf.name_scope('cross_entropy_pr0'):
                cross_entropy_pr0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3_pr0, labels=label_pr0, name='cross-entropy_pr0'))
	
            with tf.name_scope('l2_loss_pr0'):
                l2_loss_pr0 = tf.reduce_sum(lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('weights_pr0')]))
                tf.summary.scalar('l2_loss_pr0', l2_loss_pr0)
	
            with tf.name_scope('loss_pr0'):
                loss_pr0 = cross_entropy_pr0 + l2_loss_pr0
                tf.summary.scalar('loss_pr0', loss_pr0)
            
            # accuracy
            with tf.name_scope('accuracy_pr0'):
                correct_pr0 = tf.equal(tf.argmax(softmax_pr0, 1), tf.argmax(label_pr0, 1))
                accuracy_pr0 = tf.reduce_mean(tf.cast(correct_pr0, tf.float32))
                tf.summary.scalar('accuracy_pr0', accuracy_pr0)
            
            topk_correct_pr0 = tf.nn.in_top_k(softmax_pr0, tf.argmax(label_pr0, 1), k=5)
            topk_accuracy_pr0 = tf.reduce_mean(tf.cast(topk_correct_pr0, tf.float32))
	
            global_step_pr0 = tf.Variable(0, trainable=False)
            epoch_pr0 = tf.div(global_step_pr0, num_batches)
	
            # momentum optimizer
            with tf.name_scope('optimizer_pr0'):
#               optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
                optimizer_pr0 = tf.train.MomentumOptimizer(learning_rate=lr_pr0, momentum=momentum).minimize(loss_pr0, global_step=global_step_pr0)
            
            # merge summaries to write them to file
            merged_pr0 = tf.summary.merge_all()

            # checkpoint saver
            saver_pr0 = tf.train.Saver()

            coord_pr0 = tf.train.Coordinator()

            #init = tf.initialize_all_variables()
            init_pr0 = tf.global_variables_initializer()
        


        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(graph=pruning_model_pr0, config=config) as sess:
#        with tf.Session(graph=pruning_model_pr0, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            if analysis:
                saver_pr0.restore(sess, os.path.join(save_path_pr0, 'alexnet-cnn.ckpt'))

                wcnn1_after = sess.run(wcnn1_pr0)
                bcnn1_after = sess.run(bcnn1_pr0)
                wcnn2_after = sess.run(wcnn2_pr0)
                bcnn2_after = sess.run(bcnn2_pr0)
                wcnn3_after = sess.run(wcnn3_pr0)
                bcnn3_after = sess.run(bcnn3_pr0)
                wcnn4_after = sess.run(wcnn4_pr0)
                bcnn4_after = sess.run(bcnn4_pr0)
                wcnn5_after = sess.run(wcnn5_pr0)
                bcnn5_after = sess.run(bcnn5_pr0)
                wfc1_after = sess.run(wfc1_pr0)
                bfc1_after = sess.run(bfc1_pr0)
                wfc2_after = sess.run(wfc2_pr0)
                bfc2_after = sess.run(bfc2_pr0)
                wfc3_after = sess.run(wfc3_pr0)
                bfc3_after = sess.run(bfc3_pr0)

#                with open('wfc1_re.txt','w')as f:
#                    for i in range(0,int(wfc1_after.shape[0]-1)):
 #                       for j in range(0,int(wfc1_after.shape[1]-1)):
#                     np.savetxt('wfc1_re.csv',wfc1_after,fmt='%f',delimiter=',')       f.write('{}\n'.format(wfc1_after[i][j]))
                            



                shape_list_after = []
                non_zero_element_after = []
                memory_size_after = []

                shape_list_after.append(wcnn1_after.shape)
                shape_list_after.append(bcnn1_after.shape)
                shape_list_after.append(wcnn2_after.shape)
                shape_list_after.append(bcnn2_after.shape)
                shape_list_after.append(wcnn3_after.shape)
                shape_list_after.append(bcnn3_after.shape)
                shape_list_after.append(wcnn4_after.shape)
                shape_list_after.append(bcnn4_after.shape)
                shape_list_after.append(wcnn5_after.shape)
                shape_list_after.append(bcnn5_after.shape)
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
                non_zero_element_after.append(wcnn3_after.size)
                non_zero_element_after.append(bcnn3_after.size)
                non_zero_element_after.append(wcnn4_after.size)
                non_zero_element_after.append(bcnn4_after.size)
                non_zero_element_after.append(wcnn5_after.size)
                non_zero_element_after.append(bcnn5_after.size)
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
                memory_size_after.append(wcnn3_after.size*4)
                memory_size_after.append(bcnn3_after.size*4)
                memory_size_after.append(wcnn4_after.size*4)
                memory_size_after.append(bcnn4_after.size*4)
                memory_size_after.append(wcnn5_after.size*4)
                memory_size_after.append(bcnn5_after.size*4)
                memory_size_after.append(min(2*np.count_nonzero(wfc1_after)+wfc1_after.shape[1]+1, wfc1_after.size)*4)
                memory_size_after.append(bfc1_after.size*4)
                memory_size_after.append(min(2*np.count_nonzero(wfc2_after)+wfc2_after.shape[0]+1, wfc2_after.size)*4)
                memory_size_after.append(bfc2_after.size*4)
                memory_size_after.append(min(2*np.count_nonzero(wfc3_after)+wfc3_after.shape[1]+1, wfc3_after.size)*4)
                memory_size_after.append(bfc3_after.size*4)


                cnn_memory_size_after = 0
                for i in range(10):
                    cnn_memory_size_after += memory_size_after[i]

                fc_memory_size_after = 0
                for i in range(10, 16):
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

                fc_memory_size_after_merge = memory_size_after[13]+memory_size_after[14]+memory_size_after[15]+memory_size_after[16]
                add_one = 32
                total_memory_size_after_merge = cnn_memory_size_after+add_one+fc_memory_size_after_merge



                print("")
                print("after")
                print("shape")
                print("wfc1:{}, bfc1:{}, wfc2:{}, bfc2:{}, wfc3:{}, bfc3:{}".format(shape_list_after[10], shape_list_after[11], shape_list_after[12], shape_list_after[13], shape_list_after[14], shape_list_after[15]))
                print("the number of non zero element")
                print("wfc1:{}, bfc1:{}, wfc2:{}, bfc2:{}, wfc3:{}, bfc3:{}".format(non_zero_element_after[10], non_zero_element_after[11], non_zero_element_after[12], non_zero_element_after[13], non_zero_element_after[14], non_zero_element_after[15]))
                print("memory size[MB]")
                print("wfc1:{}, bfc1:{}, wfc2:{}, bfc2:{}, wfc3:{}, bfc3:{}".format(memory_size_after[10]/float(1000000), memory_size_after[11]/float(1000000), memory_size_after[12]/float(1000000), memory_size_after[13]/float(1000000), memory_size_after[14]/float(1000000), memory_size_after[15]/float(1000000)))
                print("cnn memory size[MB]")
                print(cnn_memory_size_after/float(1000000))
                print("fc memory size[MB]")
                print(fc_memory_size_after/float(1000000))
                print("total memory size[MB]")
                print(total_memory_size_after/float(1000000))
                print("")

                print("merge wfc1 bfc1 wfc2")
                print("shape")
                print("wfc1_bfc1_wfc2:{}".format(shape_list_after[16]))
                print("the number of non zero element")
                print("wfc1_bfc1_wfc2:{}".format(non_zero_element_after[16]))
                print("memory size[MB]")
                print("wfc1_bfc1_wfc2:{}".format(memory_size_after[16]/float(1000000)))
                print("cnn memory size[MB]")
                print(cnn_memory_size_after/float(1000000))
                print("fc memory size[MB]")
                print(fc_memory_size_after_merge/float(1000000))
                print("total memory size[MB]")
                print(total_memory_size_after_merge/float(1000000))



#                weight_array = sess.run(wfc1_pr0)
#                m = np.count_nonzero(weight_array)
#                print("weight_array size is {}".format(weight_array.size))
#                nonzero_ratio = (m / weight_array.size) *100

#                n = weight_array.shape[1]

#                storage_sparse = (2*m+n+1)*4
#                storage_dense = weight_array.size*4
#                print("m = {}, n = {}, nonzero_ratio = {}%".format(m,n,nonzero_ratio))
#                print("storage sparse is {}[B]".format(storage_sparse))
#                print("storage dense is {}[B]".format(storage_dense))
                sys.exit()
            
            if test:
                saver_pr0.restore(sess, os.path.join(save_path_pr0, 'alexnet-cnn.ckpt'))

                test_images = sorted(os.listdir(os.path.join(imagenet_path, 'ILSVRC2012_img_val')))
                test_labels = tu.read_test_labels(os.path.join(imagenet_path, 'ILSVRC2012_devkit_t12/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'))
                test_examples = len(test_images)

                total_top1_accuracy = 0.
                total_topk_accuracy = 0.

                for i in range(test_examples):
                    image_patches_1_fix = tu.read_k_patches(os.path.join(imagenet_path, 'ILSVRC2012_img_val', test_images[i]), 1, True)
                    label = test_labels[i]

                    top1_a, topk_a = sess.run([accuracy_pr0, topk_accuracy_pr0], feed_dict={data_pr0: image_patches_1_fix, label_pr0: [label], lr_pr0: learning_rate, keep_prob_pr0: 1.0})
                    total_top1_accuracy += top1_a
                    total_topk_accuracy += topk_a

                    if i % display_step == 0:
                        print ('Examples done: {:5d}/{} ---- Top-1: {:.4f} -- Top-{}: {:.4f}'.format(i + 1, test_examples, total_top1_accuracy / (i + 1), top_k, total_topk_accuracy / (i + 1)))
                
                print ('---- Final accuracy ----')
                print ('Top-1: {:.4f} -- Top-{}: {:.4f}'.format(total_top1_accuracy / test_examples, top_k, total_topk_accuracy / test_examples))
                print ('Top-1 error rate: {:.4f} -- Top-{} error rate: {:.4f}'.format(1 - (total_top1_accuracy / test_examples), top_k, 1 - (total_topk_accuracy / test_examples)))

                sys.exit()

            
            elif resume:
                saver_pr0.restore(sess, os.path.join(save_path_pr0, 'alexnet-cnn.ckpt'))
#                saver_un0.restore(sess, '/home/sin-skmt/h-iwasaki/alexnet/codes/ckpt-alexnet-un0/alexnet-cnn.ckpt')
            else:
                sess.run(init_pr0)
            

                sess.run(assign_masked_wcnn1_pr0, feed_dict = {masked_wcnn1_pr0 : wcnn1_val})
                sess.run(assign_masked_bcnn1_pr0, feed_dict = {masked_bcnn1_pr0 : bcnn1_val})
                sess.run(assign_masked_wcnn2_pr0, feed_dict = {masked_wcnn2_pr0 : wcnn2_val})
                sess.run(assign_masked_bcnn2_pr0, feed_dict = {masked_bcnn2_pr0 : bcnn2_val})
                sess.run(assign_masked_wcnn3_pr0, feed_dict = {masked_wcnn3_pr0 : wcnn3_val})
                sess.run(assign_masked_bcnn3_pr0, feed_dict = {masked_bcnn3_pr0 : bcnn3_val})
                sess.run(assign_masked_wcnn4_pr0, feed_dict = {masked_wcnn4_pr0 : wcnn4_val})
                sess.run(assign_masked_bcnn4_pr0, feed_dict = {masked_bcnn4_pr0 : bcnn4_val})
                sess.run(assign_masked_wcnn5_pr0, feed_dict = {masked_wcnn5_pr0 : wcnn5_val})
                sess.run(assign_masked_bcnn5_pr0, feed_dict = {masked_bcnn5_pr0 : bcnn5_val})
                sess.run(assign_masked_wfc1_pr0, feed_dict = {masked_wfc1_pr0 : small_wfc1})
                sess.run(assign_masked_bfc1_pr0, feed_dict = {masked_bfc1_pr0 : small_bfc1})
                sess.run(assign_masked_wfc2_pr0, feed_dict = {masked_wfc2_pr0 : small_wfc2})
                sess.run(assign_masked_bfc2_pr0, feed_dict = {masked_bfc2_pr0 : bfc2_val})
                sess.run(assign_masked_wfc3_pr0, feed_dict = {masked_wfc3_pr0 : wfc3_val})
                sess.run(assign_masked_bfc3_pr0, feed_dict = {masked_bfc3_pr0 : bfc3_val})

                sess.run(assign_reset_wfc1_pr0, feed_dict = {reset_masked_wfc1_pr0 : wfc1_reset_array})
                sess.run(assign_reset_wfc2_pr0, feed_dict = {reset_masked_wfc2_pr0 : wfc2_reset_array})
                sess.run(assign_reset_wfc3_pr0, feed_dict = {reset_masked_wfc3_pr0 : wfc3_reset_array})


                sess.run(assign_initial_wfc1_pr0, feed_dict = {initial_wfc1_ph_pr0 : small_wfc1})
                sess.run(assign_initial_bfc1_pr0, feed_dict = {initial_bfc1_ph_pr0 : small_bfc1})
                sess.run(assign_initial_wfc2_pr0, feed_dict = {initial_wfc2_ph_pr0 : small_wfc2})

                saving = tf.train.Saver()
                save_path = saving.save(sess, os.path.join(save_path_pr0, 'alexnet-cnn.ckpt'))
                print('Variables saved in file: %s' % save_path)
                if  (not test and not resume and not analysis):
                    sys.exit()
#            reset_weight_init = sess.run(reset_array_init)
#            if pruning_stage != NUM_OF_PRUNING_STEP:
#                reset_weight = make_reset_array(int(pruning_num_list[0]), reset_weight_init, min_order_index_list)
#            else:
#                reset_weight = make_reset_array(int(pruning_num_list[1]), reset_weight_init, min_order_index_list)


            # enqueuing batches procedure
            def enqueue_batches_pr0():
                while not coord_pr0.should_stop():
                    im_pr0, l_pr0 = tu.read_batch(batch_size, train_img_path, wnid_labels)
                    sess.run(enqueue_op_pr0, feed_dict={x_pr0: im_pr0,y_pr0: l_pr0})

            # creating and starting parallel threads to fill the queue
            num_threads_pr0 = threads_numbers
            for i in range(num_threads_pr0):
                t_pr0 = threading.Thread(target=enqueue_batches_pr0)
                t_pr0.setDaemon(True)
                t_pr0.start()
            

            # operation to write logs for tensorboard visualization
            train_writer_pr0 = tf.summary.FileWriter(os.path.join(summary_path, 'train'), sess.graph)

            start_time = time.time()
#		    print ("num_batches : {}".format(num_batches))
#		    num_batches = 1

#            print("pruning num is {}".format(int(pruning_num_list[pruning_stage])))
#            current_step = sess.run(global_step_pr0)
#            print("current step is {}".format(current_step))
#            if current_step <= 50000:
#                learning_rate = 0.01
#            elif current_step > 50000 and current_step <= 100000:
#                learning_rate = 0.001
#            elif current_step > 100000 and current_step <= 150000:
#                learning_rate = 0.0001
#            else:
#                learning_rate = 0.00001
            
            for e in range(sess.run(epoch_pr0), epochs):
                for i in range(num_batches):

                    data, label = sess.run([x_b_pr0, y_b_pr0])
                    

                    _, step = sess.run([optimizer_pr0, global_step_pr0], feed_dict={data_pr0:data, label_pr0:label, lr_pr0: learning_rate, keep_prob_pr0: dropout})
                    
                    sess.run(reset_wfc1_pr0)
                    sess.run(reset_wfc2_pr0)
                    sess.run(reset_wfc3_pr0)

#                    count=0
#                    for k in range(array.shape[0]):
#                        for l in range(array.shape[1]):
#                            if array[k][l]==0:
#                                count+=1
#                    print("number of 0 is {}".format(count))



#                    target_array = sess.run(wfc1_un0)
#                    if pruning_stage != NUM_OF_PRUNING_STEP:
#                        masked_weight=prune_weights(int(pruning_num_list[0]), target_array, min_order_index_list)
#                    else:
#                        print(pruning_num_list[1])
#                        masked_weight=prune_weights(int(pruning_num_list[1]), target_array, min_order_index_list)
#                    sess.run(assign_masked_wfc1_un0, feed_dict = {masked_wfc1_un0 : masked_weight})
                    #train_writer.add_summary(summary, step)

                    # decaying learning rate
#                    if step == 25000 or step == 50000 or step==75000:
#                        learning_rate /= 10

                    # display current training informations


                    if step % display_step == 0:
                        temp_time=time.time()
                        data, label = sess.run([x_b_pr0, y_b_pr0])
                        c, a = sess.run([loss_pr0, accuracy_pr0], feed_dict={data_pr0:data, label_pr0:label, lr_pr0: learning_rate, keep_prob_pr0: 1.0})
                        print ("time: ",temp_time-start_time,'Epoch: {:03d} Step/Batch: {:09d} --- Loss: {:.7f} Training accuracy: {:.4f}, learning rate: {}'.format(e, step, c, a, learning_rate))
					
                    # make test and evaluate validation accuracy
                    if step % test_step == 0:
                        val_im, val_cls = tu.read_validation_batch(batch_size, os.path.join(imagenet_path, 'ILSVRC2012_img_val'), os.path.join(imagenet_path, 'ILSVRC2012_devkit_t12/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'))
                        v_a, topk = sess.run([accuracy_pr0, topk_accuracy_pr0], feed_dict={data_pr0: val_im, label_pr0: val_cls, lr_pr0: learning_rate, keep_prob_pr0: 1.0})
                        # intermediate time
                        int_time = time.time()
                        print ('Elapsed time: {}'.format(tu.format_time(int_time - start_time)))
                        print ('Validation accuracy: {:.04f}'.format(v_a))
                        print ('Validation accuracy top5: {:.04f}'.format(topk))
                        # save weights to file
                        save_path = saver_pr0.save(sess, os.path.join(save_path_pr0, 'alexnet-cnn.ckpt'))
                        print('Variables saved in file: %s' % save_path)

            end_time = time.time()
            print ('Elapsed time: {}').format(tu.format_time(end_time - start_time))
            save_path = saver_pr0.save(sess, os.path.join(save_path_pr0, 'alexnet-cnn.ckpt'))
            print('Variables saved in file: %s' % save_path)
            coord_pr0.request_stop()









if __name__ == '__main__':
	Threads_numbers=4
	DROPOUT = 0.5
	LAMBDA = 0.0005 # for weight decay
	MOMENTUM = 0.9
	LEARNING_RATE = 0.000001
	EPOCHS = 90
	BATCH_SIZE = 256
	top_k = 5
	QUALITY_PARAMETER = 1.7
	column_num = int(sys.argv[1])
#	CKPT_PATH = '/home/sin-skmt/h-iwasaki/alexnet/codes/ckpt-alexnet-nonrelu'
#	if not os.path.exists(CKPT_PATH):
#		os.makedirs(CKPT_PATH)
	SUMMARY = '/home/sin-skmt/h-iwasaki/alexnet/codes/summary/kmeans/fc1/qp_' + str(QUALITY_PARAMETER) + '/manual/validation/' + str(column_num)
	if not os.path.exists(SUMMARY):
		os.makedirs(SUMMARY)
	CKPT_PATH_un0 = '/home/sin-skmt/h-iwasaki/alexnet/codes/ckpt/retrain_with_relu/fc1/qp_1.7'
	if not os.path.exists(CKPT_PATH_un0):
		os.makedirs(CKPT_PATH_un0)
	CKPT_PATH_un1 = '/home/sin-skmt/h-iwasaki/alexnet/codes/ckpt-alexnet-un0'
	if not os.path.exists(CKPT_PATH_un1):
		os.makedirs(CKPT_PATH_un1)
	SAVE_PATH_pr0 = '/home/sin-skmt/h-iwasaki/alexnet/codes/ckpt_pcp/kmeans/fc1/qp_' + str(QUALITY_PARAMETER) + '/manual/' + str(column_num)
	if not os.path.exists(SAVE_PATH_pr0):
		os.makedirs(SAVE_PATH_pr0)
	SAVE_PATH_un1 = '/home/sin-skmt/h-iwasaki/alexnet/codes/ckpt-alexnet-un1'
	if not os.path.exists(SAVE_PATH_un1):
		os.makedirs(SAVE_PATH_un1)
    
	analysis = False
	IMAGENET_PATH = '/home/h-iwasaki/data/ILSVRC2012/'
	DISPLAY_STEP = 10
	TEST_STEP = 500
	if len(sys.argv)==2:
		resume=False
		test = False
	elif sys.argv[2] == '-resume':
		resume = True
		test = False
	elif sys.argv[2] == '-test':
		resume = False
		test = True
        elif sys.argv[2] == '-analysis':
                resume = False
                test = False
                analysis=True

	proposed_scheme(
		Threads_numbers,
		EPOCHS, 
		BATCH_SIZE, 
		LEARNING_RATE, 
		DROPOUT, 
		LAMBDA, 
		MOMENTUM,
		analysis,
		resume, 
        test,
        top_k,
		column_num,
		IMAGENET_PATH, 
		DISPLAY_STEP, 
		TEST_STEP, 
		CKPT_PATH_un0,
        CKPT_PATH_un1, 
		SUMMARY,
        SAVE_PATH_pr0,
        SAVE_PATH_un1)
