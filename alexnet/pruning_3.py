
import sys
import os.path
import time
#from models import alexnet_nonrelu_un0
import tensorflow as tf
import util_un0 as tu
import numpy as np
import threading
import sympy


FC1_LEN=37748736
FC1_ROW=9216
FC1_COL=4096
PRUNING_RATIO=0.9
NUM_OF_PRUNING_STEP=1
ORIGINAL_DROPOUT_RATIO = 0.5
#REMAINED_COLUMN = math.floor(48000*(1-PRUNING_RATIO)/400)
#PRUNING_RATIO_REAL = (48000-400*REMAINED_COLUMN)/48000
#PRUNING_RATIO_PER_ITERATION=PRUNING_RATIO_REAL/NUM_OF_PRUNING_STEP


def pruning_number_list(total_len, row, col, unstructured, sparse_ratio_change):
    pruning_num_list = []
    delete_element = total_len* PRUNING_RATIO
    column_num = delete_element // row
    delete_element_true = row * column_num
    pruning_ratio_true = (delete_element_true / total_len) * 100
    pruning_ratio_changed = pruning_ratio_true

    if sparse_ratio_change:
        remain_elemet_true = total_len - delete_element_true
        remain_element_changed = (remain_elemet_true - col - 1)//2
        delete_element_true = total_len - remain_element_changed
        pruning_ratio_changed = (delete_element_true / total_len) * 100


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
                pruning_num_list.append((i+1)*pruning_number_regular)
            else:
                pruning_num_list.append(delete_element_true)
#        pruning_num_list.append(pruning_number_regular)
#        pruning_num_list.append(pruning_number_last)
    
    return pruning_num_list, pruning_ratio_true, pruning_ratio_changed


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


def prune_weights(num_of_pruning, array, min_order_index_list, retrain_iteration):
#	num_of_pruning=math.floor(len(min_order_index_list)*PRUNING_RATIO_PER_ITERATION*pruning_step)
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


def make_reset_array(num_of_pruning, target_row, target_col, min_order_index_list):
    target_array = np.ones((target_row, target_col), dtype = np.float32)
    for i in range(num_of_pruning):
        row=min_order_index_list[i][0]
        col=min_order_index_list[i][1]
        target_array[row][col]=0.0
    return target_array



def pruning(
threads_numbers,
		epochs, 
		batch_size, 
		learning_rate, 
		dropout, 
		lmbda,
		momentum, 
		check,
		resume, 
        test,
        retrain,
        compare,
        top_k,
        column_num,
        memory_size_dict,
		quality_parameter,
		retrain_iteration,
		imagenet_path, 
		display_step, 
		test_step, 
		ckpt_path_un0,
        ckpt_path_un1, 
		summary_path,
		save_path_un0,
        save_path_retrain):

#    pruning_stage = -1
    train_img_path = os.path.join(imagenet_path, 'ILSVRC2012_img_train')
    ts_size = tu.imagenet_size(train_img_path)
    print ("ts_size : {}".format(ts_size))
    num_batches = int(float(ts_size) / batch_size)
#    num_batches = 20

    wnid_labels, _ = tu.load_imagenet_meta(os.path.join(imagenet_path, 'ILSVRC2012_devkit_t12/ILSVRC2012_devkit_t12/data/meta.mat'))
#    pruning_num_list, pruning_ratio_true, pruning_ratio_changed = pruning_number_list(FC1_LEN, FC1_ROW, FC1_COL, True, False)


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
        wcnn1 = graph.get_tensor_by_name("alexnetwork/conv1/wcnn1:0")
        bcnn1 = graph.get_tensor_by_name("alexnetwork/conv1/bcnn1:0")
        wcnn2 = graph.get_tensor_by_name("alexnetwork/conv2/wcnn2:0")
        bcnn2 = graph.get_tensor_by_name("alexnetwork/conv2/bcnn2:0")
        wcnn3 = graph.get_tensor_by_name("alexnetwork/conv3/wcnn3:0")
        bcnn3 = graph.get_tensor_by_name("alexnetwork/conv3/bcnn3:0")
        wcnn4 = graph.get_tensor_by_name("alexnetwork/conv4/wcnn4:0")
        bcnn4 = graph.get_tensor_by_name("alexnetwork/conv4/bcnn4:0")
        wcnn5 = graph.get_tensor_by_name("alexnetwork/conv5/wcnn5:0")
        bcnn5 = graph.get_tensor_by_name("alexnetwork/conv5/bcnn5:0")
        wfc1 = graph.get_tensor_by_name("classifier/fullyconected1/wfc1:0")
        bfc1 = graph.get_tensor_by_name("classifier/fullyconected1/bfc1:0")
        wfc2 = graph.get_tensor_by_name("classifier/fullyconected2/wfc2:0")
        bfc2 = graph.get_tensor_by_name("classifier/fullyconected2/bfc2:0")
        wfc3 = graph.get_tensor_by_name("classifier/classifier_output/wfc3:0")
        bfc3 = graph.get_tensor_by_name("classifier/classifier_output/bfc3:0")
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

    


    
#    pruning_num_list, pruning_ratio_true = pruning_number_list(FC1_LEN, FC1_ROW, True)

    array_list = []
    array_list.append(wfc1_val)
    array_list.append(wfc2_val)
    array_list.append(wfc3_val)

    reset_weight_list = []
    for i in range(retrain_iteration):
        reset_weight_list.append([])

    if compare == True:
       x = sympy.Symbol('x')
       sparse_ratio = sympy.solve(2*(26214400*(1-x))+4097+4096+2*(16777216*(1-x))+4097+4096+2*(4096000*(1-x))+1001+1000-memory_size_dict[quality_parameter][column_num]*250000)
       print("sparse ratio:{}%".format(sparse_ratio[0]*100))
#       sys.exit()

       for i in range(len(array_list)):
           num_of_pruning = int(array_list[i].size * sparse_ratio[0])
           min_order_index_list = get_min_order_index_compare(array_list[i], array_list[i].shape[1])

           reset_weight = prune_weights(num_of_pruning, array_list[i], min_order_index_list, retrain_iteration)
           for j in range(retrain_iteration):
               reset_weight_list[j].append(reset_weight[j])


    elif test == False:
        std_list = []
        threshold_list = []


#    array_list = []
#    array_list.append(wfc1_val)
#    array_list.append(wfc2_val)
#    array_list.append(wfc3_val)

#    std_list = []
#    threshold_list = []
#    reset_weight_list = []
#    for i in range(retrain_iteration):
#        reset_weight_list.append([])

        for i in range(len(array_list)):
            std = np.std(array_list[i])
            threshold = std * quality_parameter
            std_list.append(std)
            threshold_list.append(threshold)

            min_order_index_list, the_number_of_under_threshold = get_min_order_index(array_list[i], threshold_list[i], array_list[i].shape[1])

            print("min_order_index_list:{}".format(min_order_index_list[0]))
            reset_weight = prune_weights(the_number_of_under_threshold, array_list[i], min_order_index_list, retrain_iteration)

            print("reset_weight_size:{}".format(reset_weight[0].size))
            for j in range(retrain_iteration):
                reset_weight_list[j].append(reset_weight[j])



        

    

         


#    target_array = wfc1_val
#    target_row = target_array.shape[0]
#    target_col = target_array.shape[1]
#    print("wfc1_shape:{}".format(target_array.shape))
#    min_order_index_list = get_min_order_index(target_array, target_col)
#    pruning_stage += 1
#    masked_weight=prune_weights(int(pruning_num_list[pruning_stage]), target_array, min_order_index_list)
#    reset_weight = make_reset_array(int(pruning_num_list[pruning_stage]), target_row, target_col, min_order_index_list)
#    dropout = 1-(ORIGINAL_DROPOUT_RATIO * np.sqrt((FC1_LEN-int(pruning_num_list[pruning_stage]))/FC1_LEN))
#    if pruning_stage != NUM_OF_PRUNING_STEP:
#        masked_weight=prune_weights(int(pruning_num_list[0]), target_array, min_order_index_list)
#        reset_weight = make_reset_array(int(pruning_num_list[0]), target_row, target_col, min_order_index_list)
#    else:
#        print(pruning_num_list[1])
#        masked_weight=prune_weights(int(pruning_num_list[1]), target_array, min_order_index_list)
#        reset_weight = make_reset_array(int(pruning_num_list[1]), target_row, target_col, min_order_index_list)
#    print("pruning stage {}".format(pruning_stage))
#    size = target_array.shape[1]



    pruning_model_un0 = tf.Graph()
    with pruning_model_un0.as_default():

        with tf.device('/gpu:0'):
            x_un0 = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x_un0')
            y_un0 = tf.placeholder(tf.float32, [None, 1000], name='y_un0')
            lr_un0 = tf.placeholder(tf.float32, name='lr_un0')
            keep_prob_un0 = tf.placeholder(tf.float32, name='keep_prob_un0')

            data_un0 = tf.placeholder(tf.float32, [None, 224, 224, 3], name="data_un0")
            label_un0 = tf.placeholder(tf.float32, [None, 1000], name="label_un0")

            masked_wcnn1_un0 = tf.placeholder(tf.float32, [11,11,3,96], name="masked_wcnn1_un0")
            masked_bcnn1_un0 = tf.placeholder(tf.float32, [96], name="masked_bcnn1_un0")
            masked_wcnn2_un0 = tf.placeholder(tf.float32, [5,5,96,256], name="masked_wcnn2_un0")
            masked_bcnn2_un0 = tf.placeholder(tf.float32, [256], name="masked_bcnn2_un0")
            masked_wcnn3_un0 = tf.placeholder(tf.float32, [3,3,256,384], name="masked_wcnn3_un0")
            masked_bcnn3_un0 = tf.placeholder(tf.float32, [384], name="masked_bcnn3_un0")
            masked_wcnn4_un0 = tf.placeholder(tf.float32, [3,3,384,384], name="masked_wcnn4_un0")
            masked_bcnn4_un0 = tf.placeholder(tf.float32, [384], name="masked_bcnn4_un0")
            masked_wcnn5_un0 = tf.placeholder(tf.float32, [3,3,384,256], name="masked_wcnn5_un0")
            masked_bcnn5_un0 = tf.placeholder(tf.float32, [256], name="masked_bcnn5_un0")
            masked_wfc1_un0 = tf.placeholder(tf.float32, [6400,4096], name="masked_wfc1_un0")
            masked_bfc1_un0 = tf.placeholder(tf.float32, [4096], name="masked_bfc1_un0")
            masked_wfc2_un0 = tf.placeholder(tf.float32, [4096,4096], name="masked_wfc2_un0")
            masked_bfc2_un0 = tf.placeholder(tf.float32, [4096], name="masked_bfc2_un0")
            masked_wfc3_un0 = tf.placeholder(tf.float32, [4096,1000], name="masked_wfc3_un0")
            masked_bfc3_un0 = tf.placeholder(tf.float32, [1000], name="masked_bfc3_un0")


#            reset_array_init = tf.Variable(tf.constant(1.0, [9216, size]), trainable=False, name="reset_array_init")

            reset_masked_wfc1_un0 = tf.placeholder(tf.float32, [6400,4096], name="reset_masked_wfc1_un0")
            reset_masked_wfc2_un0 = tf.placeholder(tf.float32, [4096,4096], name="reset_masked_wfc2_un0")
            reset_masked_wfc3_un0 = tf.placeholder(tf.float32, [4096,1000], name="reset_masked_wfc3_un0")
        
        # queue of examples being filled on the cpu
        with tf.device('/cpu:0'):
            q_un0 = tf.FIFOQueue(batch_size * 3, [tf.float32, tf.float32], shapes=[[224, 224, 3], [1000]])
            enqueue_op_un0 = q_un0.enqueue_many([x_un0, y_un0], name='enqueue_op_un0')
            x_b_un0, y_b_un0 = q_un0.dequeue_many(batch_size)

        

        with tf.name_scope('alexnetwork_un0') as scope:
            with tf.name_scope('conv1_un0') as inner_scope:
                wcnn1_un0 = tu.weight([11, 11, 3, 96], name='wcnn1_un0', train=False)
                bcnn1_un0 = tu.bias(0.0, [96], name='bcnn1_un0', train=False)
                conv1_un0 = tf.add(tu.conv2d(data_un0, wcnn1_un0, stride=(4, 4), padding='VALID'), bcnn1_un0)
                #conv1 = tu.batch_norm(conv1)
                conv1_un0 = tu.relu(conv1_un0)
#                pool1_un0 = tu.max_pool2d(conv1_un0, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')
                norm1_un0 = tu.lrn(conv1_un0, depth_radius=5, bias=2.0, alpha=1e-04, beta=0.75)
                pool1_un0 = tu.max_pool2d(norm1_un0, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')

            with tf.name_scope('conv2_un0') as inner_scope:
                wcnn2_un0 = tu.weight([5, 5, 96, 256], name='wcnn2_un0', train=False)
                bcnn2_un0 = tu.bias(0.1, [256], name='bcnn2_un0', train=False)
                conv2_un0 = tf.add(tu.conv2d(pool1_un0, wcnn2_un0, stride=(1, 1), padding='SAME'), bcnn2_un0)
                #conv2 = tu.batch_norm(conv2)
                conv2_un0 = tu.relu(conv2_un0)
#                pool2_un0 = tu.max_pool2d(conv2_un0, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')
                norm2_un0 = tu.lrn(conv2_un0, depth_radius=5, bias=2.0, alpha=1e-04, beta=0.75)
                pool2_un0 = tu.max_pool2d(norm2_un0, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')

            with tf.name_scope('conv3_un0') as inner_scope:
                wcnn3_un0 = tu.weight([3, 3, 256, 384], name='wcnn3_un0', train=False)
                bcnn3_un0 = tu.bias(0.0, [384], name='bcnn3_un0', train=False)
                conv3_un0 = tf.add(tu.conv2d(pool2_un0, wcnn3_un0, stride=(1, 1), padding='SAME'), bcnn3_un0)
                #conv3 = tu.batch_norm(conv3)
                conv3_un0 = tu.relu(conv3_un0)

            with tf.name_scope('conv4_un0') as inner_scope:
                wcnn4_un0 = tu.weight([3, 3, 384, 384], name='wcnn4_un0', train=False)
                bcnn4_un0 = tu.bias(0.1, [384], name='bcnn4_un0', train=False)
                conv4_un0 = tf.add(tu.conv2d(conv3_un0, wcnn4_un0, stride=(1, 1), padding='SAME'), bcnn4_un0)
                #conv4 = tu.batch_norm(conv4)
                conv4_un0 = tu.relu(conv4_un0)

            with tf.name_scope('conv5_un0') as inner_scope:
                wcnn5_un0 = tu.weight([3, 3, 384, 256], name='wcnn5_un0', train=False)
                bcnn5_un0 = tu.bias(0.1, [256], name='bcnn5_un0', train=False)
                conv5_un0 = tf.add(tu.conv2d(conv4_un0, wcnn5_un0, stride=(1, 1), padding='SAME'), bcnn5_un0)
                #conv5 = tu.batch_norm(conv5)
                conv5_un0 = tu.relu(conv5_un0)
                pool5_un0 = tu.max_pool2d(conv5_un0, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')
            

#        dim_un0 = pool5_un0.get_shape().as_list()
#        flat_dim_un0 = dim_un0[1] * dim_un0[2] * dim_un0[3] # 6 * 6 * 256
        flat_un0 = tf.reshape(pool5_un0, [-1, 6400])
        print("flat shape")
        print(flat_un0.get_shape().as_list())


        with tf.name_scope('classifier_un0') as scope:
            with tf.name_scope('fullyconected1_un0') as inner_scope:
                wfc1_un0 = tu.weight([6400, 4096], name='wfc1_un0', train=True)
                bfc1_un0 = tu.bias(0.1, [4096], name='bfc1_un0', train=True)
                fc1_un0 = tf.add(tf.matmul(flat_un0, wfc1_un0), bfc1_un0)
                #fc1 = tu.batch_norm(fc1)
                fc1_un0 = tu.relu(fc1_un0)
                fc1_un0 = tf.nn.dropout(fc1_un0, keep_prob_un0)
#               fc1 = tf.multiply(fc1, half)

            with tf.name_scope('fullyconected2_un0') as inner_scope:
                wfc2_un0 = tu.weight([4096, 4096], name='wfc2_un0', train=True)
                bfc2_un0 = tu.bias(0.1, [4096], name='bfc2_un0', train=True)
                fc2_un0 = tf.add(tf.matmul(fc1_un0, wfc2_un0), bfc2_un0)
                #fc2 = tu.batch_norm(fc2)
                fc2_un0 = tu.relu(fc2_un0)
                fc2_un0 = tf.nn.dropout(fc2_un0, keep_prob_un0)
#               fc2 = tf.multiply(fc2, half)

            with tf.name_scope('classifier_output_un0') as inner_scope:
                wfc3_un0 = tu.weight([4096, 1000], name='wfc3_un0', train=True)
                bfc3_un0 = tu.bias(0.0, [1000], name='bfc3_un0', train=True)
                fc3_un0 = tf.add(tf.matmul(fc2_un0, wfc3_un0), bfc3_un0)
                softmax_un0 = tf.nn.softmax(fc3_un0)
        













        

#        pred_un0, _ = alexnet_nonrelu_un0.classifier_un0(x_b_un0, keep_prob_un0, size)

        assign_masked_wcnn1_un0 = tf.assign(wcnn1_un0, masked_wcnn1_un0)
        assign_masked_bcnn1_un0 = tf.assign(bcnn1_un0, masked_bcnn1_un0)
        assign_masked_wcnn2_un0 = tf.assign(wcnn2_un0, masked_wcnn2_un0)
        assign_masked_bcnn2_un0 = tf.assign(bcnn2_un0, masked_bcnn2_un0)
        assign_masked_wcnn3_un0 = tf.assign(wcnn3_un0, masked_wcnn3_un0)
        assign_masked_bcnn3_un0 = tf.assign(bcnn3_un0, masked_bcnn3_un0)
        assign_masked_wcnn4_un0 = tf.assign(wcnn4_un0, masked_wcnn4_un0)
        assign_masked_bcnn4_un0 = tf.assign(bcnn4_un0, masked_bcnn4_un0)
        assign_masked_wcnn5_un0 = tf.assign(wcnn5_un0, masked_wcnn5_un0)
        assign_masked_bcnn5_un0 = tf.assign(bcnn5_un0, masked_bcnn5_un0)
        assign_masked_wfc1_un0 = tf.assign(wfc1_un0, masked_wfc1_un0)
        assign_masked_bfc1_un0 = tf.assign(bfc1_un0, masked_bfc1_un0)
        assign_masked_wfc2_un0 = tf.assign(wfc2_un0, masked_wfc2_un0)
        assign_masked_bfc2_un0 = tf.assign(bfc2_un0, masked_bfc2_un0)
        assign_masked_wfc3_un0 = tf.assign(wfc3_un0, masked_wfc3_un0)
        assign_masked_bfc3_un0 = tf.assign(bfc3_un0, masked_bfc3_un0)

        make_reset_wfc1_un0 = tf.multiply(wfc1_un0, reset_masked_wfc1_un0)
        reset_wfc1_un0 = tf.assign(wfc1_un0, make_reset_wfc1_un0)

        make_reset_wfc2_un0 = tf.multiply(wfc2_un0, reset_masked_wfc2_un0)
        reset_wfc2_un0 = tf.assign(wfc2_un0, make_reset_wfc2_un0)

        make_reset_wfc3_un0 = tf.multiply(wfc3_un0, reset_masked_wfc3_un0)
        reset_wfc3_un0 = tf.assign(wfc3_un0, make_reset_wfc3_un0)


        with tf.device('/gpu:0'):
            # cross-entropy and weight decay
            with tf.name_scope('cross_entropy_un0'):
                cross_entropy_un0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3_un0, labels=label_un0, name='cross-entropy_un0'))
	
            with tf.name_scope('l2_loss_un0'):
                l2_loss_un0 = tf.reduce_sum(lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('weights_un0')]))
                tf.summary.scalar('l2_loss_un0', l2_loss_un0)
	
            with tf.name_scope('loss_un0'):
                loss_un0 = cross_entropy_un0 + l2_loss_un0
                tf.summary.scalar('loss_un0', loss_un0)
            
            # accuracy
            with tf.name_scope('accuracy_un0'):
                correct_un0 = tf.equal(tf.argmax(softmax_un0, 1), tf.argmax(label_un0, 1))
                accuracy_un0 = tf.reduce_mean(tf.cast(correct_un0, tf.float32))
                tf.summary.scalar('accuracy_un0', accuracy_un0)
            
            topk_correct_un0 = tf.nn.in_top_k(softmax_un0, tf.argmax(label_un0, 1), k=5)
            topk_accuracy_un0 = tf.reduce_mean(tf.cast(topk_correct_un0, tf.float32))
	
            global_step_un0 = tf.Variable(0, trainable=False)
            epoch_un0 = tf.div(global_step_un0, num_batches)
            step_reset = tf.assign(global_step_un0, 0)
	
            # momentum optimizer
            with tf.name_scope('optimizer_un0'):
#               optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
                optimizer_un0 = tf.train.MomentumOptimizer(learning_rate=lr_un0, momentum=momentum).minimize(loss_un0, global_step=global_step_un0)
            
            # merge summaries to write them to file
            merged_un0 = tf.summary.merge_all()

            # checkpoint saver
            saver_un0 = tf.train.Saver()

            coord_un0 = tf.train.Coordinator()

            #init = tf.initialize_all_variables()
            init_un0 = tf.global_variables_initializer()
        

        config = tf.ConfigProto(allow_soft_placement=True)
#        config.gpu_options.allow_growth = True
#        with tf.Session(graph=pruning_model_un0, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True, log_device_placement=True)) as sess:
        with tf.Session(graph=pruning_model_un0, config=config) as sess:
            if check:
                saver_un0.restore(sess, os.path.join(save_path_un0, 'alexnet-cnn.ckpt'))

                weight_array = sess.run(wfc1_un0)
                m = np.count_nonzero(weight_array)

                n = weight_array.shape[1]

                storage = (2*m+n+1)*4
                print("m = {}, n = {}".format(m,n))
                print("storage is {}[B]".format(storage))
                sys.exit()
            

            
            if test:
                if retrain:
                   save_path_un0 = save_path_retrain
                saver_un0.restore(sess, os.path.join(save_path_un0, 'alexnet-cnn.ckpt'))

                test_images = sorted(os.listdir(os.path.join(imagenet_path, 'ILSVRC2012_img_val')))
                test_labels = tu.read_test_labels(os.path.join(imagenet_path, 'ILSVRC2012_devkit_t12/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'))
                test_examples = len(test_images)

                total_top1_accuracy = 0.
                total_topk_accuracy = 0.

                for i in range(test_examples):
                    image_patches_1_fix = tu.read_k_patches(os.path.join(imagenet_path, 'ILSVRC2012_img_val', test_images[i]), 1, True)
                    label = test_labels[i]

                    top1_a, topk_a = sess.run([accuracy_un0, topk_accuracy_un0], feed_dict={data_un0: image_patches_1_fix, label_un0: [label], lr_un0: learning_rate, keep_prob_un0: 1.0})
                    total_top1_accuracy += top1_a
                    total_topk_accuracy += topk_a

                    if i % display_step == 0:
                        print ('Examples done: {:5d}/{} ---- Top-1: {:.4f} -- Top-{}: {:.4f}'.format(i + 1, test_examples, total_top1_accuracy / (i + 1), top_k, total_topk_accuracy / (i + 1)))
                
                print ('---- Final accuracy ----')
                print ('Top-1: {:.4f} -- Top-{}: {:.4f}'.format(total_top1_accuracy / test_examples, top_k, total_topk_accuracy / test_examples))
                print ('Top-1 error rate: {:.4f} -- Top-{} error rate: {:.4f}'.format(1 - (total_top1_accuracy / test_examples), top_k, 1 - (total_topk_accuracy / test_examples)))

                sys.exit()

            elif resume:
                saver_un0.restore(sess, os.path.join(save_path_un0, 'alexnet-cnn.ckpt'))
#                saver_un0.restore(sess, '/home/h-iwasaki/alexnet/codes/ckpt-alexnet-un0/alexnet-cnn.ckpt')
            
            elif retrain:
                saver_un0.restore(sess, os.path.join(save_path_un0, 'alexnet-cnn.ckpt'))
                sess.run(step_reset)
                save_path_un0 = save_path_retrain

            else:
                sess.run(init_un0)
            

                sess.run(assign_masked_wcnn1_un0, feed_dict = {masked_wcnn1_un0 : wcnn1_val})
                sess.run(assign_masked_bcnn1_un0, feed_dict = {masked_bcnn1_un0 : bcnn1_val})
                sess.run(assign_masked_wcnn2_un0, feed_dict = {masked_wcnn2_un0 : wcnn2_val})
                sess.run(assign_masked_bcnn2_un0, feed_dict = {masked_bcnn2_un0 : bcnn2_val})
                sess.run(assign_masked_wcnn3_un0, feed_dict = {masked_wcnn3_un0 : wcnn3_val})
                sess.run(assign_masked_bcnn3_un0, feed_dict = {masked_bcnn3_un0 : bcnn3_val})
                sess.run(assign_masked_wcnn4_un0, feed_dict = {masked_wcnn4_un0 : wcnn4_val})
                sess.run(assign_masked_bcnn4_un0, feed_dict = {masked_bcnn4_un0 : bcnn4_val})
                sess.run(assign_masked_wcnn5_un0, feed_dict = {masked_wcnn5_un0 : wcnn5_val})
                sess.run(assign_masked_bcnn5_un0, feed_dict = {masked_bcnn5_un0 : bcnn5_val})
                sess.run(assign_masked_wfc1_un0, feed_dict = {masked_wfc1_un0 : wfc1_val})
                sess.run(assign_masked_bfc1_un0, feed_dict = {masked_bfc1_un0 : bfc1_val})
                sess.run(assign_masked_wfc2_un0, feed_dict = {masked_wfc2_un0 : wfc2_val})
                sess.run(assign_masked_bfc2_un0, feed_dict = {masked_bfc2_un0 : bfc2_val})
                sess.run(assign_masked_wfc3_un0, feed_dict = {masked_wfc3_un0 : wfc3_val})
                sess.run(assign_masked_bfc3_un0, feed_dict = {masked_bfc3_un0 : bfc3_val})

#            reset_weight_init = sess.run(reset_array_init)
#            if pruning_stage != NUM_OF_PRUNING_STEP:
#                reset_weight = make_reset_array(int(pruning_num_list[0]), reset_weight_init, min_order_index_list)
#            else:
#                reset_weight = make_reset_array(int(pruning_num_list[1]), reset_weight_init, min_order_index_list)


            # enqueuing batches procedure
            def enqueue_batches_un0():
                while not coord_un0.should_stop():
                    im_un0, l_un0 = tu.read_batch(batch_size, train_img_path, wnid_labels)
                    sess.run(enqueue_op_un0, feed_dict={x_un0: im_un0,y_un0: l_un0})

            # creating and starting parallel threads to fill the queue
            num_threads_un0 = threads_numbers
            for i in range(num_threads_un0):
                t_un0 = threading.Thread(target=enqueue_batches_un0)
                t_un0.setDaemon(True)
                t_un0.start()
            

            # operation to write logs for tensorboard visualization
            train_writer_un0 = tf.summary.FileWriter(os.path.join(summary_path, 'train'), sess.graph)

            start_time = time.time()
#		    print ("num_batches : {}".format(num_batches))
#		    num_batches = 1

#            print("pruning num is {}".format(int(pruning_num_list[pruning_stage])))
#            current_step = sess.run(global_step_un0)
#            print("current step is {}".format(current_step))
#            if current_step <= 50000:
#                learning_rate = 0.01
#            elif current_step > 50000 and current_step <= 100000:
#                learning_rate = 0.001
#            elif current_step > 100000 and current_step <= 150000:
#                learning_rate = 0.0001
#            else:
#                learning_rate = 0.00001
            
            for k in range(retrain_iteration):
                sess.run(reset_wfc1_un0, feed_dict={reset_masked_wfc1_un0:reset_weight_list[k][0]})
                sess.run(reset_wfc2_un0, feed_dict={reset_masked_wfc2_un0:reset_weight_list[k][1]})
                sess.run(reset_wfc3_un0, feed_dict={reset_masked_wfc3_un0:reset_weight_list[k][2]})

                learning_rate = 0.01
                sess.run(step_reset)
                for e in range(sess.run(epoch_un0), epochs):
                    for i in range(num_batches):

                        data, label = sess.run([x_b_un0, y_b_un0])
                    

                        _, step = sess.run([optimizer_un0, global_step_un0], feed_dict={data_un0:data, label_un0:label, lr_un0: learning_rate, keep_prob_un0: dropout})
                    
                        sess.run(reset_wfc1_un0, feed_dict={reset_masked_wfc1_un0:reset_weight_list[k][0]})
                        sess.run(reset_wfc2_un0, feed_dict={reset_masked_wfc2_un0:reset_weight_list[k][1]})
                        sess.run(reset_wfc3_un0, feed_dict={reset_masked_wfc3_un0:reset_weight_list[k][2]})


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
                        if step == 15000 or step == 30000 or step== 45000:
                            learning_rate /= 10

                    # display current training informations
                        if step % display_step == 0:
                            temp_time=time.time()
                            data, label = sess.run([x_b_un0, y_b_un0])
                            c, a = sess.run([loss_un0, accuracy_un0], feed_dict={data_un0:data, label_un0:label, lr_un0: learning_rate, keep_prob_un0: 1.0})
                            print ("time: ",temp_time-start_time,'retrain_iteration: {}, Epoch: {:03d} Step/Batch: {:09d} --- Loss: {:.7f} Training accuracy: {:.4f}, learning rate: {}'.format(k, e, step, c, a, learning_rate))
					
                    # make test and evaluate validation accuracy
                        if step % test_step == 0:
                            val_im, val_cls = tu.read_validation_batch(batch_size, os.path.join(imagenet_path, 'ILSVRC2012_img_val'), os.path.join(imagenet_path, 'ILSVRC2012_devkit_t12/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'))
                            v_a, topk = sess.run([accuracy_un0, topk_accuracy_un0], feed_dict={data_un0: val_im, label_un0: val_cls, lr_un0: learning_rate, keep_prob_un0: 1.0})
                        # intermediate time
                            int_time = time.time()
                            print ('Elapsed time: {}'.format(tu.format_time(int_time - start_time)))
                            print ('Validation accuracy: {:.04f}'.format(v_a))
                            print ('Validation accuracy top5: {:.04f}'.format(topk))
                        # save weights to file
                            save_path = saver_un0.save(sess, os.path.join(save_path_un0, 'alexnet-cnn.ckpt'))
                            print('Variables saved in file: %s' % save_path)

            end_time = time.time()
            print ('Elapsed time: {}').format(tu.format_time(end_time - start_time))
            save_path = saver_un0.save(sess, os.path.join(save_path_un0, 'alexnet-cnn.ckpt'))
            print('Variables saved in file: %s' % save_path)
            coord_un0.request_stop()






    





if __name__ == '__main__':
	Threads_numbers=4
	DROPOUT = 0.5
	LAMBDA = 0.0005 # for weight decay
	MOMENTUM = 0.9
	LEARNING_RATE = 0.01
	EPOCHS = 10
	BATCH_SIZE = 256
	top_k = 5
	QUALITY_PARAMETER = 2.4
	RETRAIN_ITERATION = 3
	column_num = 4000
	memory_size_dict = {1.7:{4000:32.767892, 3500:29.786276, 3000:26.411244, 2500:22.84582, 2000:19.254852, 1500:15.492652, 750:9.562388, 500:7.79614, 250:6.185484, 100:5.069284, 75:4.792088, 50:4.476756, 25:3.987576, 10:3.357756}, 2.4:{4000:7.423364, 3000:6.27818, 2000:4.886444, 1000:3.25586, 750:2.480836, 500:2.190612, 250:1.481076, 100:2.024316}, 2.8:{4000:3.05978, 3000:2.59022, 2000:1.995164, 1000:1.868564, 750:1.798236, 500:1.404052, 100:1.765076}}
#	CKPT_PATH = '/home/h-iwasaki/alexnet/codes/ckpt-alexnet-nonrelu'
#	if not os.path.exists(CKPT_PATH):
#		os.makedirs(CKPT_PATH)
	SUMMARY = '/home/h-iwasaki/alexnet/codes/summary/compare/unstructured/fc1/qp_' + str(QUALITY_PARAMETER) + '/validation/' + str(column_num)
	if not os.path.exists(SUMMARY):
		os.makedirs(SUMMARY)
	CKPT_PATH_un0 = '/home/h-iwasaki/alexnet/codes/ckpt-256_norm_pool_random'
	if not os.path.exists(CKPT_PATH_un0):
		os.makedirs(CKPT_PATH_un0)
	CKPT_PATH_un1 = '/home/h-iwasaki/alexnet/codes/ckpt-alexnet-un0'
	if not os.path.exists(CKPT_PATH_un1):
		os.makedirs(CKPT_PATH_un1)
	SAVE_PATH_un0 = '/home/h-iwasaki/alexnet/codes/ckpt/compare/unstructured/fc1/qp_' + str(QUALITY_PARAMETER) + '/' + str(column_num)
	if not os.path.exists(SAVE_PATH_un0):
		os.makedirs(SAVE_PATH_un0)
	SAVE_PATH_retrain = '/home/h-iwasaki/alexnet/codes/ckpt/retrain_with_relu/fc1/qp_4.0'
	if not os.path.exists(SAVE_PATH_retrain):
		os.makedirs(SAVE_PATH_retrain)
	check = False
	retrain = False
	compare = False
#if retrain is True, remove relu function in fc layer
	IMAGENET_PATH = '/home/h-iwasaki/data/ILSVRC2012/'
	DISPLAY_STEP = 10
	TEST_STEP = 500
	if len(sys.argv)==1:
		resume=False
		test = False
	elif sys.argv[1] == '-resume':
		resume = True
		test = False
	elif sys.argv[1] == '-test':
		resume = False
		test = True



	pruning(
		Threads_numbers,
		EPOCHS, 
		BATCH_SIZE, 
		LEARNING_RATE, 
		DROPOUT, 
		LAMBDA, 
		MOMENTUM,
		check,
		resume, 
		test,
		retrain,
		compare,
		top_k,
		column_num,
		memory_size_dict,
		QUALITY_PARAMETER,
		RETRAIN_ITERATION,
		IMAGENET_PATH, 
		DISPLAY_STEP, 
		TEST_STEP, 
		CKPT_PATH_un0,
        CKPT_PATH_un1, 
		SUMMARY,
        SAVE_PATH_un0,
        SAVE_PATH_retrain)

        




