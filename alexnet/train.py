
import sys
import os.path
import time
from models import alexnet
import tensorflow as tf
import util as tu
import numpy as np
import threading

def train(
threads_numbers,
		epochs, 
		batch_size, 
		learning_rate, 
		dropout, 
		lmbda,
		momentum, 
		resume, 
		imagenet_path, 
		display_step, 
		test_step, 
		ckpt_path, 
		summary_path):
	"""	Procedure to train the model on ImageNet ILSVRC 2012 training set

		Args:
			resume:	boolean variable, true if want to resume the training, false to train from scratch
			imagenet_path:	path to ILSRVC12 ImageNet folder containing train images, 
						validation images, annotations and metadata file
			display_step: number representing how often printing the current training accuracy
			test_step: number representing how often make a test and print the validation accuracy
			ckpt_path: path where to save model's tensorflow checkpoint (or from where resume)
			summary_path: path where to save logs for TensorBoard

	"""
	train_img_path = os.path.join(imagenet_path, 'ILSVRC2012_img_train')
	ts_size = tu.imagenet_size(train_img_path)
	print ("ts_size : {}".format(ts_size))
	num_batches = int(float(ts_size) / batch_size)

	wnid_labels, _ = tu.load_imagenet_meta(os.path.join(imagenet_path, 'ILSVRC2012_devkit_t12/ILSVRC2012_devkit_t12/data/meta.mat'))
	with tf.device('/gpu:0'):
		x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x')
		y = tf.placeholder(tf.float32, [None, 1000], name='y')
		lr = tf.placeholder(tf.float32, name='lr')
		keep_prob = tf.placeholder(tf.float32, name='keep_prob')

		data_array = tf.placeholder(tf.float32, [None, 224, 224, 3], name="data_array")
		label_array = tf.placeholder(tf.float32, [None, 1000], name="label_array")

	# queue of examples being filled on the cpu
	with tf.device('/cpu:0'):
		q = tf.FIFOQueue(batch_size * 3, [tf.float32, tf.float32], shapes=[[224, 224, 3], [1000]])
		enqueue_op = q.enqueue_many([x, y], name='enqueue_op')
		x_b, y_b = q.dequeue_many(batch_size)

	pred, softmax = alexnet.classifier(data_array, keep_prob)
	with tf.device('/gpu:0'):
		# cross-entropy and weight decay
		with tf.name_scope('cross_entropy'):
			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label_array, name='cross-entropy'))
	
		with tf.name_scope('l2_loss'):
			l2_loss = tf.reduce_sum(lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('weights')]))
			tf.summary.scalar('l2_loss', l2_loss)
	
		with tf.name_scope('loss'):
			loss = cross_entropy + l2_loss
			tf.summary.scalar('loss', loss)

		# accuracy
		with tf.name_scope('accuracy'):
			correct = tf.equal(tf.argmax(softmax, 1), tf.argmax(label_array, 1))
			accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
			tf.summary.scalar('accuracy', accuracy)
		
		topk_correct = tf.nn.in_top_k(softmax, tf.argmax(label_array, 1), k=5)
		topk_accuracy = tf.reduce_mean(tf.cast(topk_correct, tf.float32))
	
		global_step = tf.Variable(0, trainable=False)
		epoch = tf.div(global_step, num_batches)
	
		# momentum optimizer
		with tf.name_scope('optimizer'):
#			optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
			optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum).minimize(loss, global_step=global_step)

		# merge summaries to write them to file
		merged = tf.summary.merge_all()

		# checkpoint saver
		saver = tf.train.Saver()

		coord = tf.train.Coordinator()

		#init = tf.initialize_all_variables()
		init = tf.global_variables_initializer()
# 
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
		if resume:
			saver.restore(sess, os.path.join(ckpt_path, 'alexnet-cnn.ckpt'))
		else:
			sess.run(init)

		# enqueuing batches procedure
		def enqueue_batches():
			while not coord.should_stop():
				im, l = tu.read_batch(batch_size, train_img_path, wnid_labels)
				sess.run(enqueue_op, feed_dict={x: im,y: l})

		# creating and starting parallel threads to fill the queue
		num_threads = threads_numbers
		for i in range(num_threads):
			t = threading.Thread(target=enqueue_batches)
			t.setDaemon(True)
			t.start()
		
		# operation to write logs for tensorboard visualization
		train_writer = tf.summary.FileWriter(os.path.join(summary_path, 'train'), sess.graph)

		start_time = time.time()

		current_step = sess.run(global_step)
		print("current step is {}".format(current_step))
		if current_step <= 100000:
			learning_rate = 0.01
		elif current_step > 100000 and current_step <= 200000:
			learning_rate = 0.001
		elif current_step > 200000 and current_step <= 300000:
			learning_rate = 0.0001
		else:
			learning_rate = 0.00001

#		print ("num_batches : {}".format(num_batches))
#		num_batches = 20
		for e in range(sess.run(epoch), epochs):
			for i in range(num_batches):

				data, label = sess.run([x_b, y_b])

				_, step = sess.run([optimizer, global_step], feed_dict={data_array:data, label_array:label, lr: learning_rate, keep_prob: dropout})
				#train_writer.add_summary(summary, step)

				# decaying learning rate
				if step == 100000 or step == 200000 or step==300000:
					learning_rate /= 10

				# display current training informations
				if step % display_step == 0:
					temp_time=time.time()
					data, label = sess.run([x_b, y_b])
					c, a = sess.run([loss, accuracy], feed_dict={data_array:data, label_array:label, lr: learning_rate, keep_prob: 1.0})
					print ("time: ",temp_time-start_time,'Epoch: {:03d} Step/Batch: {:09d} --- Loss: {:.7f} Training accuracy: {:.4f}, learning rate: {}'.format(e, step, c, a, learning_rate))
					
				# make test and evaluate validation accuracy
				if step % test_step == 0:
					val_im, val_cls = tu.read_validation_batch(batch_size, os.path.join(imagenet_path, 'ILSVRC2012_img_val'), os.path.join(imagenet_path, 'ILSVRC2012_devkit_t12/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'))
					v_a, topk = sess.run([accuracy, topk_accuracy], feed_dict={data_array:val_im, label_array:val_cls, lr: learning_rate, keep_prob: 1.0})
					# intermediate time
					int_time = time.time()
					print ('Elapsed time: {}'.format(tu.format_time(int_time - start_time)))
					print ('Validation accuracy: {:.04f}'.format(v_a))
					print ('Validation accuracy top5: {:.04f}'.format(topk))
					# save weights to file
					save_path = saver.save(sess, os.path.join(ckpt_path, 'alexnet-cnn.ckpt'))
					print('Variables saved in file: %s' % save_path)

		end_time = time.time()
		print ('Elapsed time: {}').format(tu.format_time(end_time - start_time))
		save_path = saver.save(sess, os.path.join(ckpt_path, 'alexnet-cnn.ckpt'))
		print('Variables saved in file: %s' % save_path)
		coord.request_stop()


if __name__ == '__main__':
	Threads_numbers=4
	DROPOUT = 0.5
	LAMBDA = 0.0005 # for weight decay
	MOMENTUM = 0.9
	LEARNING_RATE = 0.01
	EPOCHS = 90
	BATCH_SIZE = 256
	CKPT_PATH = '/home/h-iwasaki/alexnet/codes/ckpt-256_pool_norm_random'
	if not os.path.exists(CKPT_PATH):
		os.makedirs(CKPT_PATH)
	SUMMARY = '/home/h-iwasaki/alexnet/codes/summary-256_pool_norm_random'
	if not os.path.exists(SUMMARY):
		os.makedirs(SUMMARY)

	IMAGENET_PATH = '/home/h-iwasaki/data/ILSVRC2012/'
	DISPLAY_STEP = 10
	TEST_STEP = 500
	if len(sys.argv)==1:
		resume=False
	elif sys.argv[1] == '-resume':
		resume = True

	train(
		Threads_numbers,
		EPOCHS, 
		BATCH_SIZE, 
		LEARNING_RATE, 
		DROPOUT, 
		LAMBDA, 
		MOMENTUM,
		resume, 
		IMAGENET_PATH, 
		DISPLAY_STEP, 
		TEST_STEP, 
		CKPT_PATH, 
		SUMMARY)
