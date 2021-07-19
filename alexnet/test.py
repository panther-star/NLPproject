
import os.path
import tensorflow as tf
import util as tu
from models import alexnet
import numpy as np

def test( 
		top_k, 
		k_patches, 
		display_step,
		imagenet_path,
		ckpt_path):
	"""
	Procedure to evaluate top-1 and top-k accuracy (and error-rate) on the 
	ILSVRC2012 validation (test) set.

	Args:
		top_k: 	integer representing the number of predictions with highest probability
				to retrieve
		k_patches:	number of crops taken from an image and to input to the model
		display_step: number representing how often printing the current testing accuracy
		imagenet_path:	path to ILSRVC12 ImageNet folder containing train images, 
						validation images, annotations and metadata file 
		ckpt_path:	path to model's tensorflow checkpoint
	"""

	test_images = sorted(os.listdir(os.path.join(imagenet_path, 'ILSVRC2012_img_val')))
#	print("test_images")
#	print(len(test_images))
#	print(test_images[0])
	test_labels = tu.read_test_labels(os.path.join(imagenet_path, 'ILSVRC2012_devkit_t12/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'))
#	print("test_labels")
#	print(len(test_labels))
#	print(test_labels[0])
#	print(test_labels[0][489])

	test_examples = len(test_images)
#	print("test_examples")
#	print(test_examples)
#	test_examples=1001
	
	x = tf.placeholder(tf.float32, [None, 224, 224, 3])
	y = tf.placeholder(tf.float32, [None, 1000])

	_, pred, dim = alexnet.classifier(x, 1.0)
	print("shape of pred:{}".format(pred.shape))

	# calculate the average precision of the crops of the image
	avg_prediction = tf.div(tf.reduce_sum(pred, 0), k_patches)
	print("shape of avg_prediction:{}".format(avg_prediction.shape))

	# accuracy
	top1_correct = tf.equal(tf.argmax(avg_prediction, 0), tf.argmax(y, 1))
	top1_accuracy = tf.reduce_mean(tf.cast(top1_correct, tf.float32))

	topk_correct = tf.nn.in_top_k(tf.stack([avg_prediction]), tf.argmax(y, 1), k=top_k)
	topk_accuracy = tf.reduce_mean(tf.cast(topk_correct, tf.float32))

	saver = tf.train.Saver()

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
#	sess = tf.Session(config=config)
#	with tf.Session(config=tf.ConfigProto(gpu_options.allow_growth=True)) as sess:
	with tf.Session(config=config) as sess:
		saver.restore(sess, os.path.join(ckpt_path, 'alexnet-cnn.ckpt'))

		total_top1_accuracy = 0.
		total_topk_accuracy = 0.

		count_top1=[0,0,0,0]
		count_topk=[0,0,0,0]
		for i in range(test_examples):
			# taking a few patches from an image
#			image_patches = tu.read_k_patches(os.path.join(imagenet_path, 'ILSVRC2012_img_val', test_images[i]), k_patches)
#			image_patches_1 = tu.read_k_patches(os.path.join(imagenet_path, 'ILSVRC2012_img_val', test_images[i]), 1, False)
			image_patches_1_fix = tu.read_k_patches(os.path.join(imagenet_path, 'ILSVRC2012_img_val', test_images[i]), 1, True)
#			image_patches_5 = tu.read_k_patches(os.path.join(imagenet_path, 'ILSVRC2012_img_val', test_images[i]), 5)			
#			print("len(image_patches):{}".format(len(image_patches)))
			label = test_labels[i]

#			avg_p_1=sess.run(avg_prediction, feed_dict={x: image_patches_1})
#			avg_p_1_fix=sess.run(avg_prediction, feed_dict={x: image_patches_1_fix})
#			avg_p_5=sess.run(avg_prediction, feed_dict={x: image_patches_5})
#			print("avg_prediction")
#			print("shape of avg_prediction:{}".format(avg_p.shape))
#			avg_p_1_reverse=np.argsort(avg_p_1)[::-1]
#			avg_p_5_reverse=np.argsort(avg_p_5)[::-1]
#			print("avg_prediction when patch 1")
#			for j in range(5):
#				print("top {} :{}".format(j+1, avg_p_1_reverse[j]))
#			print("avg_prediction when patch 5")
#			for j in range(5):
#				print("top {} :{}".format(j+1, avg_p_5_reverse[j]))
#			test_label_reverse=np.argsort(label)[::-1]
#			print("label:{}".format(test_label_reverse[0]))
#			print(avg_p)
#			print(avg_p[0])
#			print(avg_p[1])
			top1_a, topk_a, top1_c, topk_c = sess.run([top1_accuracy, topk_accuracy, top1_correct, topk_correct], feed_dict={x: image_patches_1_fix, y: [label]})
#			top1_a_f, topk_a_f, top1_c_f, topk_c_f = sess.run([top1_accuracy, topk_accuracy, top1_correct, topk_correct], feed_dict={x: image_patches_1_fix, y: [label]})
#			top1_a, topk_a, top1_c, topk_c = sess.run([top1_accuracy, topk_accuracy, top1_correct, topk_correct], feed_dict={x: image_patches_5, y: [label]})
#			top1_a, topk_a, top1_c, topk_c = sess.run([top1_accuracy, topk_accuracy, top1_correct, topk_correct], feed_dict={x: image_patches, y: [label]})
#			if ((top1_c==False) and (top1_c_f==False)):
#				count_top1[0]+=1
#			elif ((top1_c==True) and (top1_c_f==True)):
#				count_top1[1]+=1
#			elif ((top1_c==True) and (top1_c_f==False)):
#				count_top1[2]+=1
#			else:
#				count_top1[3]+=1

#			if ((topk_c==False) and (topk_c_f==False)):
#				count_topk[0]+=1
#			elif ((topk_c==True) and (topk_c_f==True)):
#				count_topk[1]+=1
#			elif ((topk_c==True) and (topk_c_f==False)):
#				count_topk[2]+=1
#			else:
#				count_topk[3]+=1
#			print("top1_correct when patch 5 in image {}:{}".format(i, top1_c))
#			print("topk_correct when patch 5 in image {}:{}".format(i, topk_c))
			total_top1_accuracy += top1_a
			total_topk_accuracy += topk_a

			if i % display_step == 0:
				print ('Examples done: {:5d}/{} ---- Top-1: {:.4f} -- Top-{}: {:.4f}, dim={}'.format(i + 1, test_examples, total_top1_accuracy / (i + 1), top_k, total_topk_accuracy / (i + 1), dim))
		
		print ('---- Final accuracy ----')
		print ('Top-1: {:.4f} -- Top-{}: {:.4f}'.format(total_top1_accuracy / test_examples, top_k, total_topk_accuracy / test_examples))
		print ('Top-1 error rate: {:.4f} -- Top-{} error rate: {:.4f}'.format(1 - (total_top1_accuracy / test_examples), top_k, 1 - (total_topk_accuracy / test_examples)))
#		print("top1")
#		print("both false:{}, both true:{}, rondom true:{}, fix true:{}".format(count_top1[0], count_top1[1], count_top1[2], count_top1[3]))
#		print("topk")
#		print("both false:{}, both true:{}, random true:{}, fix true:{}".format(count_topk[0], count_topk[1], count_topk[2], count_topk[3]))


if __name__ == '__main__':
	TOP_K = 5
	K_PATCHES = 1
	DISPLAY_STEP = 10
	IMAGENET_PATH = '/home/h-iwasaki/data/ILSVRC2012'
	CKPT_PATH = 'ckpt-128_norm_pool_random'

	test( 
		TOP_K, 
		K_PATCHES, 
		DISPLAY_STEP,
		IMAGENET_PATH,
		CKPT_PATH)


