import tensorflow as tf
import cv2
import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import time
import sys



#Functions for model creation
def weight_variables(shape,identifier):
	initial=tf.truncated_normal(shape,dtype=tf.float32,stddev=0.1)
	return tf.Variable(initial,name=identifier)
 
def bias_variables(shape,identifier):
	initial=tf.constant(1.0,shape=shape, dtype=tf.float32)
	return tf.Variable(initial, trainable=True, name=identifier)

def conv2d(input,patch):
	return tf.nn.conv2d(input,patch,strides=[1,1,1,1],padding='SAME')

def max_pool2d(input,identifier):
	return tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name=identifier)

###########################################
#Function that creates convolution filters#
###########################################	
#num_input_filters is the number of input filters, for instance for the first convolutional block it is 3
#since we have 3 input channels (red, green and blue)
def create_convolutions(num_input_filters, num_output_filters):
	
	#definition of the weights of a convolutional block
	first_convolution=weight_variables([3,3,num_input_filters,num_output_filters],'first_weights')
	first_bias=bias_variables([num_output_filters],'first_biases')
	second_convolution=weight_variables([3,3,num_output_filters,num_output_filters],'second_weights')
	second_bias=bias_variables([num_output_filters],'second_biases')
	third_convolution=weight_variables([3,3,num_output_filters,num_output_filters],'third_weights')
	third_bias=bias_variables([num_output_filters],'third_biases')
	fourth_convolution=weight_variables([3,3,num_output_filters,num_output_filters],'fourth_weights')
	fourth_bias=bias_variables([num_output_filters],'fourth_biases')
	fifth_convolution=weight_variables([3,3,num_output_filters,num_output_filters],'fifth_weights')
	fifth_bias=bias_variables([num_output_filters],'fifth_biases')
	sixth_convolution=weight_variables([3,3,num_output_filters,num_output_filters],'sixth_weights')
	sixth_bias=bias_variables([num_output_filters],'sixth_biases')
	seventh_convolution=weight_variables([3,3,num_output_filters,num_output_filters],'seventh_weights')
	seventh_bias=bias_variables([num_output_filters],'seventh_biases')
	eighth_convolution=weight_variables([3,3,num_output_filters,num_output_filters],'eighth_weights')
	eighth_bias=bias_variables([num_output_filters],'eighth_biases')
	ninth_convolution=weight_variables([3,3,num_output_filters,num_output_filters],'ninth_weights')
	ninth_bias=bias_variables([num_output_filters],'ninth_biases')
	#end of definition
	return(first_convolution,first_bias,second_convolution,second_bias,third_convolution,third_bias,fourth_convolution,fourth_bias,
	fifth_convolution,fifth_bias,sixth_convolution,sixth_bias,seventh_convolution,seventh_bias,eighth_convolution,eighth_bias,ninth_convolution,ninth_bias)



#This function is called to launch the model, only once at the begining, once called it no longer needs to be recalled
def launch_model():
	global sess
	global soft_valid
	global right_validation_data
	global left_validation_data
	
	sess = tf.InteractiveSession() 
	
	#give out the model path
	model="./validation_weights/weights.ckpt"

	#Definition of filters per convlution block
	num_first_convolutions=32
	num_second_convolutions=32
	num_third_convolutions=32
	num_fourth_convolutions=32
	
	#Number of neurones for the fully connected
	num_first_fully=512
	num_second_fully=256
	
	#width and height of the input images
	input_height=76
	input_width=256
	
	uncorrelated_input_width=128
	
	#mean and variance for input normalization 
	mean = tf.constant([91.97232819, 81.13652039, 91.6187439], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
	variance=tf.constant([3352.71875,3293.62133789,3426.63623047], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_var')
	
	standard_deviation=tf.sqrt(variance)
	keep_probability=1
	num_classes=2
	
	#scaler used to scale output probabilities, by default it is set at 0.5 to correct dropout scaling
	scaler=0.5
	
	right_validation_data=tf.placeholder(tf.float32,shape=[None,input_height,uncorrelated_input_width,3])
	left_validation_data=tf.placeholder(tf.float32,shape=[None,input_height,uncorrelated_input_width,3])
	
	#data normalization
	right_normalized=tf.divide((tf.subtract(right_validation_data,mean)),standard_deviation)
	left_normalized=tf.divide((tf.subtract(left_validation_data,mean)),standard_deviation)

	##############################################################
	#Function that computes convolutions for left and right input#
	##############################################################
	def convolution_computation(right_validation_input,left_validation_input):
		#right side
		right_conv1_valid=conv2d(right_validation_input,first_convolution)
		right_out_valid=tf.nn.bias_add(right_conv1_valid,first_bias)
		right_out_valid=tf.nn.relu(right_out_valid)
		
		right_conv2_valid=conv2d(right_out_valid,second_convolution)
		right_out_valid=tf.nn.bias_add(right_conv2_valid,second_bias)
		right_out_valid=tf.nn.relu(right_out_valid)

		right_conv3_valid=conv2d(right_out_valid,third_convolution)
		right_out_valid=tf.nn.bias_add(right_conv3_valid,third_bias)
		right_out_valid=tf.nn.relu(right_out_valid)

		right_conv4_valid=conv2d(right_out_valid,fourth_convolution)
		right_out_valid=tf.nn.bias_add(right_conv4_valid,fourth_bias)
		right_out_valid=tf.nn.relu(right_out_valid)

		right_conv5_valid=conv2d(right_out_valid,fifth_convolution)
		right_out_valid=tf.nn.bias_add(right_conv5_valid,fifth_bias)
		right_out_valid=tf.nn.relu(right_out_valid)
		
		right_conv6_valid=conv2d(right_out_valid,sixth_convolution)
		right_out_valid=tf.nn.bias_add(right_conv6_valid,sixth_bias)
		right_out_valid=tf.nn.relu(right_out_valid)
		
		right_conv7_valid=conv2d(right_out_valid,seventh_convolution)
		right_out_valid=tf.nn.bias_add(right_conv7_valid,seventh_bias)
		right_out_valid=tf.nn.relu(right_out_valid)
		
		right_conv8_valid=conv2d(right_out_valid,eighth_convolution)
		right_out_valid=tf.nn.bias_add(right_conv8_valid,eighth_bias)
		right_out_valid=tf.nn.relu(right_out_valid)	
		
		right_conv9_valid=conv2d(right_out_valid,ninth_convolution)
		right_out_valid=tf.nn.bias_add(right_conv9_valid,ninth_bias)
		right_out_valid=tf.nn.relu(right_out_valid)
		
		#left side
		left_conv1_valid=conv2d(left_validation_input,first_convolution)
		left_out_valid=tf.nn.bias_add(left_conv1_valid,first_bias)
		left_out_valid=tf.nn.relu(left_out_valid)
		
		left_conv2_valid=conv2d(left_out_valid,second_convolution)
		left_out_valid=tf.nn.bias_add(left_conv2_valid,second_bias)
		left_out_valid=tf.nn.relu(left_out_valid)

		left_conv3_valid=conv2d(left_out_valid,third_convolution)
		left_out_valid=tf.nn.bias_add(left_conv3_valid,third_bias)
		left_out_valid=tf.nn.relu(left_out_valid)
		
		left_conv4_valid=conv2d(left_out_valid,fourth_convolution)
		left_out_valid=tf.nn.bias_add(left_conv4_valid,fourth_bias)
		left_out_valid=tf.nn.relu(left_out_valid)

		left_conv5_valid=conv2d(left_out_valid,fifth_convolution)
		left_out_valid=tf.nn.bias_add(left_conv5_valid,fifth_bias)
		left_out_valid=tf.nn.relu(left_out_valid)	

		left_conv6_valid=conv2d(left_out_valid,sixth_convolution)
		left_out_valid=tf.nn.bias_add(left_conv6_valid,sixth_bias)
		left_out_valid=tf.nn.relu(left_out_valid)
		
		left_conv7_valid=conv2d(left_out_valid,seventh_convolution)
		left_out_valid=tf.nn.bias_add(left_conv7_valid,seventh_bias)
		left_out_valid=tf.nn.relu(left_out_valid)
		
		left_conv8_valid=conv2d(left_out_valid,eighth_convolution)
		left_out_valid=tf.nn.bias_add(left_conv8_valid,eighth_bias)
		left_out_valid=tf.nn.relu(left_out_valid)
		
		left_conv9_valid=conv2d(left_out_valid,ninth_convolution)
		left_out_valid=tf.nn.bias_add(left_conv9_valid,ninth_bias)
		left_out_valid=tf.nn.relu(left_out_valid)
		
		return(right_out_valid,left_out_valid)

	with tf.name_scope('Conv_Block_1'):
		#definition of the weights for the first block
		(first_convolution,first_bias,second_convolution,second_bias,third_convolution,third_bias,fourth_convolution,fourth_bias,
		fifth_convolution,fifth_bias,sixth_convolution,sixth_bias,seventh_convolution,seventh_bias,eighth_convolution,eighth_bias,ninth_convolution,ninth_bias)=create_convolutions(3, num_first_convolutions)
		#end of definition
		#convolution computation
		(right_out_valid,left_out_valid)=convolution_computation(right_normalized,left_normalized)
		#end convolution computation
		#Pooling layers
		right_pool1_valid=max_pool2d(right_out_valid,'right_first_pool_valid')
		left_pool1_valid=max_pool2d(left_out_valid,'left_first_pool_valid')
		#end Pooling layers

	with tf.name_scope('Conv_Block_2'):
		#definition of the weights for the second block
		(first_convolution,first_bias,second_convolution,second_bias,third_convolution,third_bias,fourth_convolution,fourth_bias,
		fifth_convolution,fifth_bias,sixth_convolution,sixth_bias,seventh_convolution,seventh_bias,eighth_convolution,eighth_bias,ninth_convolution,ninth_bias)=create_convolutions(num_first_convolutions, num_second_convolutions)
		#end of definition		
		#convolution computation
		(right_out_valid,left_out_valid)=convolution_computation(right_pool1_valid,left_pool1_valid)
		#end convolution computation		
		#Pooling layers
		right_pool2_valid=max_pool2d(right_out_valid,'right_second_pool_valid')
		left_pool2_valid=max_pool2d(left_out_valid,'left_second_pool_valid')
		#end Pooling layers

	with tf.name_scope('Conv_Block_3'):
		#definition of the weights for the third block
		(first_convolution,first_bias,second_convolution,second_bias,third_convolution,third_bias,fourth_convolution,fourth_bias,
		fifth_convolution,fifth_bias,sixth_convolution,sixth_bias,seventh_convolution,seventh_bias,eighth_convolution,eighth_bias,ninth_convolution,ninth_bias)=create_convolutions(num_second_convolutions, num_third_convolutions)
		#end of definition
		#convolution computation
		(right_out_valid,left_out_valid)=convolution_computation(right_pool2_valid,left_pool2_valid)
		#end convolution computation		
		#Pooling layers
		right_pool3_valid=max_pool2d(right_out_valid,'right_third_pool_valid')
		left_pool3_valid=max_pool2d(left_out_valid,'left_third_pool_valid')
		#end Pooling layers

	with tf.name_scope('Conv_Block_4'):

		#definition of the weights for the third block
		(first_convolution,first_bias,second_convolution,second_bias,third_convolution,third_bias,fourth_convolution,fourth_bias,
		fifth_convolution,fifth_bias,sixth_convolution,sixth_bias,seventh_convolution,seventh_bias,eighth_convolution,eighth_bias,ninth_convolution,ninth_bias)=create_convolutions(num_third_convolutions, num_fourth_convolutions)
		#end of definition		
		#convolution computation
		(right_out_valid,left_out_valid)=convolution_computation(right_pool3_valid,left_pool3_valid)
		#end convolution computation
		

	with tf.name_scope('first_fully_connected_layer'):

		#definition of fully connected layer
		#we get the product of the shape of the last pool to flatten it
		'''CONCAT'''
		pool4_valid=tf.concat([left_out_valid, right_out_valid], 2)
		'''END OF CONCAT'''
		shape = int(np.prod(pool4_valid.get_shape()[1:]))   
		first_fc=weight_variables([shape,num_first_fully],'weights')
		first_fc_bias=bias_variables([num_first_fully],'bias')
		pool4_flat_valid=tf.reshape(pool4_valid,[-1,shape])
		fc1_valid=tf.nn.bias_add(tf.matmul(pool4_flat_valid,first_fc),first_fc_bias)
		fc1_valid=tf.nn.relu(fc1_valid)
		
	with tf.name_scope('second_fully_connected_layer'):

		second_fc=weight_variables([num_first_fully,num_second_fully],'weights')
		second_fc_bias=bias_variables([num_second_fully],'bias')
		fc2_valid=tf.nn.bias_add(tf.matmul(fc1_valid,second_fc),second_fc_bias)
		fc2_valid=tf.nn.relu(fc2_valid)
		#end of definition
		

	with tf.name_scope('classifition_layer'):
		#definition of classification layer
		
		'''THIS MUST BE REMOVED'''
		#it is not needed since we keep each neuron for testing
		fc2_valid = tf.nn.dropout(fc2_valid, keep_probability)		
		classifier=weight_variables([num_second_fully,num_classes],'weights')		
		classifier_bias=bias_variables([num_classes],'bias')		
		# output of the neural network before softmax
		# classification_valid=tf.nn.bias_add(tf.matmul(fc2_valid,classifier),classifier_bias)
		# here we try to lower our classification for latter thresholding, it can also be viewed as a way of correcting dropout scaling that was not considered for the training phase
		classification_valid=tf.multiply(tf.nn.bias_add(tf.matmul(fc2_valid,classifier),classifier_bias),scaler)
		soft_valid=tf.nn.softmax(classification_valid)
		soft_valid=100*soft_valid
		soft_valid=tf.cast(soft_valid,tf.uint8)
	
	#Launch the session and therefore the model
	saver=tf.train.Saver()
	
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	print("")
	print("Neural Network Launched")
	print("")
	#load the weights
	if(os.path.isfile(model+".meta")):
		
		print("Loading Neural Network Weights.....")
		saver.restore(sess,model)
		print("Neural Network Weights Loaded")
		print("\n")
	else:
		print("")
		print("No weights were found....")
		print("")
		print("We generate a new model")

	

#This function is called to classify a given image	
def classify(image):	
	
	input=[]
	input.append(image)
	input=np.asarray(input)
	(validation_left_image_batch,validation_right_image_batch)=np.split(input,2,axis=2)   

	probabilities=sess.run([soft_valid],feed_dict={
		right_validation_data:validation_right_image_batch,
		left_validation_data:validation_left_image_batch
		})
		
	return(probabilities)

    