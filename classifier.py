import tensorflow as tf
import cv2
import os.path
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import time



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




#This function is called to launch the model, only once at the begining, once called it no longer needs to be recalled
def launch_model():
	
	#give out the model
	model="./validation_weights/weights_iteration_890.ckpt"
	
	#Define how many convolutions we are using per conv block
	three_convs_per_block=False
	four_convs_per_block=False
	
	if(four_convs_per_block):
		three_convs_per_block=True
	
	#Definition of filters per convlution block
	num_first_convolutions=16
	num_second_convolutions=32
	num_third_convolutions=32
	
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
	
	#scaler used to scale output probabilities
	scaler=1
	
	right_validation_data=tf.placeholder(tf.float32,shape=[None,input_height,uncorrelated_input_width,3])
	left_validation_data=tf.placeholder(tf.float32,shape=[None,input_height,uncorrelated_input_width,3])
	
	#data normalization
	right_validation_data=tf.divide((tf.subtract(right_validation_data,mean)),standard_deviation)
	left_validation_data=tf.divide((tf.subtract(left_validation_data,mean)),standard_deviation)

	
	with tf.name_scope('Conv_Block_1'):
		#definition of the weights for the first block
		first_convolution=weight_variables([3,3,3,num_first_convolutions],'first_weights')
		first_bias=bias_variables([num_first_convolutions],'first_biases')
		second_convolution=weight_variables([3,3,num_first_convolutions,num_first_convolutions],'second_weights')
		second_bias=bias_variables([num_first_convolutions],'second_biases')
		if(three_convs_per_block):
			third_convolution=weight_variables([3,3,num_first_convolutions,num_first_convolutions],'third_weights')
			third_bias=bias_variables([num_first_convolutions],'third_biases')
		if(four_convs_per_block):
			fourth_convolution=weight_variables([3,3,num_first_convolutions,num_first_convolutions],'fourth_weights')
			fourth_bias=bias_variables([num_first_convolutions],'fourth_biases')
		#end of definition

		# definition of first block computation for training and validation
		'''UNCORRELATED'''
		#right side
		right_conv1_valid=conv2d(right_validation_data,first_convolution)
		right_out_valid=tf.nn.bias_add(right_conv1_valid,first_bias)
		right_out_valid=tf.nn.relu(right_out_valid)
		
		right_conv2_valid=conv2d(right_out_valid,second_convolution)
		right_out_valid=tf.nn.bias_add(right_conv2_valid,second_bias)
		right_out_valid=tf.nn.relu(right_out_valid)
		
		if(three_convs_per_block):
			right_conv3_valid=conv2d(right_out_valid,third_convolution)
			right_out_valid=tf.nn.bias_add(right_conv3_valid,third_bias)
			right_out_valid=tf.nn.relu(right_out_valid)
			
		if(four_convs_per_block):
			right_conv4_valid=conv2d(right_out_valid,fourth_convolution)
			right_out_valid=tf.nn.bias_add(right_conv4_valid,fourth_bias)
			right_out_valid=tf.nn.relu(right_out_valid)
			
		right_pool1_valid=max_pool2d(right_out_valid,'right_first_pool_valid')
		
		#left side
		left_conv1_valid=conv2d(left_validation_data,first_convolution)
		left_out_valid=tf.nn.bias_add(left_conv1_valid,first_bias)
		left_out_valid=tf.nn.relu(left_out_valid)
		
		left_conv2_valid=conv2d(left_out_valid,second_convolution)
		left_out_valid=tf.nn.bias_add(left_conv2_valid,second_bias)
		left_out_valid=tf.nn.relu(left_out_valid)
		
		if(three_convs_per_block):
			left_conv3_valid=conv2d(left_out_valid,third_convolution)
			left_out_valid=tf.nn.bias_add(left_conv3_valid,third_bias)
			left_out_valid=tf.nn.relu(left_out_valid)
			
		if(four_convs_per_block):
			left_conv4_valid=conv2d(left_out_valid,fourth_convolution)
			left_out_valid=tf.nn.bias_add(left_conv4_valid,fourth_bias)
			left_out_valid=tf.nn.relu(left_out_valid)
			
		left_pool1_valid=max_pool2d(left_out_valid,'left_first_pool_valid')
		'''END UNCORRELATED'''
		#end of definition
		
	with tf.name_scope('Conv_Block_2'):

		#definition of the weights for the second block
		first_convolution=weight_variables([3,3,num_first_convolutions,num_second_convolutions],'first_weights')
		first_bias=bias_variables([num_second_convolutions],'first_biases')
		second_convolution=weight_variables([3,3,num_second_convolutions,num_second_convolutions],'second_weights')
		second_bias=bias_variables([num_second_convolutions],'second_biases')
		if(three_convs_per_block):
			third_convolution=weight_variables([3,3,num_second_convolutions,num_second_convolutions],'third_weights')
			third_bias=bias_variables([num_second_convolutions],'third_biases')
		if(four_convs_per_block):
			fourth_convolution=weight_variables([3,3,num_second_convolutions,num_second_convolutions],'fourth_weights')
			fourth_bias=bias_variables([num_second_convolutions],'fourth_biases')
		#end of definition
		
		#definition of second block computation

		'''UNCORRELATED'''
		#right side
		right_conv1_valid=conv2d(right_pool1_valid,first_convolution)
		right_out_valid=tf.nn.bias_add(right_conv1_valid,first_bias)
		right_out_valid=tf.nn.relu(right_out_valid)
		
		right_conv2_valid=conv2d(right_out_valid,second_convolution)
		right_out_valid=tf.nn.bias_add(right_conv2_valid,second_bias)
		right_out_valid=tf.nn.relu(right_out_valid)
		
		if(three_convs_per_block):
			right_conv3_valid=conv2d(right_out_valid,third_convolution)
			right_out_valid=tf.nn.bias_add(right_conv3_valid,third_bias)
			right_out_valid=tf.nn.relu(right_out_valid)
		
		if(four_convs_per_block):
			right_conv4_valid=conv2d(right_out_valid,fourth_convolution)
			right_out_valid=tf.nn.bias_add(right_conv4_valid,fourth_bias)
			right_out_valid=tf.nn.relu(right_out_valid)
			
		right_pool2_valid=max_pool2d(right_out_valid,'right_second_pool_valid')
		
		#left side
		left_conv1_valid=conv2d(left_pool1_valid,first_convolution)
		left_out_valid=tf.nn.bias_add(left_conv1_valid,first_bias)
		left_out_valid=tf.nn.relu(left_out_valid)
		
		left_conv2_valid=conv2d(left_out_valid,second_convolution)
		left_out_valid=tf.nn.bias_add(left_conv2_valid,second_bias)
		left_out_valid=tf.nn.relu(left_out_valid)
		
		if(three_convs_per_block):
			left_conv3_valid=conv2d(left_out_valid,third_convolution)
			left_out_valid=tf.nn.bias_add(left_conv3_valid,third_bias)
			left_out_valid=tf.nn.relu(left_out_valid)
			
		if(four_convs_per_block):
			left_conv4_valid=conv2d(left_out_valid,fourth_convolution)
			left_out_valid=tf.nn.bias_add(left_conv4_valid,fourth_bias)
			left_out_valid=tf.nn.relu(left_out_valid)
			
		left_pool2_valid=max_pool2d(left_out_valid,'left_second_pool_valid')
		'''END UNCORRELATED'''
		#end of definition
		
		#visualization information

	with tf.name_scope('Conv_Block_3'):

		#definition of the weights for the third block
		first_convolution=weight_variables([3,3,num_second_convolutions,num_third_convolutions],'first_weights')
		first_bias=bias_variables([num_third_convolutions],'first_biases')
		second_convolution=weight_variables([3,3,num_third_convolutions,num_third_convolutions],'second_weights')
		second_bias=bias_variables([num_third_convolutions],'second_biases')
		if(three_convs_per_block):
			third_convolution=weight_variables([3,3,num_third_convolutions,num_third_convolutions],'third_weights')
			third_bias=bias_variables([num_third_convolutions],'third_biases')
		if(four_convs_per_block):
			fourth_convolution=weight_variables([3,3,num_third_convolutions,num_third_convolutions],'fourth_weights')
			fourth_bias=bias_variables([num_third_convolutions],'fourth_biases')
		#end of definition

		#definition of third block computation
		'''UNCORRELATED'''
		#right side
		right_conv1_valid=conv2d(right_pool2_valid,first_convolution)
		right_out_valid=tf.nn.bias_add(right_conv1_valid,first_bias)
		right_out_valid=tf.nn.relu(right_out_valid)
		
		right_conv2_valid=conv2d(right_out_valid,second_convolution)
		right_out_valid=tf.nn.bias_add(right_conv2_valid,second_bias)
		right_out_valid=tf.nn.relu(right_out_valid)
		
		if(three_convs_per_block):
			right_conv3_valid=conv2d(right_out_valid,third_convolution)
			right_out_valid=tf.nn.bias_add(right_conv3_valid,third_bias)
			right_out_valid=tf.nn.relu(right_out_valid)
			
		if(four_convs_per_block):
			right_conv4_valid=conv2d(right_out_valid,fourth_convolution)
			right_out_valid=tf.nn.bias_add(right_conv4_valid,fourth_bias)
			right_out_valid=tf.nn.relu(right_out_valid)	
		
		right_pool3_valid=max_pool2d(right_out_valid,'right_third_pool_valid')
		
		#left side
		left_conv1_valid=conv2d(left_pool2_valid,first_convolution)
		left_out_valid=tf.nn.bias_add(left_conv1_valid,first_bias)
		left_out_valid=tf.nn.relu(left_out_valid)
		
		left_conv2_valid=conv2d(left_out_valid,second_convolution)
		left_out_valid=tf.nn.bias_add(left_conv2_valid,second_bias)
		left_out_valid=tf.nn.relu(left_out_valid)
		
		if(three_convs_per_block):
			left_conv3_valid=conv2d(left_out_valid,third_convolution)
			left_out_valid=tf.nn.bias_add(left_conv3_valid,third_bias)
			left_out_valid=tf.nn.relu(left_out_valid)
			
		if(four_convs_per_block):
			left_conv4_valid=conv2d(left_out_valid,fourth_convolution)
			left_out_valid=tf.nn.bias_add(left_conv4_valid,fourth_bias)
			left_out_valid=tf.nn.relu(left_out_valid)
			
		left_pool3_valid=max_pool2d(left_out_valid,'left_third_pool_valid')
		
		'''END UNCORRELATED'''
		#end of definition

	with tf.name_scope('first_fully_connected_layer'):

		#definition of fully connected layer
		#we get the product of the shape of the last pool to flatten it, here 40960
		pool3_valid=tf.concat([left_pool3_valid, right_pool3_valid], 2)
		
		shape = int(np.prod(pool3_valid.get_shape()[1:]))   
		
		first_fc=weight_variables([shape,num_first_fully],'weights')
		
		first_fc_bias=bias_variables([num_first_fully],'bias')
		
		pool3_flat_valid=tf.reshape(pool3_valid,[-1,shape])
		
		fc1_valid=tf.nn.bias_add(tf.matmul(pool3_flat_valid,first_fc),first_fc_bias)
		
		fc1_valid=tf.nn.relu(fc1_valid)
			#end of definition
	
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
		sess = tf.InteractiveSession() 
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		
		#load the weights
		if(os.path.isfile(model+".meta")):
			print("")
			print( "We found a previous model")
			print("Model weights are being restored.....")
			saver.restore(sess,model)
			print("Model weights have been restored")
		else:
			print("")
			print("No model weights were found....")
			print("")
			print("We generate a new model")



#This function is called to classify a given image	
def classify(image):	
	
	input_data_batch=sess.run([resized_image])
        # so=sess.run([soft],feed_dict={train_data: input_data_batch})
        # vote[i]=np.argmax(so)   
        # if(vote[i]==0):
            # votes_motos=votes_motos+1
        # if(vote[i]==1):
            # votes_ok=votes_ok+1    
        # if(vote[i]==2):
            # votes_plus=votes_plus+1
    # final_vote.append([votes_motos,votes_ok,votes_plus])
    # print(final_vote)
    # dec=np.argmax((final_vote))
    # classname=""
    # if(dec==0):
        # classname="MOTOS"
    # if(dec==1):
        # classname="OK"
    # if(dec==2):
        # classname="PLUSIEURS_VEHICULES"
    # sess.close()
    # return classname
    