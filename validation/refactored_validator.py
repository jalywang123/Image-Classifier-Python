import tensorflow as tf
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import time
import csv

#CODE THAT SPEEDS UP TRAINING BY 37.5 PERCENT
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

#########################################
#TO BE REMOVED BEFORE TRAINING THE MODEL#
######################################### 

#This is the training script that will be used for the classification of one vehicle, multiple vehicles and motocycles

####################################################
#First we define some hyperparameters for our model#
####################################################
#We define our session

ckpt=tf.train.get_checkpoint_state("D:/classifier weights/weights")
evaluation="uncorrelated_evaluations_128x76.txt"
num_first_convolutions=32
num_second_convolutions=32
num_third_convolutions=32
num_fourth_convolutions=32
num_first_fully=512
num_second_fully=256
tfrecords_validation_file="D:/classifier data/data/original/valid/valid.tfrecords"
input_height=76
input_width=256
uncorrelated_input_width=128
validation_batch=40
capacity=1
mean = tf.constant([91.97232819, 81.13652039, 91.6187439], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
variance=tf.constant([3352.71875,3293.62133789,3426.63623047], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_var')
standard_deviation=tf.sqrt(variance)

models=ckpt.all_model_checkpoint_paths[:]
models=["./validation_weights/weights.ckpt"]
num_models=len(models)
print("We are evaluating :",num_models)

num_threads=1
min_after_dequeue=0
shuffler=False
keep_probability=1

#We give out all* of the information to the tensorboard(cross_entropy, accuracy, histogram of weights, histogram of distributions)
num_classes=2

#This defines the number of epochs we can run
#Our batch generator generates batch_size*capacity(number of images in our dataset)
#So for the training to not reach out of range num_iterations*batch_size=num_epochs*capacity(number of images in our dataset)
num_iterations=200
#Mean and Standard deviation for batch normalization


################################
#Variables for model evaluation#
################################
training_feedback=False
if(not(training_feedback)):
    validation_feedback=True
else:
    validation_feedback=False
    
#Labels as one hot vectors
#1 0 
label_ok=np.zeros(2)
np.put(label_ok,0,1)
#0 1
label_plus=np.zeros(2)
np.put(label_plus,1,1)  

counter_ok=0
counter_plus=0
counter_miss_ok=0
counter_miss_plus=0
ok_to_plus=0
plus_to_ok=0
correct_ok_classified=0
correct_plus_classified=0
def initialize_counters():
    global counter_ok
    global counter_plus
    global counter_miss_ok
    global counter_miss_plus
    global ok_to_plus
    global plus_to_ok
    counter_ok=0
    counter_plus=0
    counter_miss_ok=0
    counter_miss_plus=0
    ok_to_plus=0
    plus_to_ok=0


###########################################
#Function that creates convolution filters#
###########################################	
#num_input_filters is the number of input filters, for instance for the first convolutional block it is 3
#since we have 3 input channels (red, green and blue)
def create_convolutions(num_input_filters, num_output_filters):
	
	#definition of the weights for the first block
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
	# return(first_convolution,first_bias,second_convolution,second_bias,third_convolution,third_bias,fourth_convolution,fourth_bias,
	# fifth_convolution,fifth_bias,sixth_convolution,sixth_bias,seventh_convolution,seventh_bias,eighth_convolution,eighth_bias)

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

	
#################################################
#Function For data fetching from TFRecords files#
#################################################
#Function for read and decode from a TFRecords File 
def read_and_decode(filename_queue,batch_size,capacity):

    reader =tf.TFRecordReader()
    _, serialized_data = reader.read(filename_queue)
    
    #Get the features from serialized data
    features =tf.parse_single_example(
                serialized_data,
                features={
                    'input':tf.FixedLenFeature([],tf.string),
                    'label':tf.FixedLenFeature([],tf.string)
                         })
    #Decode it from Bytes(Raw information) to the suitable type
    image=tf.decode_raw(features['input'],tf.uint8)
    label=tf.decode_raw(features['label'],tf.float64)
    image=tf.cast(image,dtype=tf.float32)
    #Reshape it since it has no shape yet
    resized_image=tf.reshape(image,[input_height,input_width,3])
    resized_label=tf.reshape(label,[2])
    
    #Creation of the batch
    #Batch of size batch_size from 100 samples that are randomized
    images,labels=tf.train.shuffle_batch([resized_image,resized_label],
                                            batch_size=batch_size,
                                            capacity=capacity,
                                            num_threads=num_threads,
                                            min_after_dequeue=min_after_dequeue
                                        )
    return images, labels



##########################################################
#We get the segmented data that will be used for training#
##########################################################
right_validation_data=tf.placeholder(tf.float32,shape=[None,input_height,uncorrelated_input_width,3])
left_validation_data=tf.placeholder(tf.float32,shape=[None,input_height,uncorrelated_input_width,3])
validation_label=tf.placeholder(tf.float32,shape=[None,num_classes])
#####################################################################
#Definition of convolution and max pooling weight matrix generators #
#####################################################################

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


    
#####################
#We define the model#
#####################

with tf.name_scope('Conv_Block_1'):

	#definition of the weights for the first block
	# (first_convolution,first_bias,second_convolution,second_bias,third_convolution,third_bias,fourth_convolution,fourth_bias,
	# fifth_convolution,fifth_bias,sixth_convolution,sixth_bias,seventh_convolution,seventh_bias,eighth_convolution,eighth_bias)=create_convolutions(3, num_first_convolutions)
	(first_convolution,first_bias,second_convolution,second_bias,third_convolution,third_bias,fourth_convolution,fourth_bias,
	fifth_convolution,fifth_bias,sixth_convolution,sixth_bias,seventh_convolution,seventh_bias,eighth_convolution,eighth_bias,ninth_convolution,ninth_bias)=create_convolutions(3, num_first_convolutions)
	#end of definition

	#convolution computation
	(right_out_valid,left_out_valid)=convolution_computation(right_validation_data,left_validation_data)
	#end convolution computation
	
	#Pooling layers
	right_pool1_valid=max_pool2d(right_out_valid,'right_first_pool_valid')
	left_pool1_valid=max_pool2d(left_out_valid,'left_first_pool_valid')
	#end Pooling layers

    

with tf.name_scope('Conv_Block_2'):

    #definition of the weights for the second block
	# (first_convolution,first_bias,second_convolution,second_bias,third_convolution,third_bias,fourth_convolution,fourth_bias,
	# fifth_convolution,fifth_bias,sixth_convolution,sixth_bias,seventh_convolution,seventh_bias,eighth_convolution,eighth_bias)=create_convolutions(num_first_convolutions, num_second_convolutions)
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
	# (first_convolution,first_bias,second_convolution,second_bias,third_convolution,third_bias,fourth_convolution,fourth_bias,
	# fifth_convolution,fifth_bias,sixth_convolution,sixth_bias,seventh_convolution,seventh_bias,eighth_convolution,eighth_bias)=create_convolutions(num_second_convolutions, num_third_convolutions)
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
	# (first_convolution,first_bias,second_convolution,second_bias,third_convolution,third_bias,fourth_convolution,fourth_bias,
	# fifth_convolution,fifth_bias,sixth_convolution,sixth_bias,seventh_convolution,seventh_bias,eighth_convolution,eighth_bias)=create_convolutions(num_third_convolutions, num_fourth_convolutions)
	(first_convolution,first_bias,second_convolution,second_bias,third_convolution,third_bias,fourth_convolution,fourth_bias,
	fifth_convolution,fifth_bias,sixth_convolution,sixth_bias,seventh_convolution,seventh_bias,eighth_convolution,eighth_bias,ninth_convolution,ninth_bias)=create_convolutions(num_third_convolutions, num_fourth_convolutions)
	#end of definition
	
	#convolution computation
	(right_out_valid,left_out_valid)=convolution_computation(right_pool3_valid,left_pool3_valid)
	#end convolution computation
	

with tf.name_scope('first_fully_connected_layer'):

	#definition of fully connected layer
	#we get the product of the shape of the last pool to flatten it, here 40960
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
	classification_valid=tf.multiply(tf.nn.bias_add(tf.matmul(fc2_valid,classifier),classifier_bias),0.5)
	soft_valid=tf.nn.softmax(classification_valid)
	soft_valid=100*soft_valid
	#if we want to have integers for our final percentages, uncomment this
	# soft_valid=tf.cast(soft_valid,tf.uint8)
	#visualization information


#We count the number or correct predictions
correct_prediction_valid=tf.equal(tf.argmax(classification_valid,1), tf.argmax(validation_label,1))
#We define our accuracy
accuracy_valid=tf.reduce_mean(tf.cast(correct_prediction_valid,tf.float32))


print("\n")
print("----------Tensorflow has been set----------")
print("\n")

#we are going to be testing every model that we had previously saved
saver=tf.train.Saver()
sess = tf.InteractiveSession()  
#Creation of a queue, working with num_epochs epochs so num_epochs*100 images, an image will basically be shown num_epochs times
filename_validation_queue=tf.train.string_input_producer([tfrecords_validation_file],shuffle=shuffler,num_epochs=None)

#Get an image batch
validation_images,validation_labels=read_and_decode(filename_validation_queue,validation_batch,capacity)
unnormalized_validation_images=validation_images
#Normalization of data
validation_images=tf.divide((tf.subtract(validation_images,mean)),standard_deviation)

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

#We run our batch coordinator
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(coord=coord)
limit=100

for model in models : 
	# if(len(models)>1):
		# print("we stop because there are too many models")
		# break
	#the following list's purpose is to track recognition rate and error rate, it will be used to evaluate model thresholding
	#it contains a pair that represents probablity of the majority class and the classification result, either well classified or not
	#example (0.89,1) represents an image that was corrctly classified due to 1 with a probability of 0.98
	#to turn it on, set roc_evaluation to True
	roc_evaluation=True
	if(roc_evaluation):
		limit=10000
	data=[]	
	
	target=open(evaluation,"a")

	#so we have a certain number of models that we are going to be testing.
	#First we check if there is a model, if so, we restore it
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
		

	final_accuracy=0
	error_dump=False
	########################
	#RUN THE NEURAL NETWORK#
	########################
	initial_start=time.time()
	initialize_counters()
	for i in range(num_iterations):
	  
		normallized_images,validation_output,validation_input=sess.run([validation_images,validation_labels,unnormalized_validation_images])
		
		start=time.time()
		
		(validation_left_image_batch,validation_right_image_batch)=np.split(normallized_images,2,axis=2)
		
		
		acc_val,classif_valid,so_valid=sess.run([accuracy_valid,classification_valid,soft_valid],feed_dict={
		right_validation_data:validation_right_image_batch,
		left_validation_data:validation_left_image_batch,
		validation_label:validation_output
		})

		
		if(validation_feedback):
		#We count the number of images in each class for training
			for c in range(validation_batch):
				if (np.array_equal(validation_output[c],label_ok)):
					counter_ok=counter_ok+1
					#ok class
					if (np.argmax(classif_valid[c],0)==0 and correct_ok_classified<limit):
						correct_ok_classified=correct_ok_classified+1
						if(roc_evaluation):
							#adding information to data
							data.append((so_valid[c][0],1))
						if(error_dump):
							image=tf.reshape(validation_input[c,:,:,:],[input_height,input_width,3])
							image=(tf.cast(image,dtype=tf.uint8))
							cv2.imwrite("./perfect/ok/%d_OK_%d%%_PLUSIEURS_VEHICULES_%d%%.tiff"%(correct_ok_classified,so_valid[c][0],so_valid[c][1]),image.eval())
						
					if (np.argmax(classif_valid[c],0)==1):
						ok_to_plus=ok_to_plus+1
						if(roc_evaluation):
							#adding information to data
							data.append((so_valid[c][1],0))
						if(error_dump):
							image=tf.reshape(validation_input[c,:,:,:],[input_height,input_width,3])
							image=(tf.cast(image,dtype=tf.uint8))
							cv2.imwrite("./errors/ok_to_plus/%d_OK_%d%%_PLUSIEURS_VEHICULES_%d%%.tiff"%(ok_to_plus,so_valid[c][0],so_valid[c][1]),image.eval())
										  
			   
				if (np.array_equal(validation_output[c],label_plus)):
					counter_plus=counter_plus+1
					if (np.argmax(classif_valid[c],0)==1 and correct_plus_classified<limit):
						correct_plus_classified=correct_plus_classified+1
						if(roc_evaluation):
							#adding information to data
							data.append((so_valid[c][1],1))
						if(error_dump):
							image=tf.reshape(validation_input[c,:,:,:],[input_height,input_width,3])
							image=(tf.cast(image,dtype=tf.uint8))
							cv2.imwrite("./perfect/plus/%d_OK_%d%%_PLUSIEURS_VEHICULES_%d%%.tiff"%(correct_plus_classified,so_valid[c][0],so_valid[c][1]),image.eval())
					if (np.argmax(classif_valid[c],0)==0):
						plus_to_ok=plus_to_ok+1
						if(roc_evaluation):
							#adding information to data
							data.append((so_valid[c][0],0))
						if(error_dump):
							image=tf.reshape(validation_input[c,:,:,:],[input_height,input_width,3])
							image=(tf.cast(image,dtype=tf.uint8))
							cv2.imwrite("./errors/plus_to_ok/%d_OK_%d%%_PLUSIEURS_VEHICULES_%d%%.tiff"%(plus_to_ok,so_valid[c][0],so_valid[c][1]),image.eval())
		final_accuracy=final_accuracy+(100*acc_val)
		end=time.time()
		if((100*(i+1)/num_iterations)%25==0) :
			print("-------------------------------------------------------------")
			print("Evaluation of model %s at %d%%"%(model,(100*(i+1)/num_iterations)))
			# print("this iteration took %d seconds"%((end-start)))
			print("-------------------------------------------------------------")
		if(validation_feedback and (i+1)==num_iterations):
		# if(True):
			print("#############   VALIDATION INFORMATION   #############")
			
			if(counter_ok!=0):
				print("the model was shown %d OK images"%(counter_ok))
			if(counter_plus!=0):
				print("the model was shown %d PLUSIEURS VEHICULES images"%(counter_plus))
			if(ok_to_plus!=0):
				print("%d OK vehicles were classified as PLUSIEURS VEHICULES"%(ok_to_plus))
			if(plus_to_ok!=0):
				print("%d PLUSIEURS VEHICULES vehicles were classified as OK"%(plus_to_ok))
		
			print("-------------------------------------------------------------")
	
	print(data)	
	if(roc_evaluation):
		with open('data.csv','w+',newline='') as csvfile:
			writer=csv.writer(csvfile,delimiter=";",quotechar="'")
			
			data.sort(key=lambda tup: tup[0])
			data.reverse()
			print("\n")
			writer.writerow(["threshold","confined error rate","confined recognition rate","global error rate","confined recognition rate"])
			
			for threshold in reversed(range(50,100)):
				passed=0
				error=0
				counter=0
				for i in range(len(data)):
					if(data[i][0]>=threshold):
						passed+=data[i][1]
						counter+=1
					if(data[i][0]<threshold):
						break
				error=counter-passed
				writer.writerow([threshold,'%.4f'%((counter-passed)/len(data)),'%.4f'%(passed/len(data)),'%.4f'%((counter-passed)/counter),'%.4f'%(passed/len(data))])
				# print("For threshold %d, we took into account %d images, %d vehicles were correctly classified and %d were not correctly classified "%(threshold,counter,passed,error))		
				# print("Therefore the error rate is %.4f and the recognition rate is %.4f\n"%(((error/len(data)) if counter>0 else 0),((passed/len(data)) if counter>0 else 0)))

	print("Final accuracy is %.2f%%"%((final_accuracy)/num_iterations))
	target.write("\n")
	target.write("Evaluation of model %s yields %.2f%% %d OK - %d PLUS"%(model,((final_accuracy)/num_iterations),counter_ok,counter_plus))
	target.write("\n")
	target.write("%d OK vehicles were classified as PLUSIEURS VEHICULES"%(ok_to_plus))
	target.write("\n")
	target.write("%d PLUSIEURS VEHICULES vehicles were classified as OK"%(plus_to_ok))
	target.write("\n")
	target.close()
	# print("The validation computation proces took %d minutes "%(((end-initial_start)/60))) 
	print("The validation computation process took %d seconds "%(((end-initial_start)))) 
	print("")

coord.request_stop()
coord.join(threads)    
    


    
    









