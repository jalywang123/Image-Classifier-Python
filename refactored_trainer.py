import tensorflow as tf
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import time

#This is the training script that will be used for the classification of one vehicle and multiple vehicles 

#CODE THAT SPEEDS UP TRAINING BY 37.5 PERCENT
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

####################################################
#First we define some hyperparameters for our model#
####################################################
#We define our session
sess = tf.InteractiveSession()
#where we store and restore our model's weights
model_path="D:/classifier weights/weights/weights"
#where the tensorboard information is dumped
tensorboard_path="./tensorboard/9_convs_per_conv_then_9_convs/32_filters"
#where we fetch our training and validation data
tfrecords_file="D:/classifier data/data/original+reversed+flipped/train/train.tfrecords"
tfrecords_validation_file="D:/classifier data/data/original+reversed+flipped/valid/valid.tfrecords"

#data that is used to jump start training for huge networks
# tfrecords_file="overfitting_data/train/train.tfrecords"
# tfrecords_validation_file="overfitting_data/valid/valid.tfrecords"

#convolution parameters
num_first_convolutions=32
num_second_convolutions=32
num_third_convolutions=32
num_fourth_convolutions=32

num_first_fully=512
num_second_fully=256
#Input sizes
input_height=76
input_width=256
#separated image widths
uncorrelated_input_width=128
#After a certain number of iterations we save the model
weight_saver=500
#Batch sizes
batch_size=20
validation_batch=20
#Number of training images
capacity=20
#we do not use dropout for our validation
keep_probability=0.5
#Optimization scheme parameters
learning_rate=1e-5
momentum=0.9
#Dumping information to tensorboard or not
model_information=True
num_classes=2

#Number of iterations
info_dump=1000
num_iterations=3000000


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
    global plus_to_ok
    global ok_to_plus
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

##############################################################
#Function that computes convolutions for left and right input#
##############################################################
def convolution_computation(right_input,left_input,right_validation_input,left_validation_input):
	#right side
	right_conv1=conv2d(right_input,first_convolution)
	right_conv1_valid=conv2d(right_validation_input,first_convolution)
	right_out=tf.nn.bias_add(right_conv1,first_bias)
	right_out_valid=tf.nn.bias_add(right_conv1_valid,first_bias)
	right_out=tf.nn.relu(right_out)
	right_out_valid=tf.nn.relu(right_out_valid)
	
	right_conv2=conv2d(right_out,second_convolution)
	right_conv2_valid=conv2d(right_out_valid,second_convolution)
	right_out=tf.nn.bias_add(right_conv2,second_bias)
	right_out_valid=tf.nn.bias_add(right_conv2_valid,second_bias)
	right_out=tf.nn.relu(right_out)
	right_out_valid=tf.nn.relu(right_out_valid)

	right_conv3=conv2d(right_out,third_convolution)
	right_conv3_valid=conv2d(right_out_valid,third_convolution)
	right_out=tf.nn.bias_add(right_conv3,third_bias)
	right_out_valid=tf.nn.bias_add(right_conv3_valid,third_bias)
	right_out=tf.nn.relu(right_out)
	right_out_valid=tf.nn.relu(right_out_valid)

	right_conv4=conv2d(right_out,fourth_convolution)
	right_conv4_valid=conv2d(right_out_valid,fourth_convolution)
	right_out=tf.nn.bias_add(right_conv4,fourth_bias)
	right_out_valid=tf.nn.bias_add(right_conv4_valid,fourth_bias)
	right_out=tf.nn.relu(right_out)
	right_out_valid=tf.nn.relu(right_out_valid)

	right_conv5=conv2d(right_out,fifth_convolution)
	right_conv5_valid=conv2d(right_out_valid,fifth_convolution)
	right_out=tf.nn.bias_add(right_conv5,fifth_bias)
	right_out_valid=tf.nn.bias_add(right_conv5_valid,fifth_bias)
	right_out=tf.nn.relu(right_out)
	right_out_valid=tf.nn.relu(right_out_valid)
	
	right_conv6=conv2d(right_out,sixth_convolution)
	right_conv6_valid=conv2d(right_out_valid,sixth_convolution)
	right_out=tf.nn.bias_add(right_conv6,sixth_bias)
	right_out_valid=tf.nn.bias_add(right_conv6_valid,sixth_bias)
	right_out=tf.nn.relu(right_out)
	right_out_valid=tf.nn.relu(right_out_valid)
	
	right_conv7=conv2d(right_out,seventh_convolution)
	right_conv7_valid=conv2d(right_out_valid,seventh_convolution)
	right_out=tf.nn.bias_add(right_conv7,seventh_bias)
	right_out_valid=tf.nn.bias_add(right_conv7_valid,seventh_bias)
	right_out=tf.nn.relu(right_out)
	right_out_valid=tf.nn.relu(right_out_valid)
	
	right_conv8=conv2d(right_out,eighth_convolution)
	right_conv8_valid=conv2d(right_out_valid,eighth_convolution)
	right_out=tf.nn.bias_add(right_conv8,eighth_bias)
	right_out_valid=tf.nn.bias_add(right_conv8_valid,eighth_bias)
	right_out=tf.nn.relu(right_out)
	right_out_valid=tf.nn.relu(right_out_valid)	
	
	right_conv9=conv2d(right_out,ninth_convolution)
	right_conv9_valid=conv2d(right_out_valid,ninth_convolution)
	right_out=tf.nn.bias_add(right_conv9,ninth_bias)
	right_out_valid=tf.nn.bias_add(right_conv9_valid,ninth_bias)
	right_out=tf.nn.relu(right_out)
	right_out_valid=tf.nn.relu(right_out_valid)
	
	#left side
	left_conv1=conv2d(left_input,first_convolution)
	left_conv1_valid=conv2d(left_validation_input,first_convolution)
	left_out=tf.nn.bias_add(left_conv1,first_bias)
	left_out_valid=tf.nn.bias_add(left_conv1_valid,first_bias)
	left_out=tf.nn.relu(left_out)
	left_out_valid=tf.nn.relu(left_out_valid)
	
	left_conv2=conv2d(left_out,second_convolution)
	left_conv2_valid=conv2d(left_out_valid,second_convolution)
	left_out=tf.nn.bias_add(left_conv2,second_bias)
	left_out_valid=tf.nn.bias_add(left_conv2_valid,second_bias)
	left_out=tf.nn.relu(left_out)
	left_out_valid=tf.nn.relu(left_out_valid)

	left_conv3=conv2d(left_out,third_convolution)
	left_conv3_valid=conv2d(left_out_valid,third_convolution)
	left_out=tf.nn.bias_add(left_conv3,third_bias)
	left_out_valid=tf.nn.bias_add(left_conv3_valid,third_bias)
	left_out=tf.nn.relu(left_out)
	left_out_valid=tf.nn.relu(left_out_valid)
	
	left_conv4=conv2d(left_out,fourth_convolution)
	left_conv4_valid=conv2d(left_out_valid,fourth_convolution)
	left_out=tf.nn.bias_add(left_conv4,fourth_bias)
	left_out_valid=tf.nn.bias_add(left_conv4_valid,fourth_bias)
	left_out=tf.nn.relu(left_out)
	left_out_valid=tf.nn.relu(left_out_valid)

	left_conv5=conv2d(left_out,fifth_convolution)
	left_conv5_valid=conv2d(left_out_valid,fifth_convolution)
	left_out=tf.nn.bias_add(left_conv5,fifth_bias)
	left_out_valid=tf.nn.bias_add(left_conv5_valid,fifth_bias)
	left_out=tf.nn.relu(left_out)
	left_out_valid=tf.nn.relu(left_out_valid)	

	left_conv6=conv2d(left_out,sixth_convolution)
	left_conv6_valid=conv2d(left_out_valid,sixth_convolution)
	left_out=tf.nn.bias_add(left_conv6,sixth_bias)
	left_out_valid=tf.nn.bias_add(left_conv6_valid,sixth_bias)
	left_out=tf.nn.relu(left_out)
	left_out_valid=tf.nn.relu(left_out_valid)
	
	left_conv7=conv2d(left_out,seventh_convolution)
	left_conv7_valid=conv2d(left_out_valid,seventh_convolution)
	left_out=tf.nn.bias_add(left_conv7,seventh_bias)
	left_out_valid=tf.nn.bias_add(left_conv7_valid,seventh_bias)
	left_out=tf.nn.relu(left_out)
	left_out_valid=tf.nn.relu(left_out_valid)
	
	left_conv8=conv2d(left_out,eighth_convolution)
	left_conv8_valid=conv2d(left_out_valid,eighth_convolution)
	left_out=tf.nn.bias_add(left_conv8,eighth_bias)
	left_out_valid=tf.nn.bias_add(left_conv8_valid,eighth_bias)
	left_out=tf.nn.relu(left_out)
	left_out_valid=tf.nn.relu(left_out_valid)
	
	left_conv9=conv2d(left_out,ninth_convolution)
	left_conv9_valid=conv2d(left_out_valid,ninth_convolution)
	left_out=tf.nn.bias_add(left_conv9,ninth_bias)
	left_out_valid=tf.nn.bias_add(left_conv9_valid,ninth_bias)
	left_out=tf.nn.relu(left_out)
	left_out_valid=tf.nn.relu(left_out_valid)
	
	return(right_out,left_out,right_out_valid,left_out_valid)
	
	
	
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
    #Batch of size batch_size 
    images,labels=tf.train.shuffle_batch([resized_image,resized_label],
                                            batch_size=batch_size,
                                            capacity=capacity,
                                            num_threads=4,
                                            min_after_dequeue=0
                                        )
    return images, labels
with tf.name_scope('Input-producer'): 

	#Mean and Standard deviation for batch normalization
	mean = tf.constant([91.97232819, 81.13652039, 91.6187439], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
	variance=tf.constant([3352.71875,3293.62133789,3426.63623047], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_var')
	standard_deviation=tf.sqrt(variance)

	#Creation of a queue, working with num_epochs epochs so num_epochs*100 images, an image will basically be shown num_epochs times
	filename_queue=tf.train.string_input_producer([tfrecords_file],shuffle=True,num_epochs=None)
	filename_validation_queue=tf.train.string_input_producer([tfrecords_validation_file],shuffle=True,num_epochs=None)

	#Get an image batches
	image_batch,label_batch=read_and_decode(filename_queue,batch_size,capacity)
	validation_images,validation_labels=read_and_decode(filename_validation_queue,validation_batch,capacity)
	#Normalization of data
	image_batch=tf.divide((tf.subtract(image_batch,mean)),standard_deviation)
	validation_images=tf.divide((tf.subtract(validation_images,mean)),standard_deviation)
	##########################################################
	#We get the segmented data that will be used for training#
	##########################################################
	
	'''implementation of left and right image batch'''
	right_train_data=tf.placeholder(tf.float32,shape=[None,input_height,uncorrelated_input_width,3])
	left_train_data=tf.placeholder(tf.float32,shape=[None,input_height,uncorrelated_input_width,3])
	right_validation_data=tf.placeholder(tf.float32,shape=[None,input_height,uncorrelated_input_width,3])
	left_validation_data=tf.placeholder(tf.float32,shape=[None,input_height,uncorrelated_input_width,3])
	'''end of implementation of left and right image input'''

	train_label=tf.placeholder(tf.float32,shape=[None,num_classes])
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
	(first_convolution,first_bias,second_convolution,second_bias,third_convolution,third_bias,fourth_convolution,fourth_bias,
	fifth_convolution,fifth_bias,sixth_convolution,sixth_bias,seventh_convolution,seventh_bias,eighth_convolution,eighth_bias,ninth_convolution,ninth_bias)=create_convolutions(3, num_first_convolutions)
	#end of definition
	
	#convolution computation
	(right_out,left_out,right_out_valid,left_out_valid)=convolution_computation(right_train_data,left_train_data,right_validation_data,left_validation_data)
	#end convolution computation

	#Pooling layers
	right_pool1=max_pool2d(right_out,'right_first_pool')
	right_pool1_valid=max_pool2d(right_out_valid,'right_first_pool_valid')
	left_pool1=max_pool2d(left_out,'left_first_pool')
	left_pool1_valid=max_pool2d(left_out_valid,'left_first_pool_valid')
	#end Pooling layers

with tf.name_scope('Conv_Block_2'):

	#definition of the weights for the second block
	(first_convolution,first_bias,second_convolution,second_bias,third_convolution,third_bias,fourth_convolution,fourth_bias,
	fifth_convolution,fifth_bias,sixth_convolution,sixth_bias,seventh_convolution,seventh_bias,eighth_convolution,eighth_bias,ninth_convolution,ninth_bias)=create_convolutions(num_first_convolutions, num_second_convolutions)
	#end of definition

	#convolution computation
	(right_out,left_out,right_out_valid,left_out_valid)=convolution_computation(right_pool1,left_pool1,right_pool1_valid,left_pool1_valid)
	#end convolution computation
	
	#Pooling layers
	right_pool2=max_pool2d(right_out,'right_second_pool')
	right_pool2_valid=max_pool2d(right_out_valid,'right_second_pool_valid')
	left_pool2=max_pool2d(left_out,'left_second_pool')
	left_pool2_valid=max_pool2d(left_out_valid,'left_second_pool_valid')
	#end Pooling layers   

with tf.name_scope('Conv_Block_3'):

	#definition of the weights for the third block
	(first_convolution,first_bias,second_convolution,second_bias,third_convolution,third_bias,fourth_convolution,fourth_bias,
	fifth_convolution,fifth_bias,sixth_convolution,sixth_bias,seventh_convolution,seventh_bias,eighth_convolution,eighth_bias,ninth_convolution,ninth_bias)=create_convolutions(num_second_convolutions, num_third_convolutions)
	#end of definition
	
	#convolution computation
	(right_out,left_out,right_out_valid,left_out_valid)=convolution_computation(right_pool2,left_pool2,right_pool2_valid,left_pool2_valid)
	#end convolution computation
	
	#Pooling layers
	right_pool3=max_pool2d(right_out,'right_third_pool')
	right_pool3_valid=max_pool2d(right_out_valid,'right_third_pool_valid')
	left_pool3=max_pool2d(left_out,'left_third_pool')
	left_pool3_valid=max_pool2d(left_out_valid,'left_third_pool_valid')
	#end Pooling layers
	

with tf.name_scope('Conv_Block_4'):
	#definition of the weights for the third block
	(first_convolution,first_bias,second_convolution,second_bias,third_convolution,third_bias,fourth_convolution,fourth_bias,
	fifth_convolution,fifth_bias,sixth_convolution,sixth_bias,seventh_convolution,seventh_bias,eighth_convolution,eighth_bias,ninth_convolution,ninth_bias)=create_convolutions(num_third_convolutions, num_fourth_convolutions)
	#end of definition
	
	#convolution computation
	(right_out,left_out,right_out_valid,left_out_valid)=convolution_computation(right_pool3,left_pool3,right_pool3_valid,left_pool3_valid)
	#end convolution computation
	#Note that fourht convolution block has no pooling layer

with tf.name_scope('first_fully_connected_layer'):

	#definition of fully connected layer
	#we get the product of the shape of the last pool to flatten it, here 40960
	'''CONCAT'''
	pool4=tf.concat([left_out, right_out], 2)
	pool4_valid=tf.concat([left_out_valid, right_out_valid], 2)
	'''END OF CONCAT'''
	shape = int(np.prod(pool4.get_shape()[1:]))   
	first_fc=weight_variables([shape,num_first_fully],'weights')
	first_fc_bias=bias_variables([num_first_fully],'bias')
	pool4_flat=tf.reshape(pool4,[-1,shape])
	pool4_flat_valid=tf.reshape(pool4_valid,[-1,shape])
	fc1=tf.nn.bias_add(tf.matmul(pool4_flat,first_fc),first_fc_bias)
	fc1_valid=tf.nn.bias_add(tf.matmul(pool4_flat_valid,first_fc),first_fc_bias)
	fc1=tf.nn.relu(fc1)
	fc1_valid=tf.nn.relu(fc1_valid)
	#end of definition


with tf.name_scope('second_fully_connected_layer'):

    second_fc=weight_variables([num_first_fully,num_second_fully],'weights') 
    second_fc_bias=bias_variables([num_second_fully],'bias')
    fc2=tf.nn.bias_add(tf.matmul(fc1,second_fc),second_fc_bias)
    fc2_valid=tf.nn.bias_add(tf.matmul(fc1_valid,second_fc),second_fc_bias)
    fc2=tf.nn.relu(fc2)
    fc2_valid=tf.nn.relu(fc2_valid)
    #end of definition

with tf.name_scope('classifition_layer'):
	#definition of classification layer
	
	#dropout implementation
	fc2_drop = tf.nn.dropout(fc2, keep_probability)
	#No dropout for validation, but if you want uncomment the following line, and put the _valid for future references
	#fc1_drop_valid = tf.nn.dropout(fc1_valid, keep_probability)
	classifier=weight_variables([num_second_fully,num_classes],'weights')
	classifier_bias=bias_variables([num_classes],'bias')
	# output of the neural network before softmax
	classification=tf.nn.bias_add(tf.matmul(fc2_drop,classifier),classifier_bias)
	classification_valid=tf.nn.bias_add(tf.matmul(fc2_valid,classifier),classifier_bias)


######################################
#We define the training for the model#
###################################### 
with tf.name_scope('Gradient-computation'):
    #We define our loss   
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_label, logits=classification))

    #DO NOT CALL TRAIN STEP FOR THE VALIDATION SET
    #We define our optimization scheme ADAM
    train_step=tf.train.AdamOptimizer(learning_rate,momentum).minimize(cross_entropy)

    #We count the number or correct predictions
    correct_prediction=tf.equal(tf.argmax(classification,1), tf.argmax(train_label,1))
    correct_prediction_valid=tf.equal(tf.argmax(classification_valid,1), tf.argmax(validation_label,1))

    #We define our accuracy

    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    accuracy_valid=tf.reduce_mean(tf.cast(correct_prediction_valid,tf.float32))

############################
#TensordBoard Visualization#
############################

if(model_information):
    cross_view=tf.summary.scalar("cross_entropy",cross_entropy)
	
    accuracy_view=tf.summary.scalar("accuracy",accuracy)
    accuracy_valid_view=tf.summary.scalar("accuracy_validation",accuracy_valid)

    #merge all of the variables for visualization
    merged=tf.summary.merge_all()

    mixed_writer=tf.summary.FileWriter(tensorboard_path,sess.graph)

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

#For model saving and restoration, we keep at most 10000 files in our checkpoint
saver=tf.train.Saver(max_to_keep=100000)

print("\n"*2)
print("----------Tensorflow has been set----------")
print("We are using the train tfrecord file : %s"%(tfrecords_file))
print("We are using the validation tfrecord file : %s"%(tfrecords_validation_file))
print("Information for the tensorboard is being dumped in %s"%(tensorboard_path))
print("Weights are being saved in %s"%(model_path))
print("We will be running %d iterations"%(num_iterations))
print("We will be using a learning rate of : %f"%(learning_rate))
print("We will be using a dropout of : %f"%(keep_probability))
print("\n"*2)

#First we check if there is a model, if so, we restore it
if(os.path.isfile(model_path+".ckpt.meta")):
    print("")
    print( "We found a previous model")
    print("Model weights are being restored.....")
    saver.restore(sess,model_path+".ckpt")
    print("Model weights have been restored")
    print("\n"*2)
else:
    print("")
    print("No model weights were found....")
    print("")
    print("We generate a new model")
    print("\n"*2)
    
#We run our batch coordinator
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(coord=coord)

########################
#RUN THE NEURAL NETWORK#
########################
initial_start=time.time()
for i in range(num_iterations):
  
	initialize_counters()
	if(i%info_dump==0):
		start=time.time()
		
	input_data_batch,input_label_batch,validation_input,validation_output=sess.run([image_batch,label_batch,validation_images,validation_labels])

	(left_image_batch,right_image_batch)=np.split(input_data_batch,2,axis=2)
	(validation_left_image_batch,validation_right_image_batch)=np.split(validation_input,2,axis=2)


	#FEED THE RIGHT AND THE LEFT IMAGE BATCH BEFORE HAND 
	_,cross,acc,acc_val,classif,classif_valid,pred,summary=sess.run(
	[train_step,cross_entropy,accuracy,accuracy_valid,classification,classification_valid,correct_prediction,merged],
	feed_dict={
	train_label:input_label_batch,
	validation_label:validation_output,
	right_train_data:right_image_batch,
	left_train_data:left_image_batch,
	right_validation_data:validation_right_image_batch,
	left_validation_data:validation_left_image_batch
	})
		
	if((i+1)%info_dump==0 and i!=0):
		end=time.time()
		print("-------------------------------------------------------------")
		print("we called the model %d times"%(i+1))
		print("The current loss is : ",cross)
		print("The accuracy on the the training set is %d%%"%(100*acc))
		print("The accuracy on the validation set is %d%%"%(100*acc_val))
		print("this iteration took %d seconds"%((end-start)))
		print("-------------------------------------------------------------")

	if((i+1)%weight_saver==0):
		print("we are at iteration %d so we are going to save the model"%(i+1))
		print("model is being saved.....")
		save_path=saver.save(sess,model_path+"_iteration_%d.ckpt"%(i+1))
		print("model has been saved succesfully")

	# if(model_information):
		# mixed_writer.add_summary(summary,i)


######################
#We close everything #
######################
coord.request_stop()
coord.join(threads)
print("")
print("The Learning Process has been achieved.....")
print("")
print("It took %d hours "%(((end-initial_start)/60)/60))  
sess.close()
