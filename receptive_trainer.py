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

small_images=True
if(small_images):
	#where we store and restore our model's weights
	model_path="D:\prototype/weights"
	#where the tensorboard information is dumped
	tensorboard_path="./tensorboard/prototype"
	#where we fetch our training and validation data
	tfrecords_file="./data_128/train/train.tfrecords"
	tfrecords_validation_file="./data_128/valid/valid.tfrecords"
	#convolution parameters
	num_first_convolutions=16
	num_second_convolutions=32
	num_third_convolutions=32
	num_first_fully=512
	num_second_fully=256
	#Input sizes
	input_height=76
	input_width=256
	#After a certain number of iterations we save the model
	weight_saver=5000
	#Batch sizes
	batch_size=20
	validation_batch=20
	#Number of training images
	capacity=20
	#we do not use dropout for our validation
	keep_probability=0.5
	#Optimization scheme parameters
	learning_rate=1e-4
	momentum=0.9
	#Dumping information to tensorboard or not
	model_information=True
	num_classes=2
else:
    #where we store and restore our model's weights
    model_path="D:\weights_256_170/weights"
    #where the tensorboard information is dumped
    tensorboard_path="./tensorboard"
    #where we fetch our training and validation data
    tfrecords_file="./data/train/train.tfrecords"
    tfrecords_validation_file="./data/valid/valid.tfrecords"
    #convolution parameters
    num_first_convolutions=16
    num_second_convolutions=16
    num_third_convolutions=32
    num_fourth_convolutions=32
    num_first_fully=1024
    num_second_fully=512
    #Input sizes
    input_height=170
    input_width=512
    #After a certain number of iterations we save the model
    weight_saver=50000
    #Batch sizes
    batch_size=20
    validation_batch=20
    #Number of training images
    capacity=20
    #we do not use dropout for our validation
    keep_probability=0.5
    #Optimization scheme parameters
    learning_rate=1e-4
    momentum=0.9
    #Dumping information to tensorboard or not
    model_information=True
    num_classes=2
    

#Number of iterations
info_dump=100
num_iterations=500000


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
    if(small_images):
        #Mean and Standard deviation for batch normalization
        mean = tf.constant([91.97232819, 81.13652039, 91.6187439], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        variance=tf.constant([3352.71875,3293.62133789,3426.63623047], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_var')
        standard_deviation=tf.sqrt(variance)
    else:
        #Mean and Standard deviation for batch normalization
        mean = tf.constant([77.00479126, 75.57216644, 76.80247498], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        variance=tf.constant([2436.91015625,2331.30224609,2513.8762207], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_var')
        standard_deviation=tf.sqrt(variance)
    
    
    #Creation of a queue, working with num_epochs epochs so num_epochs*100 images, an image will basically be shown num_epochs times
    filename_queue=tf.train.string_input_producer([tfrecords_file],shuffle=False,num_epochs=None)
    filename_validation_queue=tf.train.string_input_producer([tfrecords_validation_file],shuffle=False,num_epochs=None)

    #Get an image batches
    image_batch,label_batch=read_and_decode(filename_queue,batch_size,capacity)
    validation_images,validation_labels=read_and_decode(filename_validation_queue,validation_batch,capacity)
	#Normalization of data
    image_batch=tf.divide((tf.subtract(image_batch,mean)),standard_deviation)
    validation_images=tf.divide((tf.subtract(validation_images,mean)),standard_deviation)
    ##########################################################
    #We get the segmented data that will be used for training#
    ##########################################################

    train_data=tf.placeholder(tf.float32,shape=[None,input_height,input_width,3])
    train_label=tf.placeholder(tf.float32,shape=[None,num_classes])
    validation_data=tf.placeholder(tf.float32,shape=[None,input_height,input_width,3])
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
	first_convolution=weight_variables([3,3,3,num_first_convolutions],'first_weights')

	first_bias=bias_variables([num_first_convolutions],'first_biases')

	second_convolution=weight_variables([3,3,num_first_convolutions,num_first_convolutions],'second_weights')

	second_bias=bias_variables([num_first_convolutions],'second_biases')

	third_convolution=weight_variables([3,3,num_first_convolutions,num_first_convolutions],'third_weights')

	third_bias=bias_variables([num_first_convolutions],'third_biases')

	#end of definition

	conv1=conv2d(train_data,first_convolution)
	conv1_valid=conv2d(validation_data,first_convolution)

	out=tf.nn.bias_add(conv1,first_bias)
	out_valid=tf.nn.bias_add(conv1_valid,first_bias)

	out=tf.nn.relu(out)
	out_valid=tf.nn.relu(out_valid)

	conv2=conv2d(out,second_convolution)
	conv2_valid=conv2d(out_valid,second_convolution)

	out=tf.nn.bias_add(conv2,second_bias)
	out_valid=tf.nn.bias_add(conv2_valid,second_bias)

	out=tf.nn.relu(out)
	out_valid=tf.nn.relu(out_valid)

	conv3=conv2d(out,third_convolution)
	conv3_valid=conv2d(out_valid,third_convolution)

	out=tf.nn.bias_add(conv3,third_bias)
	out_valid=tf.nn.bias_add(conv3_valid,third_bias)

	out=tf.nn.relu(out)
	out_valid=tf.nn.relu(out_valid)

	pool1=max_pool2d(out,'first_pool')
	pool1_valid=max_pool2d(out_valid,'first_pool_valid')
	#end of definition

with tf.name_scope('Conv_Block_2'):

	#definition of the weights for the second block
	first_convolution=weight_variables([3,3,num_first_convolutions,num_second_convolutions],'first_weights')

	first_bias=bias_variables([num_second_convolutions],'first_biases')

	second_convolution=weight_variables([3,3,num_second_convolutions,num_second_convolutions],'second_weights')

	second_bias=bias_variables([num_second_convolutions],'second_biases')

	third_convolution=weight_variables([3,3,num_second_convolutions,num_second_convolutions],'third_weights')

	third_bias=bias_variables([num_second_convolutions],'third_biases')

	#end of definition

	#definition of second block computation
	conv1=conv2d(pool1,first_convolution)
	conv1_valid=conv2d(pool1_valid,first_convolution)

	out=tf.nn.bias_add(conv1,first_bias)
	out_valid=tf.nn.bias_add(conv1_valid,first_bias)

	out=tf.nn.relu(out)
	out_valid=tf.nn.relu(out_valid)

	conv2=conv2d(out,second_convolution)
	conv2_valid=conv2d(out_valid,second_convolution)

	out=tf.nn.bias_add(conv2,second_bias)
	out_valid=tf.nn.bias_add(conv2_valid,second_bias)

	out=tf.nn.relu(out)
	out_valid=tf.nn.relu(out_valid)

	conv3=conv2d(out,third_convolution)
	conv3_valid=conv2d(out_valid,third_convolution)

	out=tf.nn.bias_add(conv3,third_bias)
	out_valid=tf.nn.bias_add(conv3_valid,third_bias)

	out=tf.nn.relu(out)
	out_valid=tf.nn.relu(out_valid)

	pool2=max_pool2d(out,'second_pool')
	pool2_valid=max_pool2d(out_valid,'second_pool_valid')
	#end of definition
    

with tf.name_scope('Conv_Block_3'):

	#definition of the weights for the third block
	first_convolution=weight_variables([3,3,num_second_convolutions,num_third_convolutions],'first_weights')

	first_bias=bias_variables([num_third_convolutions],'first_biases')

	second_convolution=weight_variables([3,3,num_third_convolutions,num_third_convolutions],'second_weights')

	second_bias=bias_variables([num_third_convolutions],'second_biases')

	third_convolution=weight_variables([3,3,num_third_convolutions,num_third_convolutions],'third_weights')

	third_bias=bias_variables([num_third_convolutions],'third_biases')
	
	#end of definition

	#definition of third block computation
	conv1=conv2d(pool2,first_convolution)
	conv1_valid=conv2d(pool2_valid,first_convolution)

	out=tf.nn.bias_add(conv1,first_bias)
	out_valid=tf.nn.bias_add(conv1_valid,first_bias)

	out=tf.nn.relu(out)
	out_valid=tf.nn.relu(out_valid)

	conv2=conv2d(out,second_convolution)
	conv2_valid=conv2d(out_valid,second_convolution)

	out=tf.nn.bias_add(conv2,second_bias)
	out_valid=tf.nn.bias_add(conv2_valid,second_bias)

	out=tf.nn.relu(out)
	out_valid=tf.nn.relu(out_valid)

	conv3=conv2d(out,third_convolution)
	conv3_valid=conv2d(out_valid,third_convolution)

	out=tf.nn.bias_add(conv3,third_bias)
	out_valid=tf.nn.bias_add(conv3_valid,third_bias)

	out=tf.nn.relu(out)
	out_valid=tf.nn.relu(out_valid)

	pool3=max_pool2d(out,'third_pool')    
	pool3_valid=max_pool2d(out_valid,'third_pool_valid')   

	#end of definition

if(small_images):
    with tf.name_scope('first_fully_connected_layer'):

        #definition of fully connected layer
        #we get the product of the shape of the last pool to flatten it, here 40960
        
        shape = int(np.prod(pool3.get_shape()[1:]))   
        
        first_fc=weight_variables([shape,num_first_fully],'weights')
        
        first_fc_bias=bias_variables([num_first_fully],'bias')
        
        pool3_flat=tf.reshape(pool3,[-1,shape])
        pool3_flat_valid=tf.reshape(pool3_valid,[-1,shape])
        
        fc1=tf.nn.bias_add(tf.matmul(pool3_flat,first_fc),first_fc_bias)
        fc1_valid=tf.nn.bias_add(tf.matmul(pool3_flat_valid,first_fc),first_fc_bias)
        
        fc1=tf.nn.relu(fc1)
        fc1_valid=tf.nn.relu(fc1_valid)
        #end of definition
else :

    with tf.name_scope('Conv_Block_4'):

		#definition of the weights for the third block
		first_convolution=weight_variables([3,3,num_third_convolutions,num_fourth_convolutions],'first_weights')

		first_bias=bias_variables([num_fourth_convolutions],'first_biases')

		second_convolution=weight_variables([3,3,num_fourth_convolutions,num_fourth_convolutions],'second_weights')

		second_bias=bias_variables([num_fourth_convolutions],'second_biases')

		third_convolution=weight_variables([3,3,num_fourth_convolutions,num_fourth_convolutions],'third_weights')

		third_bias=bias_variables([num_fourth_convolutions],'third_biases')
		#end of definition

		#definition of third block computation
		conv1=conv2d(pool3,first_convolution)
		conv1_valid=conv2d(pool3_valid,first_convolution)

		out=tf.nn.bias_add(conv1,first_bias)
		out_valid=tf.nn.bias_add(conv1_valid,first_bias)

		out=tf.nn.relu(out)
		out_valid=tf.nn.relu(out_valid)

		conv2=conv2d(out,second_convolution)
		conv2_valid=conv2d(out_valid,second_convolution)

		out=tf.nn.bias_add(conv2,second_bias)
		out_valid=tf.nn.bias_add(conv2_valid,second_bias)

		out=tf.nn.relu(out)
		out_valid=tf.nn.relu(out_valid)

		conv3=conv2d(out,third_convolution)
		conv3_valid=conv2d(out_valid,third_convolution)

		out=tf.nn.bias_add(conv3,third_bias)
		out_valid=tf.nn.bias_add(conv3_valid,third_bias)

		out=tf.nn.relu(out)
		out_valid=tf.nn.relu(out_valid)

		pool4=max_pool2d(out,'fourth_pool')
		pool4_valid=max_pool2d(out_valid,'fourth_pool_valid')    
		#end of definition



    with tf.name_scope('first_fully_connected_layer'):

        #definition of fully connected layer
        #we get the product of the shape of the last pool to flatten it, here 40960
        
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
	
    _,cross,acc,acc_val,classif,classif_valid,pred,summary=sess.run([train_step,cross_entropy,accuracy,accuracy_valid,classification,classification_valid,correct_prediction,merged],feed_dict={train_data: input_data_batch,train_label:input_label_batch, validation_data:validation_input ,validation_label:validation_output})

    if((i+1)%info_dump==0 and i!=0):
        end=time.time()
        print("-------------------------------------------------------------")
        print("we called the model %d times"%(i+1))
        print("The current loss is : ",cross)
        print("The accuracy on the the training set is %d%%"%(100*acc))
        print("The accuracy on the validation set is %d%%"%(100*acc_val))
        print("this iteration took %d seconds"%((end-start)))
        print("-------------------------------------------------------------")
    """if(training_feedback):
    #We count the number of images in each class for training
        for c in range(batch_size):
            if (np.array_equal(input_label_batch[c],label_ok)):
                counter_ok=counter_ok+1
                if (np.argmax(classif[c],0)==1):
                    ok_to_plus=ok_to_plus+1 

            if (np.array_equal(input_label_batch[c],label_plus)):
                counter_plus=counter_plus+1
                if (np.argmax(classif[c],0)==0):
                    plus_to_ok=plus_to_ok+1
        print("#############    TRAINING INFORMATION    #############")
        if(counter_ok!=0):
            print("the model was shown %d OK images"%(counter_ok))
        if(counter_plus!=0):
            print("the model was shown %d PLUSIEURS VEHICULES images"%(counter_plus))
            
        if((ok_to_plus!=0) and counter_ok!=0 ):
            print("%d%% of OK vehicles were misclassified"%(((ok_to_plus)/counter_ok)*100))
        if(ok_to_plus!=0):
            print("%d OK vehicles were classified as PLUSIEURS VEHICULES"%(ok_to_plus))
    
        if((plus_to_ok!=0) and counter_plus!=0 ):
            print("%d%% of PLUSIEURS VEHICULES vehicles were misclassified"%(((plus_to_ok)/counter_plus)*100)) 
        if(plus_to_ok!=0):
            print("%d PLUSIEURS VEHICULES vehicles were classified as OK"%(plus_to_ok))

        print("-------------------------------------------------------------")
        print(" ")
        print(" ")      
    if(validation_feedback):
    #We count the number of images in each class for training
        for c in range(validation_batch):
            if (np.array_equal(validation_output[c],label_ok)):
                counter_ok=counter_ok+1
                if (np.argmax(classif_valid[c],0)==1):
                    ok_to_plus=ok_to_plus+1
            if (np.array_equal(validation_output[c],label_plus)):
                counter_plus=counter_plus+1
                if (np.argmax(classif_valid[c],0)==0):
                    plus_to_ok=plus_to_ok+1
        print("#############    VALIDATION INFORMATION  #############")
        if(counter_ok!=0):
            print("the model was shown %d OK images"%(counter_ok))
        if(counter_plus!=0):
            print("the model was shown %d PLUSIEURS VEHICULES images"%(counter_plus))
        
        if((ok_to_plus!=0) and counter_ok!=0 ):
            print("%d%% of OK vehicles were misclassified"%(((ok_to_plus)/counter_ok)*100))
        if(ok_to_plus!=0):
            print("%d OK vehicles were classified as PLUSIEURS VEHICULES"%(ok_to_plus))

        if((plus_to_ok!=0) and counter_plus!=0 ):
            print("%d%% of PLUSIEURS VEHICULES vehicles were misclassified"%(((plus_to_ok)/counter_plus)*100))    
        if(plus_to_ok!=0):
            print("%d PLUSIEURS VEHICULES vehicles were classified as OK"%(plus_to_ok))

        print("-------------------------------------------------------------")
        print(" ")
        print(" ")  """  
    
    if((i+1)%weight_saver==0):
        print("we are at iteration %d so we are going to save the model"%(i+1))
        print("model is being saved.....")
        save_path=saver.save(sess,model_path+"_iteration_%d.ckpt"%(i+1))
        print("model has been saved succesfully")
    
    # if(model_information):
        # mixed_writer.add_summary(summary,i)
  
coord.request_stop()
coord.join(threads)
#########################################
#We save the best snapchat of the model #
#########################################
print("")
print("The Learning Process has been achieved.....")
print("")
print("It took %d hours "%(((end-initial_start)/60)/60))  
sess.close()

    
