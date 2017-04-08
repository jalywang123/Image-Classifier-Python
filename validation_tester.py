import tensorflow as tf
import cv2
import os.path
import numpy as np
import time


#########################################
#TO BE REMOVED BEFORE TRAINING THE MODEL#
######################################### 

#This is the training script that will be used for the classification of one vehicle, multiple vehicles and motocycles

####################################################
#First we define some hyperparameters for our model#
####################################################
#We define our session
sess = tf.InteractiveSession()
model_path="./validation_weights/first_weights.ckpt"
tfrecords_validation_file="../data/valid/valid.tfrecords"

validation_batch=300
capacity=3000
keep_probability=1


#We give out all* of the information to the tensorboard(cross_entropy, accuracy, histogram of weights, histogram of distributions)
num_classes=2

#This defines the number of epochs we can run
#Our batch generator generates batch_size*capacity(number of images in our dataset)
#So for the training to not reach out of range num_iterations*batch_size=num_epochs*capacity(number of images in our dataset)
num_iterations=10

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

#we get an image that will be used for validation feed dict
# image=cv2.imread("./%08d.tiff"%(4))
# image=tf.cast(image,dtype=tf.float32)
# blank_input=tf.reshape(image,[1,76,256,3])
# label=label_motos
# blank_output=tf.reshape(label,[1,3])
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
    resized_image=tf.reshape(image,[76,256,3])
    resized_label=tf.reshape(label,[2])
    
    #Creation of the batch
    #Batch of size batch_size from 100 samples that are randomized
    images,labels=tf.train.shuffle_batch([resized_image,resized_label],
                                            batch_size=batch_size,
                                            capacity=capacity,
                                            num_threads=4,
                                            min_after_dequeue=batch_size
                                        )
    return images, labels
    
#Creation of a queue, working with num_epochs epochs so num_epochs*100 images, an image will basically be shown num_epochs times
filename_validation_queue=tf.train.string_input_producer([tfrecords_validation_file],num_epochs=(num_iterations/10))

#Get an image batch
validation_images,validation_labels=read_and_decode(filename_validation_queue,validation_batch,3000)

##########################################################
#We get the segmented data that will be used for training#
##########################################################
validation_data=tf.placeholder(tf.float32,shape=[None,76,256,3])
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
    first_convolution=weight_variables([3,3,3,16],'first_weights')
    
    first_bias=bias_variables([16],'first_biases')
    
    second_convolution=weight_variables([3,3,16,16],'second_weights')

    second_bias=bias_variables([16],'second_biases')
    #end of definition
    
    # definition of first block computation for training and validation
    # train_data=tf.nn.batch_normalization(train_data,mean=78.805528,variance=3567.471480,offset=0.0,scale=1.0,variance_epsilon=0.0)
    # validation_data=tf.nn.batch_normalization(validation_data,mean=78.661817,variance=3534.406429,offset=0.0,scale=1.0,variance_epsilon=0.0)
    conv1_valid=conv2d(validation_data,first_convolution)
    
    out_valid=tf.nn.bias_add(conv1_valid,first_bias)
    
    out_valid=tf.nn.relu(out_valid)
    
    conv2_valid=conv2d(out_valid,second_convolution)
    
    out_valid=tf.nn.bias_add(conv2_valid,second_bias)
    
    out_valid=tf.nn.relu(out_valid)
    
    pool1_valid=max_pool2d(out_valid,'first_pool_valid')
    #end of definition
    

with tf.name_scope('Conv_Block_2'):

    #definition of the weights for the second block
    first_convolution=weight_variables([3,3,16,32],'first_weights')
    
    first_bias=bias_variables([32],'first_biases')
    
    second_convolution=weight_variables([3,3,32,32],'second_weights')
    
    second_bias=bias_variables([32],'second_biases')
    #end of definition
    
    #definition of second block computation
    conv1_valid=conv2d(pool1_valid,first_convolution)
    
    out_valid=tf.nn.bias_add(conv1_valid,first_bias)
    
    out_valid=tf.nn.relu(out_valid)
  
    conv2_valid=conv2d(out_valid,second_convolution)
    
    out_valid=tf.nn.bias_add(conv2_valid,second_bias)
    
    out_valid=tf.nn.relu(out_valid)
    
    pool2_valid=max_pool2d(out_valid,'second_pool_valid')
    #end of definition
    
    #visualization information

with tf.name_scope('Conv_Block_3'):

    #definition of the weights for the third block
    first_convolution=weight_variables([3,3,32,64],'first_weights')
    
    first_bias=bias_variables([64],'first_biases')
    
    second_convolution=weight_variables([3,3,64,64],'second_weights')
    
    second_bias=bias_variables([64],'second_biases')
    #end of definition
    
    #definition of third block computation
    conv1_valid=conv2d(pool2_valid,first_convolution)
    
    out_valid=tf.nn.bias_add(conv1_valid,first_bias)
    
    out_valid=tf.nn.relu(out_valid)
    
    conv2_valid=conv2d(out_valid,second_convolution)

    out_valid=tf.nn.bias_add(conv2_valid,second_bias)

    out_valid=tf.nn.relu(out_valid)
       
    pool3_valid=max_pool2d(out_valid,'third_pool_valid')    
    #end of definition
    


with tf.name_scope('first_fully_connected_layer'):

    #definition of fully connected layer
    #we get the product of the shape of the last pool to flatten it, here 40960
    
    shape = int(np.prod(pool3_valid.get_shape()[1:]))   
    
    first_fc=weight_variables([shape,1024],'weights')
    
    first_fc_bias=bias_variables([1024],'bias')
    
    pool3_flat_valid=tf.reshape(pool3_valid,[-1,shape])
    
    fc1_valid=tf.nn.bias_add(tf.matmul(pool3_flat_valid,first_fc),first_fc_bias)
    
    fc1_valid=tf.nn.relu(fc1_valid)
    #end of definition
    
with tf.name_scope('classifition_layer'):
    #definition of classification layer
    
    classifier=weight_variables([1024,num_classes],'weights')
    
    classifier_bias=bias_variables([num_classes],'bias')
    
    # output of the neural network before softmax
    classification_valid=tf.nn.bias_add(tf.matmul(fc1_valid,classifier),classifier_bias)
    soft_valid=tf.nn.softmax(classification_valid)
    soft_valid=100*soft_valid
    soft_valid=tf.cast(soft_valid,tf.uint8)
    #visualization information

        

#We count the number or correct predictions
correct_prediction_valid=tf.equal(tf.argmax(classification_valid,1), tf.argmax(validation_label,1))

#We define our accuracy

accuracy_valid=tf.reduce_mean(tf.cast(correct_prediction_valid,tf.float32))

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

saver=tf.train.Saver()

print("\n"*30)
print("----------Tensorflow has been set----------")
print("\n"*10)

#First we check if there is a model, if so, we restore it
if(os.path.isfile(model_path+".meta")):
    print("")
    print( "We found a previous model")
    print("Model weights are being restored.....")
    saver.restore(sess,model_path)
    print("Model weights have been restored")
    print("\n"*5)
else:
    print("")
    print("No model weights were found....")
    print("")
    print("We generate a new model")
    print("\n"*5)
    
#We run our batch coordinator
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(coord=coord)
final_accuracy=0
error_dump=False
########################
#RUN THE NEURAL NETWORK#
########################
initial_start=time.time()
for i in range(num_iterations):

    # initialize_counters()

    validation_input,validation_output=sess.run([validation_images,validation_labels])

    start=time.time()

    validation_feedback=True
    acc_val,classif_valid,so_valid=sess.run([accuracy_valid,classification_valid,soft_valid],feed_dict={validation_data:validation_input ,validation_label:validation_output})
    
    
    print("-------------------------------------------------------------")
    print("we called the model %d times"%(i+1))
    print("The accuracy on the validation set is %d%%"%(100*acc_val))
    
    
    if(validation_feedback):
    #We count the number of images in each class for training
        for c in range(validation_batch):
            if (np.array_equal(validation_output[c],label_ok)):
                counter_ok=counter_ok+1
                #ok class
                if (np.argmax(classif_valid[c],0)==0 and correct_ok_classified<40):
                    correct_ok_classified=correct_ok_classified+1
                    image=tf.reshape(validation_input[c,:,:,:],[76,256,3])
                    image=(tf.cast(image,dtype=tf.uint8))
                    cv2.imwrite("./perfect/ok/%d_OK_%d%%_PLUSIEURS_VEHICULES_%d%%.tiff"%(correct_ok_classified,so_valid[c][0],so_valid[c][1]),image.eval())
                    
                if (np.argmax(classif_valid[c],0)==1):
                    ok_to_plus=ok_to_plus+1
                    if(error_dump):
                        image=tf.reshape(validation_input[c,:,:,:],[76,256,3])
                        image=(tf.cast(image,dtype=tf.uint8))
                        cv2.imwrite("./errors/ok_to_plus/%d_OK_%d%%_PLUSIEURS_VEHICULES_%d%%.tiff"%(ok_to_plus,so_valid[c][0],so_valid[c][1]),image.eval())
                                      
           
            if (np.array_equal(validation_output[c],label_plus)):
                counter_plus=counter_plus+1
                if (np.argmax(classif_valid[c],0)==1 and correct_plus_classified<40):
                    correct_plus_classified=correct_plus_classified+1
                    image=tf.reshape(validation_input[c,:,:,:],[76,256,3])
                    image=(tf.cast(image,dtype=tf.uint8))
                    cv2.imwrite("./perfect/plus/%d_OK_%d%%_PLUSIEURS_VEHICULES_%d%%.tiff"%(correct_plus_classified,so_valid[c][0],so_valid[c][1]),image.eval())
                if (np.argmax(classif_valid[c],0)==0):
                    plus_to_ok=plus_to_ok+1
                    if(error_dump):
                        image=tf.reshape(validation_input[c,:,:,:],[76,256,3])
                        image=(tf.cast(image,dtype=tf.uint8))
                        cv2.imwrite("./errors/plus_to_ok/%d_OK_%d%%_PLUSIEURS_VEHICULES_%d%%.tiff"%(plus_to_ok,so_valid[c][0],so_valid[c][1]),image.eval())
    final_accuracy=final_accuracy+(100*acc_val)
    end=time.time()
    print("this iteration took %d seconds"%((end-start)))
    print("-------------------------------------------------------------")
    if(validation_feedback and (i+1)==num_iterations):
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
        print(" ")
        print(" ")    
        
        
    
    
    #Here we will put the evaluation of the whole validation set on all of the model

      
coord.request_stop()
coord.join(threads)
#########################################
#We save the best snapchat of the model #
#########################################
print("")
print("Final accuracy is %d%%"%((final_accuracy)/10))
print("The validation computation proces took %d minutes "%(((end-initial_start)/60)))  
sess.close()

    
    









