import tensorflow as tf
import cv2
import os.path
import numpy as np
import time


#########################################
#TO BE REMOVED BEFORE TRAINING THE MODEL#
######################################### 

#This is the training script that will be used for the classification of one vehicle and multiple vehicles 

####################################################
#First we define some hyperparameters for our model#
####################################################
#We define our session
sess = tf.InteractiveSession()
model_path="./weights/first_weights.ckpt"
tensorboard_path="./tensorboard"
tfrecords_file="./data/train/train.tfrecords"
tfrecords_validation_file="./data/valid/valid.tfrecords"

#after a certain number of iterations we save the model
weight_saver=100
batch_size=20
validation_batch=150
#our number of training images
capacity=40000
#we do not use dropout for our validation
keep_probability=1
learning_rate=1e-4
momentum=0.9

#We give out all* of the information to the tensorboard(cross_entropy, accuracy, histogram of weights, histogram of distributions)
model_information=True
num_classes=2

#This defines the number of epochs we can run
#Our batch generator generates batch_size*capacity(number of images in our dataset)
#So for the training to not reach out of range num_iterations*batch_size=num_epochs*capacity(number of images in our dataset)
num_epochs=50
num_iterations=100000

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
image=cv2.imread("./data/%08d.tiff"%(4))
image=tf.cast(image,dtype=tf.float32)
blank_input=tf.reshape(image,[1,76,256,3])
label=label_ok
blank_output=tf.reshape(label,[1,2])
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
filename_queue=tf.train.string_input_producer([tfrecords_file],num_epochs=num_epochs)
filename_validation_queue=tf.train.string_input_producer([tfrecords_validation_file],num_epochs=(num_iterations/20))

#Get an image batch
image_batch,label_batch=read_and_decode(filename_queue,batch_size,capacity)
validation_images,validation_labels=read_and_decode(filename_validation_queue,validation_batch,3000)

##########################################################
#We get the segmented data that will be used for training#
##########################################################

train_data=tf.placeholder(tf.float32,shape=[None,76,256,3])
train_label=tf.placeholder(tf.float32,shape=[None,num_classes])
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
    
    pool1=max_pool2d(out,'first_pool')
    pool1_valid=max_pool2d(out_valid,'first_pool_valid')
    #end of definition
    
    #visualization information
    if(model_information):
        tf.summary.histogram('first_convoultion_histogram',first_convolution)
        tf.summary.histogram('second_convoultion_histogram',second_convolution)
        tf.summary.histogram('first_bias',first_bias)
        tf.summary.histogram('second_bias',second_bias)

with tf.name_scope('Conv_Block_2'):

    #definition of the weights for the second block
    first_convolution=weight_variables([3,3,16,32],'first_weights')
    
    first_bias=bias_variables([32],'first_biases')
    
    second_convolution=weight_variables([3,3,32,32],'second_weights')
    
    second_bias=bias_variables([32],'second_biases')
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
    
    pool2=max_pool2d(out,'second_pool')
    pool2_valid=max_pool2d(out_valid,'second_pool_valid')
    #end of definition
    
    #visualization information
    if(model_information):
        tf.summary.histogram('first_convoultion_histogram',first_convolution)
        tf.summary.histogram('second_convoultion_histogram',second_convolution)
        tf.summary.histogram('first_bias',first_bias)
        tf.summary.histogram('second_bias',second_bias)    

with tf.name_scope('Conv_Block_3'):

    #definition of the weights for the third block
    first_convolution=weight_variables([3,3,32,64],'first_weights')
    
    first_bias=bias_variables([64],'first_biases')
    
    second_convolution=weight_variables([3,3,64,64],'second_weights')
    
    second_bias=bias_variables([64],'second_biases')
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
    
    pool3=max_pool2d(out,'third_pool')    
    pool3_valid=max_pool2d(out_valid,'third_pool_valid')    
    #end of definition
    
    #visualization information
    if(model_information):
        tf.summary.histogram('first_convoultion_histogram',first_convolution)
        tf.summary.histogram('second_convoultion_histogram',second_convolution)
        tf.summary.histogram('first_bias',first_bias)
        tf.summary.histogram('second_bias',second_bias)


with tf.name_scope('first_fully_connected_layer'):

    #definition of fully connected layer
    #we get the product of the shape of the last pool to flatten it, here 40960
    
    shape = int(np.prod(pool3.get_shape()[1:]))   
    
    first_fc=weight_variables([shape,1024],'weights')
    
    first_fc_bias=bias_variables([1024],'bias')
    
    pool3_flat=tf.reshape(pool3,[-1,shape])
    pool3_flat_valid=tf.reshape(pool3_valid,[-1,shape])
    
    fc1=tf.nn.bias_add(tf.matmul(pool3_flat,first_fc),first_fc_bias)
    fc1_valid=tf.nn.bias_add(tf.matmul(pool3_flat_valid,first_fc),first_fc_bias)
    
    fc1=tf.nn.relu(fc1)
    fc1_valid=tf.nn.relu(fc1_valid)
    #end of definition
    
    #visualization information
    if(model_information):
        tf.summary.histogram('first_fully_connected',first_fc)
        tf.summary.histogram('first_fully_connected_bias',first_fc_bias)
    
with tf.name_scope('classifition_layer'):
    #definition of classification layer
    
    #dropout implementation
    fc1_drop = tf.nn.dropout(fc1, keep_probability)
    #No dropout for validation, but if you want uncomment the following line, and put the _valid for future references
    #fc1_drop_valid = tf.nn.dropout(fc1_valid, keep_probability)
    
    classifier=weight_variables([1024,num_classes],'weights')
    
    classifier_bias=bias_variables([num_classes],'bias')
    
    # output of the neural network before softmax
    classification=tf.nn.bias_add(tf.matmul(fc1_drop,classifier),classifier_bias)
    classification_valid=tf.nn.bias_add(tf.matmul(fc1_valid,classifier),classifier_bias)
    # soft=tf.nn.softmax(classification)
    # soft=100*soft
    # soft=tf.cast(soft,tf.uint8)
    #visualization information
    if(model_information):
        tf.summary.histogram('classifier',classifier)
        tf.summary.histogram('classifier_bias',classifier_bias)
        tf.summary.histogram('classification',classification)

        
        
######################################
#We define the training for the model#
###################################### 

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

#For model saving and restoration, we keep at most 100 files in our checkpoint
saver=tf.train.Saver(max_to_keep=100000)

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

########################
#RUN THE NEURAL NETWORK#
########################
initial_start=time.time()
for i in range(num_iterations):
	
    initialize_counters()
    if(i==0):
        input_data_batch,input_label_batch,validation_input,validation_output=sess.run([image_batch,label_batch,validation_images,validation_labels])
    else:
        input_data_batch,input_label_batch,unused_data,unused_label=sess.run([image_batch,label_batch,validation_images,validation_labels])
    # for debbuging purposes uncomment the following lines
    # so,_,cross,acc,classif,pred,summary=sess.run([soft,train_step,cross_entropy,accuracy,classification,correct_prediction,merged],feed_dict={train_data: input_data_batch,train_label:input_label_batch})
    #if we want to dump information for tensorboard
    start=time.time()
    if(model_information):
        if((i+1)%weight_saver==0):
            validation_feedback=True
            _,cross,acc,acc_val,classif,classif_valid,pred,summary=sess.run([train_step,cross_entropy,accuracy,accuracy_valid,classification,classification_valid,correct_prediction,merged],feed_dict={train_data: input_data_batch,train_label:input_label_batch, validation_data:validation_input ,validation_label:validation_output})
        else:  
            validation_feedback=False
            _,cross,acc,acc_val,classif,classif_valid,pred,summary=sess.run([train_step,cross_entropy,accuracy,accuracy_valid,classification,classification_valid,correct_prediction,merged],feed_dict={train_data: input_data_batch,train_label:input_label_batch, validation_data:blank_input.eval() ,validation_label:blank_output.eval()})
    else:
        _,cross,acc,acc_val,classif,classif_valid,pred=sess.run([train_step,cross_entropy,accuracy,accuracy_valid,classification,classification_valid,correct_prediction],feed_dict={train_data: input_data_batch,train_label:input_label_batch, validation_data:validation_input ,validation_label:validation_output })
    end=time.time()
    print("-------------------------------------------------------------")
    print("we called the model %d times"%(i+1))
    print("The current loss is : ",cross)
    print("The accuracy on the the training set is %d%%"%(100*acc))
    if((i+1)%weight_saver==0):
        print("The accuracy on the validation set is %d%%"%(100*acc_val))
    print("this iteration took %d seconds"%((end-start)))
    print("-------------------------------------------------------------")
    
    if(training_feedback):
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
        # print(so)
        # print(100*input_label_batch)
    
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
        print(" ")    
        
        
    
    
    #Here we will put the evaluation of the whole validation set on all of the model
    #Insert code here
    # if(100*acc_val>85 and (i+1)>5):
        # print("we found a model with a validation accuracy of %d"%(100*acc_val))
        # print("we are saving it.....")
        # save_path=saver.save(sess,"./best_weights/second_run_weights"+"_iteration_%d_accuracy_%d_valid_%d.ckpt"%(i+1,100*acc,100*acc_val))
        # print("model was succesfully saved")
    
    #Here we store the model every weight_saver iterations and we give out the accuracy
    if((i+1)%weight_saver==0):
        print("we are at iteration %d so we are going to save the model"%(i+1))
        print("model is being saved.....")
        save_path=saver.save(sess,model_path+"_iteration_%d_accuracy_%d_valid_%d"%(i+1,100*acc,100*acc_val))
        print("model has been saved succesfully")
    # if((i+1)%weight_saver==(weight_saver-1)):
        # print("we are at iteration %d so we are going to save the model"%(i+1))
        # print(" real model is being saved.....")
        # save_path=saver.save(sess,model_path+"_real_%d"%(i+1))
        # print("model has been saved succesfully")    
    if(model_information):
        mixed_writer.add_summary(summary,i)

      
coord.request_stop()
coord.join(threads)
#########################################
#We save the best snapchat of the model #
#########################################
print("")
print("model is being saved.....")
save_path=saver.save(sess,model_path)
print("model has been saved succesfully under ",model_path)

print("")
print("The training proces took %d hours "%(((end-initial_start)/60)/60))  
sess.close()

    
    









