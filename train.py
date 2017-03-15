import tensorflow as tf
import cv2
import os.path
import numpy as np


#########################################
#TO BE REMOVED BEFORE TRAINING THE MODEL#
######################################### 

#This is the training script that will be used for the classification of one vehicle, multiple vehicles and motocycles

####################################################
#First we define some hyperparameters for our model#
####################################################
#We define our session

sess = tf.InteractiveSession()

color=False
animals=True

if(color):
    model_path="./color_weights/first_weights.ckpt"    
    tensorboard_path='./colorboard'
    tfrecords_file="color.tfrecords"
if(animals):
    model_path="./weights/first_weights.ckpt"
    tensorboard_path='./tensorboard'
    tfrecords_file="train.tfrecords"


batch_size=100

keep_probability=1.0

learning_rate=1e-6

#We give out all* of the information to the tensorboard(cross_entropy, accuracy, histogram of weights, histogram of distributions)
model_information=True

num_classes=2

#This defines the number of epochs we can run
#Our batch generator generates batch_size*(number of images in our dataset)
#So for the training to not reach out of range num_iterations*batch_size=num_epochs*(number of images in our dataset)
num_epochs=50
num_iterations=50

#################################################
#Function For data fetching from TFRecords files#
#################################################



#Function for read and decode from a TFRecords File 
def read_and_decode(filename_queue):

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
    resized_image=tf.reshape(image,[85,128,1])
    resized_label=tf.reshape(label,[2])
    
    #Creation of the batch
    #Batch of size batch_size from 100 samples that are randomized
    images,labels=tf.train.shuffle_batch([resized_image,resized_label],
                                            batch_size=batch_size,
                                            capacity=500,
                                            num_threads=4,
                                            min_after_dequeue=20
                                        )
    return images, labels
    
#Creation of a queue, working with num_epochs epochs so num_epochs*100 images, an image will basically be shown num_epochs times
filename_queue=tf.train.string_input_producer([tfrecords_file],num_epochs=num_epochs)

#Get an image batch
image_batch,label_batch=read_and_decode(filename_queue)

##########################################################
#We get the segmented data that will be used for training#
##########################################################

train_data=tf.placeholder(tf.float32,shape=[None,85,128,1])
train_label=tf.placeholder(tf.float32,shape=[None,num_classes])


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
    first_convolution=weight_variables([3,3,1,64],'first_weights')
    
    first_bias=bias_variables([64],'first_biases')
    
    second_convolution=weight_variables([3,3,64,64],'second_weights')

    second_bias=bias_variables([64],'second_biases')
    #end of definition
    
    #definition of first block computation
    conv1=conv2d(train_data,first_convolution)
    
    out=tf.nn.bias_add(conv1,first_bias)
    
    out=tf.nn.relu(out)
    
    conv2=conv2d(out,second_convolution)
    
    out=tf.nn.bias_add(conv2,second_bias)
    
    out=tf.nn.relu(out)
    
    pool1=max_pool2d(out,'first_pool')
    #end of definition
    
    #visualization information
    if(model_information):
        tf.summary.histogram('first_convoultion_histogram',first_convolution)
        tf.summary.histogram('second_convoultion_histogram',second_convolution)
        tf.summary.histogram('first_bias',first_bias)
        tf.summary.histogram('second_bias',second_bias)

with tf.name_scope('Conv_Block_2'):

    #definition of the weights for the second block
    first_convolution=weight_variables([3,3,64,128],'first_weights')
    
    first_bias=bias_variables([128],'first_biases')
    
    second_convolution=weight_variables([3,3,128,128],'second_weights')
    
    second_bias=bias_variables([128],'second_biases')
    #end of definition
    
    #definition of second block computation
    conv1=conv2d(pool1,first_convolution)
    
    out=tf.nn.bias_add(conv1,first_bias)
    
    out=tf.nn.relu(out)
    
    conv2=conv2d(out,second_convolution)
    
    out=tf.nn.bias_add(conv2,second_bias)
    
    out=tf.nn.relu(out)
    
    pool2=max_pool2d(out,'second_pool')
    #end of definition
    
    #visualization information
    if(model_information):
        tf.summary.histogram('first_convoultion_histogram',first_convolution)
        tf.summary.histogram('second_convoultion_histogram',second_convolution)
        tf.summary.histogram('first_bias',first_bias)
        tf.summary.histogram('second_bias',second_bias)    

with tf.name_scope('Conv_Block_3'):

    #definition of the weights for the third block
    first_convolution=weight_variables([3,3,128,256],'first_weights')
    
    first_bias=bias_variables([256],'first_biases')
    
    second_convolution=weight_variables([3,3,256,256],'second_weights')
    
    second_bias=bias_variables([256],'second_biases')
    #end of definition
    
    #definition of third block computation
    conv1=conv2d(pool2,first_convolution)
    
    out=tf.nn.bias_add(conv1,first_bias)
    
    out=tf.nn.relu(out)
    
    conv2=conv2d(out,second_convolution)
    
    out=tf.nn.bias_add(conv2,second_bias)
    
    out=tf.nn.relu(out)
    
    pool3=max_pool2d(out,'third_pool')    
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
    
    first_fc=weight_variables([shape,4096],'weights')
    
    first_fc_bias=bias_variables([4096],'bias')
    
    pool3_flat=tf.reshape(pool3,[-1,shape])
    
    fc1=tf.nn.bias_add(tf.matmul(pool3_flat,first_fc),first_fc_bias)
    
    fc1=tf.nn.relu(fc1)
    #end of definition
    
    #visualization information
    if(model_information):
        tf.summary.histogram('first_fully_connected',first_fc)
        tf.summary.histogram('first_fully_connected_bias',first_fc_bias)
    
with tf.name_scope('classifition_layer'):
    #definition of classification layer
    
    #dropout implementation
    fc1_drop = tf.nn.dropout(fc1, keep_probability)
    
    classifier=weight_variables([4096,num_classes],'weights')
    
    classifier_bias=bias_variables([num_classes],'bias')
    
    # output of the neural network before softmax
    classification=tf.nn.bias_add(tf.matmul(fc1_drop,classifier),classifier_bias)
    soft=tf.nn.softmax(classification)
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

#We define our optimization scheme ADAM
# train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#We define our optimization scheme SGD
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

#We count the number or correct predictions
correct_prediction=tf.equal(tf.argmax(classification,1), tf.argmax(train_label,1))

#We define our accuracy

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

############################
#TensordBoard Visualization#
############################

if(model_information):
    cross_view=tf.summary.scalar("cross_entropy",cross_entropy)

    accuracy_view=tf.summary.scalar("accuracy",accuracy)

    #merge all of the variables for visualization
    merged=tf.summary.merge_all()

    mixed_writer=tf.summary.FileWriter(tensorboard_path,sess.graph)

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

#For model saving and restoration
saver=tf.train.Saver()

for i in range(30):
    print("\n")
print("----------Tensorflow has been set----------")
for i in range(30):
    print("\n")

#First we check if there is a model, if so, we restore it
if(os.path.isfile(model_path+".meta")):
    print("")
    print( "We found a previous model")
    print("Model weights are being restored.....")
    saver.restore(sess,model_path)
    print("Model weights have been restored")
    print("")
else:
    print("")
    print("No model weights were found....")
    print("")
    print("We generate a new model")
    
#We run our batch coordinator
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(coord=coord)


########################
#RUN THE NEURAL NETWORK#
########################
for i in range(num_iterations):
	
    # input_data_batch=image_batch.eval()
    # input_label_batch=label_batch.eval()
    input_data_batch,input_label_batch=sess.run([image_batch,label_batch])
    # for debbuging purposes uncomment the following lines
    # so,_,cross,acc,classif,pred,summary=sess.run([soft,train_step,cross_entropy,accuracy,classification,correct_prediction,merged],feed_dict={train_data: input_data_batch,train_label:input_label_batch})
    #if we want to dump information for tensorboard
    if(model_information):
        so,_,cross,acc,classif,pred,summary=sess.run([soft,train_step,cross_entropy,accuracy,classification,correct_prediction,merged],feed_dict={train_data: input_data_batch,train_label:input_label_batch})
    else:
        so,_,cross,acc,classif,pred=sess.run([soft,train_step,cross_entropy,accuracy,classification,correct_prediction],feed_dict={train_data: input_data_batch,train_label:input_label_batch})
        
    print("-------------------------------------------------------------")
    print("we called the model %d times"%(i))
    print("The current loss is : ",cross)
    print("The accuracy of the model is %d%%"%(100*acc))
    # print("this is the ground truth that we are looking for",input_label_batch)
    # print("this is the softmax prediction : ",so)
    # print("these are the predictions that are correct",pred)
    print("-------------------------------------------------------------")
    print(" ")
    
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


sess.close()

    
    









