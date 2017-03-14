import tensorflow as tf
import cv2
import os.path
import numpy as np


#########################################
#TO BE REMOVED BEFORE TRAINING THE MODEL#
######################################### 
start = tf.constant('')
set_start = tf.Session()
print(set_start.run(start))

for i in range(40):
    print("\n")
print("----------Tensorflow has been set----------")
for i in range(5):
    print("\n")

#This is the training script that will be used for the classification of one vehicle, multiple vehicles and motocycles

####################################################
#First we define some hyperparameters for our model#
####################################################
#We define our session

sess = tf.InteractiveSession()

model_path="./weights/first_weights.ckpt"

batch_size=300

keep_probability=0.5

model_information=False

num_classes=2

#################################################
#Function For data fetching from TFRecords files#
#################################################

tfrecords_file="train.tfrecords"












##########################################################
#We get the segmented data that will be used for training#
##########################################################

train_data=tf.placeholder(tf.float32,shape=[None,85,128,1])
train_label=tf.placeholder(tf.float32,shape=[None,num_classes])

# train_label=tf.to_float(tf.constant([[1.0,0.0,0.0]]))
# train_data=tf.ones([85,128], tf.float32)
# train_data = tf.reshape(train_data, [-1,85,128,1])



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
    
    #visualisation information
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
    
    #visualisation information
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
    
    #visualisation information
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
    
    #visualisation information
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
    
    #visualisation information
    if(model_information):
        tf.summary.histogram('classifier',classifier)
        tf.summary.histogram('classifier_bias',classifier_bias)
        tf.summary.histogram('classification',classification)

        
        
######################################
#We define the training for the model#
###################################### 

#We define our loss   
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_label, logits=classification))

#We define our optimization scheme
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#We count the number or correct predictions
correct_prediction=tf.equal(tf.argmax(classification,1), tf.argmax(train_label,1))

#We define our accuracy
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

############################
#TensordBoard Visualisation#
############################

if(model_information):
    cross_view=tf.summary.scalar("cross_entropy",cross_entropy)

    accuracy_view=tf.summary.scalar("accuracy",accuracy)

    #merge all of the variables for visualization
    merged=tf.summary.merge_all()

    mixed_writer=tf.summary.FileWriter('./tensorboard',sess.graph)

sess.run(tf.global_variables_initializer())

#For model saving and restoration
saver=tf.train.Saver()

#First we check if there is a model, if so, we restore it
if(os.path.isfile(model_path+".meta")):
    print("")
    print( "We found a previous model")
    print("Model is being restored.....")
    saver.restore(sess,model_path)
    print("Model has been restored")
else:
    print("")
    print("No model was found....")


########################
#RUN THE NEURAL NETWORK#
########################
for i in range(10):

    train_step.run()
    print(" ")
    print("the value of cross entropy is : ",sess.run(cross_entropy))
    print("the value given by the classifier before the softmax is ",sess.run(classification))
    print("we have an accuracy of %d%%"%(100*sess.run(accuracy)))
    print("we called the model %d times"%(i))
    
    if(model_information):
        summary=sess.run(merged)
    
        mixed_writer.add_summary(summary,i)
    

#########################################
#We save the best snapchat of the model #
#########################################
print("")
print("model is being saved.....")
save_path=saver.save(sess,"./weights/first_weights.ckpt")
print("model has been saved succesfully under ./weights/first_weights.ckpt")


sess.close()
# print("the shape of classification is : " ,classification.get_shape())
# print("the value of the label that was fed was ",sess.run(train_label))
# print("the value given by the classifier is ",sess.run(classification,feed_dict={keep_prob:1}))

# classification=tf.nn.softmax(classification)
# print("the value given by the classifier after the softmax is ",sess.run(classification,feed_dict={keep_prob:1}))
# print("this is value of the cross entropy :", sess.run(cross_entropy,feed_dict={keep_prob:1}))
# print(sess.run(cross_entropy,feed_dict={keep_prob:1}))
   
# print(classification)
# print("the classifiction layer has a shape of ",str(classification.get_shape()))  
#print(str(shape))
#print(pool3)
#print(sess.run(classification))
#pool1=tf.reshape(pool1,[2,5])
#print(sess.run(pool1))
#pool1=tf.reshape(image,[a[0],a[1]])
#image_encode=tf.image.encode_jpeg(image)
#print(pool1)
#cv2.imshow("image after tensor",pool1.eval())

# for i in range(10):
    # print("\n")
    
# cv2.waitKey(0)
# cv2.destroyAllWindows() 
# print('everything is ok')
   
    
    









