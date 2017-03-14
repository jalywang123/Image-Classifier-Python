import tensorflow as tf
import numpy as np 
import cv2

"""This script is used to generate TFRecord files that will generate a binary file in which we have all of the training image data"""

for s in range(40):
	print("")
    
#first thing we do is create a large list with all of the paths for the training files
filenames=["./data_dog/%08d.png"%(i) for i in range(1,51)]
filenames.extend(["./data_cat/%08d.png"%(k) for k in range(1,51)])

#Function that will be used to Bynarize our data
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#The Tfrecord file
tfrecords_file="train.tfrecords"
writer = tf.python_io.TFRecordWriter(tfrecords_file)

#We initialize a file counter to keep track of our process
file_counter=0

for image_path in filenames:

	file_counter=file_counter+1
    
    #open the grayscale* image, and generate it's one-hot vector label
	image=cv2.imread(image_path,0)
	string_label=image_path.split("/")[1]
    
	if(string_label=="data_dog"):
        #Will generate [1.0,0.0]
		label=np.zeros(2)
		np.put(label,0,1)
        
	if(string_label=="data_cat"):
        #Will generate [0.0,1.0]
		label=np.zeros(2)
		np.put(label,1,1)
        
    #Transform data to string(direct Bytes)
	image_raw=image.tostring()
	label_raw=label.tostring()
    
    #Template and Fill the features that have to be stored
	example=tf.train.Example(features=tf.train.Features(feature={
    'input':_bytes_feature(image_raw),
    'label':_bytes_feature(label_raw)}))
    
    #Write in the TFRecord file
	writer.write(example.SerializeToString())
	print("we have processed %d files"%(file_counter))
    
    
writer.close()

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
    #Reshape it since it has no shape yet
    resized_image=tf.reshape(image,[85,128])
    resized_label=tf.reshape(label,[1,2])
    
    #Creation of the batch
    #Batch of size 10 from 100 samples that are randomized
    images,labels=tf.train.shuffle_batch([resized_image,resized_label],
                                            batch_size=20,
                                            capacity=100,
                                            num_threads=4,
                                            min_after_dequeue=20
                                        )
    return images, labels

#Creation of a queue, working with 10 epochs so 10*100 images, an image will basically be shown 10 times
filename_queue=tf.train.string_input_producer([tfrecords_file],num_epochs=10)
#Get an image batch
image_batch,label_batch=read_and_decode(filename_queue)

init_op=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())


with tf.Session() as sess : 

    sess.run(init_op)
    
    #Create a coordinator for multi-threading
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)
    
    for i in range(11):
        image,label=sess.run([image_batch,label_batch])
        print("-------------the size of the batch is : ",image_batch.get_shape()[0])
        print("the shape of the first image is : ",image[0,:,:].shape)
        print("this is the current batch : %d"%(i))
        show=tf.reshape(image[0,:,:],[85,128])
        show2=tf.reshape(image[1,:,:],[85,128])
        show3=tf.reshape(image[2,:,:],[85,128])
        show4=tf.reshape(image[3,:,:],[85,128])
        show5=tf.reshape(image[4,:,:],[85,128])
        cv2.imshow("first image",show.eval())
        cv2.imshow("second image",show2.eval())
        cv2.imshow("third image",show3.eval())
        cv2.imshow("fourth image",show4.eval())
        cv2.imshow("fifth image",show5.eval())
        print("This is the first label that is produced : ", label[0])
        print("This is the second label that is produced : ", label[1])
        print("This is the third label that is produced : ", label[2])
        print("This is the fourth label that is produced : ", label[3])
        print("This is the fifth label that is produced : ", label[4])
        cv2.waitKey(0)

    coord.request_stop()
    coord.join(threads)
        
        
    
                        
    
    
    
    