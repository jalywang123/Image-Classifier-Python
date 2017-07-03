import numpy as np
import cv2
import os

#Here we will put all of the functions used for preprocessing
#First we get the image path, open the image, large-scale, rescale it and then cut out the lower banner 
#merge them if there are two of them either-wise just copy it twice 

def preproc(imagePath1, imagePath2=""):

	#Test the presence of the file or files
	presence_1=os.path.isfile(imagePath1)
	presence_2=os.path.isfile(imagePath2)
	
	if(not(presence_1) and not(presence_2)):
		print("None of the files exists or was found in the system")
	
	#get the images after testing the presence
	if(presence_1):	
		image_1=cv2.imread(imagePath1)
	if(presence_2):	
		image_2=cv2.imread(imagePath2)
		
	#get size for preprocessing
	size=image_1.shape
	
	#test to see we got the same image resolutions
	if(presence_2):
		size_2=image_2.shape
		if(size!=size_2):
			print(" Two different resolution images were given")
			return 0

	print(size)
	
	#Preprocessing of the images given their resolution
	if(size==(900,1392)):
		print("this is a 1392*900 image")
		#resize the image
		image_1=cv2.resize(image_1,(128,82))
		#crop the bottom
		crop_1=image_1[0:66,0:128]
		#add bottom padding
		stack=np.zeros((10,128,3),dtype=np.uint8)
		crop_1=np.vstack((crop_1,stack))
		
		#if there is a second image
		if(presence_2):
			image_2=cv2.resize(image_2,(128,82))
			crop_2=image_2[0:66,0:128]
			crop_2=np.vstack((crop_2,stack))
			
		#if we find the two images we merge them together, otherwise we duplicate the image
		if(presence_1 and presence_2):
			image_3=np.hstack((crop_1,crop_2))
			
		if(presence_1 and not(presence_2)):
			#take note of this 
			print("THE RESOLUTION 1392*900 YIELDS TWO IMAGES YET ONLY ONE WAS GIVEN")
			image_3=np.hstack((crop_1,crop_1))
			
	if(size==(930,1628)):
		print("this is a 1682*930 image")
		
		#resize the image
		image_1=cv2.resize(image_1,(128,73))
		#crop the bottom
		crop_1=image_1[0:57,0:128]
		#add bottom padding
		stack=np.zeros((19,128,3),dtype=np.uint8)
		crop_1=np.vstack((crop_1,stack))
		
		#if there is a second image
		if(presence_2):
			image_2=cv2.resize(image_2,(128,73))
			crop_2=image_2[0:57,0:128]
			crop_2=np.vstack((crop_2,stack))
		
		#if we find the two images we merge them together, otherwise we duplicate the image		
		if(presence_1 and presence_2):
			image_3=np.hstack((crop_1,crop_2))
			
		if(presence_1 and not(presence_2)):
			#take note of this 
			print("THE RESOLUTION 1682*930 YIELDS TWO IMAGES YET ONLY ONE WAS GIVEN")
			image_3=np.hstack((crop_1,crop_1))
			
		
	if(size==(1488,2240)):
		print("this is a 2240*1488 image")
		
		#resize the image
		image_1=cv2.resize(image_1,(128,85))
		#crop the bottom
		crop_1=image_1[0:72,0:128]
		#add bottom padding
		stack=np.zeros((4,128,3),dtype=np.uint8)
		crop_1=np.vstack((crop_1,stack))
		#meerge image
		image_3=np.hstack((crop_1,crop_1))
		
	if(size==(2136,3216)):
		print("this is a 3216*2136 image")
		
		#resize the image
		image_1=cv2.resize(image_1,(128,85))
		#crop the bottom
		crop_1=image_1[0:76,0:128]
		#meerge image
		image_3=np.hstack((crop_1,crop_1))
		
	if(size==(3000,4496)):
		print("this is a 4496*3000 image")
		
		#resize the image
		image_1=cv2.resize(image_1,(128,85))
		#crop the bottom
		crop_1=image_1[0:76,0:128]
		#meerge image
		image_3=np.hstack((crop_1,crop_1))
		
	if(size==(4000,6000)):
		print("this is a 6000*4000 image")
		#resize the image
		image_1=cv2.resize(image_1,(128,85))
		#crop the bottom
		crop_1=image_1[0:76,0:128]
		#meerge image
		image_3=np.hstack((crop_1,crop_1))
		
	cv2.imshow("modified image",image_3)
	image=image_3.astype(float)
	
	return image_3
	