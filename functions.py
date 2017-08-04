import numpy as np
import cv2
import os

#Here we will put all of the functions used for preprocessing
#First we get the image path, open the image, large-scale, rescale it and then cut out the lower banner 
#merge them if there are two of them either-wise just copy it twice 
#NOTE THAT THE FIRST IMAGE MUST BE THE BRIGHT IMAGE, if not decrease in precision !!!

def preproc(imagePath1, imagePath2=""):

	#Test the presence of the file or files
	presence_1=os.path.isfile(imagePath1)
	presence_2=os.path.isfile(imagePath2)
	
	if(not(presence_1) and not(presence_2)):
		print("None of the files exists or was found in the system")
		return 0
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
	
	#Preprocessing of the images given their resolution
	if(size==(900,1392,3)):
		ratio=size[1]/size[0]
		resized_size=(128,int(128/ratio))
		print("this is a 1392*900 image")
		#resize the image
		image_1=cv2.resize(image_1,(128,resized_size))
		#crop the bottom
		crop_1=image_1[0:66,0:128]
		#add bottom padding
		stack=np.zeros((10,128,3),dtype=np.uint8)
		crop_1=np.vstack((crop_1,stack))
		
		#if there is a second image
		if(presence_2):
			image_2=cv2.resize(image_2,(128,resized_size))
			crop_2=image_2[0:66,0:128]
			crop_2=np.vstack((crop_2,stack))
			
		#if we find the two images we merge them together, otherwise we duplicate the image
		if(presence_1 and presence_2):
			image_3=np.hstack((crop_1,crop_2))
			
		if(presence_1 and not(presence_2)):
			#take note of this 
			print("THE RESOLUTION 1392*900 YIELDS TWO IMAGES YET ONLY ONE WAS GIVEN")
			return 0
			
	if(size==(930,1628,3)):
		ratio=size[1]/size[0]
		resized_size=(128,int(128/ratio))
		print("this is a 1682*930 image")
		#resize the image
		image_1=cv2.resize(image_1,(128,resized_size))
		#crop the bottom
		crop_1=image_1[0:57,0:128]
		#add bottom padding
		stack=np.zeros((19,128,3),dtype=np.uint8)
		crop_1=np.vstack((crop_1,stack))
		
		#if there is a second image
		if(presence_2):
			image_2=cv2.resize(image_2,(128,resized_size))
			crop_2=image_2[0:57,0:128]
			crop_2=np.vstack((crop_2,stack))
		
		#if we find the two images we merge them together, otherwise we duplicate the image		
		if(presence_1 and presence_2):
			image_3=np.hstack((crop_1,crop_2))
			
		if(presence_1 and not(presence_2)):
			#take note of this 
			print("THE RESOLUTION 1682*930 YIELDS TWO IMAGES YET ONLY ONE WAS GIVEN")
			return 0
			
		
	if(size==(1488,2240,3)):
		ratio=size[1]/size[0]
		resized_size=(128,int(128/ratio))
		print("this is a 2240*1488 image")		
		#resize the image
		image_1=cv2.resize(image_1,(128,resized_size))
		#crop the bottom
		crop_1=image_1[0:72,0:128]
		#add bottom padding
		stack=np.zeros((4,128,3),dtype=np.uint8)
		crop_1=np.vstack((crop_1,stack))
		#merge image
		image_3=np.hstack((crop_1,crop_1))
		
	if(size==(2136,3216,3)):
		ratio=size[1]/size[0]
		resized_size=(128,int(128/ratio))
		print("this is a 3216*2136 image")		
		#resize the image
		image_1=cv2.resize(image_1,(128,resized_size))
		#crop the bottom
		crop_1=image_1[0:76,0:128]
		#merge image
		image_3=np.hstack((crop_1,crop_1))
		
	if(size==(3000,4496,3)):
		ratio=size[1]/size[0]
		resized_size=(128,int(128/ratio))
		print("this is a 4496*3000 image")		
		#resize the image
		image_1=cv2.resize(image_1,(128,resized_size))
		#crop the bottom
		crop_1=image_1[0:76,0:128]
		#merge image
		image_3=np.hstack((crop_1,crop_1))
		
	if(size==(4000,6000,3)):
		ratio=size[1]/size[0]
		resized_size=(128,int(128/ratio))
		print("this is a 6000*4000 image")
		#resize the image
		image_1=cv2.resize(image_1,(128,resized_size))
		#crop the bottom
		crop_1=image_1[0:76,0:128]
		#merge image
		image_3=np.hstack((crop_1,crop_1))
	
	try:
		image_3
	except NameError:
		print("We could not merge the two images for some reason")
		print("Make sure that your input images have the correct size")
	else:	
		return image_3
	
	
	