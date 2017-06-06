import cv2
import numpy as np
import os

#This is a script to merge images that are correlated if not we just add a the same image on the right
#First we get the set 
set="train"
set_merged=set+"_merged"
num_valid=2400
#previous is the first index of your current files that are unmerged
previous=96800
#new is the number of images that you have generated
new=28000
if(set=="train"):
	
	#We start off with the OK class
	for i in range((previous+1),(previous+new+1)):
		# if(i%1000==0):
			# print("we have already processed %d files in OK"%(i))
		presence="./%s/OK/%08d.tiff"%(set,i)
		if(os.path.isfile(presence)):
			only=cv2.imread(presence)
			presence_size=only.shape
			if (presence_size==(76,128,3)):
				double=np.hstack((only,only))
				cv2.imwrite("./%s/OK/%08d.tiff"%(set_merged,i),double)
		presence_1="./%s/OK/%08d_1.tiff"%(set,i)
		presence_2="./%s/OK/%08d_2.tiff"%(set,i)
		if(os.path.isfile(presence_1) and os.path.isfile(presence_2)):
			# we first checked if the file existed 
			# Now we open the two images in grayscale and we merge them
			first=cv2.imread(presence_1)
			second=cv2.imread(presence_2)
			third=np.hstack((first,second))
			# we then save the image as one 256*85 image
			cv2.imwrite("./%s/OK/%08d.tiff"%(set_merged,i),third)


	for i in range((previous+1),(previous+new+1)):
		# if(i%1000==0):
			# print("we have already processed %d files in PLUSIEURS_VEHICULES"%(i))
		presence="./%s/PLUSIEURS_VEHICULES/%08d.tiff"%(set,i)
		if(os.path.isfile(presence)):
			only=cv2.imread(presence)
			presence_size=only.shape
			if (presence_size==(76,128,3)):
				double=np.hstack((only,only))
				cv2.imwrite("./%s/PLUSIEURS_VEHICULES/%08d.tiff"%(set_merged,i),double)
		presence_1="./%s/PLUSIEURS_VEHICULES/%08d_1.tiff"%(set,i)
		presence_2="./%s/PLUSIEURS_VEHICULES/%08d_2.tiff"%(set,i)
		if(os.path.isfile(presence_1) and os.path.isfile(presence_2)):
			
			# we first checked if the file existed 
			# Now we open the two images in grayscale and we merge them
			first=cv2.imread(presence_1)
			second=cv2.imread(presence_2)
			third=np.hstack((first,second))
			# we then save the image as one 256*85 image
			cv2.imwrite("./%s/PLUSIEURS_VEHICULES/%08d.tiff"%(set_merged,i),third)


		
if(set=="valid"):	
	#We start off with the OK class
	for i in range(1,(num_valid+1)):
		# if(i%1000==0):
			# print("we have already processed %d files in OK"%(i))
		presence="./%s/OK/%08d.tiff"%(set,i)
		if(os.path.isfile(presence)):
			only=cv2.imread(presence)
			presence_size=only.shape
			if (presence_size==(76,128,3)):
				double=np.hstack((only,only))
				cv2.imwrite("./%s/OK/%08d.tiff"%(set_merged,i),double)
		presence_1="./%s/OK/%08d_1.tiff"%(set,i)
		presence_2="./%s/OK/%08d_2.tiff"%(set,i)
		if(os.path.isfile(presence_1) and os.path.isfile(presence_2)):
			# we first checked if the file existed 
			# Now we open the two images in grayscale and we merge them
			first=cv2.imread(presence_1)
			second=cv2.imread(presence_2)
			third=np.hstack((first,second))
			# we then save the image as one 256*85 image
			cv2.imwrite("./%s/OK/%08d.tiff"%(set_merged,i),third)


	for i in range(1,(num_valid+1)):
		# if(i%1000==0):
			# print("we have already processed %d files in PLUSIEURS_VEHICULES"%(i))
		presence="./%s/PLUSIEURS_VEHICULES/%08d.tiff"%(set,i)
		if(os.path.isfile(presence)):
			only=cv2.imread(presence)
			presence_size=only.shape
			if (presence_size==(76,128,3)):
				double=np.hstack((only,only))
				cv2.imwrite("./%s/PLUSIEURS_VEHICULES/%08d.tiff"%(set_merged,i),double)
		presence_1="./%s/PLUSIEURS_VEHICULES/%08d_1.tiff"%(set,i)
		presence_2="./%s/PLUSIEURS_VEHICULES/%08d_2.tiff"%(set,i)
		if(os.path.isfile(presence_1) and os.path.isfile(presence_2)):
			
			# we first checked if the file existed 
			# Now we open the two images in grayscale and we merge them
			first=cv2.imread(presence_1)
			second=cv2.imread(presence_2)
			third=np.hstack((first,second))
			# we then save the image as one 256*85 image
			cv2.imwrite("./%s/PLUSIEURS_VEHICULES/%08d.tiff"%(set_merged,i),third)

