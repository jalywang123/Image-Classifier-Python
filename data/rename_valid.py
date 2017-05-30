import cv2
import os

###########
#IMPORTANT#
###########
number_files_validation=4800
last_training_number=92000



#first thing we do is create a large list with all of the paths for the training files
filenames_plus=["./PLUSIEURS_VEHICULES/%08d.tiff"%(i) for i in range(1,(number_files_validation+1))]
#We initialize a file counter to keep track of our process
file_counter_plus=0


for image_path_plus in filenames_plus:

	file_counter_plus=file_counter_plus+1
	if(file_counter_plus%1000==0):
		print("%d files were processed"%(file_counter_plus))
	#rename file
	os.rename(image_path_plus, "./PLUSIEURS_VEHICULES/%08d.tiff"%(file_counter_plus+last_training_number))
	# print("./PLUSIEURS_VEHICULES/%08d.tiff"%(file_counter_plus+last_training_number))


	
#first thing we do is create a large list with all of the paths for the training files
filenames_ok=["./OK/%08d.tiff"%(i) for i in range(1,(number_files_validation+1))]
#We initialize a file counter to keep track of our process
file_counter_ok=0



for image_path_ok in filenames_ok:

	file_counter_ok=file_counter_ok+1
	if(file_counter_ok%1000==0):
		print("%d files were processed"%(file_counter_plus))
	#rename file
	os.rename(image_path_ok, "./OK/%08d.tiff"%(file_counter_ok+last_training_number))
    
