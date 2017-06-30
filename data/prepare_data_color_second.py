import platform
import cv2
import csv
import numpy as np
import os.path
import os

#This script is designed to generate segmented and labelled data
#We start of by defining some formal parameters
#we define the format for our data
format=".tiff"
#The destination folder where our segmented data will be saved
# dst_dir = "./"
dst_dir = "D:\\Temp\\train"
previous=39800
train_ok=previous
train_plus=previous
valid_ok=0
valid_plus=0
num_train=previous+30400
num_valid=1


#First we are going to open the csv file with all the information necessary for segmentation
with open("train-valid-test-no_double.csv","rb") as csvfile : 
	linereader=csv.reader(csvfile,delimiter=";")
	line_counter=0
	for row in linereader:
		line_counter=line_counter+1
		if(line_counter%500==0):
			print("We have processed %d lines in the csv file"%(line_counter))

		if(valid_ok==num_valid and valid_plus==num_valid):
			print("THERE ARE %d OK FILES IN TRAIN"%(train_ok))
			print("THERE ARE %d PLUSIEURS_VEHICULES FILES IN TRAIN"%(train_plus))
			print("THERE ARE %d OK FILES IN VALID"%(valid_ok))
			print("THERE ARE %d PLUSIEURS_VEHICULES FILES IN VALID"%(valid_plus))
			break
			
		set=row[11]
		if(set=="train"):
			#first we get the last value of the line, which will give us the two filenames of our images
			element=row[10].split(",")
			# print(len(element))
			# print("\n")
			if(len(element)==2):
				filename_1=element[0].split(".")
				filename_2=element[1].split(".")
				first=filename_1[0]+".png"
				second=filename_2[0]+".png"
			else: 
				filename_1=element[0].split(".")
				first=filename_1[0]+".png"
				second=filename_1[0]+".png"
			
			#Now we get the resolution of the image file as well as it's class label, this will give us the path to the file and set the parameters for cropping.
			#It will also serve for the future storage of our segmented data
			resolution=row[3]
			label=row[4]
			# print "element is ", element
			# print "it's resolution is : ",resolution
			# print "first is : ",first
			# print "second is : ",second
			
			
			#For now we will only be dealing with 7 different resolutions
			if(resolution=="1628_930" or resolution=="1392_900" or resolution=="2240_1488" or resolution=="3216_2136" or resolution=="4496_3000" or resolution=="6000_4000"):
				#The "OK" class indexation
				if(label=="0"):
					status="OK"
				#The "MOTOS" and "PLUSIEURS_VEHICULES" class indexation
				if(label=="1"): 
					status=row[5]
			else:
				print("these resolutions %s are not considered"%(resolution))
				
			path="./%s/GDI_VIT/%s/%s/"%(set,resolution,status)
			# path="./GDI_VIT/%s/%s/"%(resolution,status)
			#Now we grayscale the image, crop it and resize it 
			
			
			if(resolution=="1628_930"):
				#test the presence of a file in the system 
				presence_1=os.path.isfile(path+first)
				presence_2=os.path.isfile(path+second)
				if(presence_1 and presence_2):
					save_set="train"			
					#we define the width and the height of the resize image
					#here the width will be 128 and the height will be 73
					# print(" the files that you are looking for exist")
					image_1=cv2.imread(path+first)
					image_2=cv2.imread(path+second) 
					crop_1=image_1[0:57,0:128]
					crop_2=image_2[0:57,0:128]
					#we then consider a stack that we will be appending to the resized image
					stack=np.zeros((19,128,3),dtype=np.uint8)
					crop_1=np.vstack((crop_1,stack))
					crop_2=np.vstack((crop_2,stack))
					
					#valid images
					if(label=="0" and train_ok==num_train and valid_ok<num_valid):
						save_set="valid"
						valid_ok=valid_ok+1
						SavePath_1="%s/clean_data/second/%s/%s/%08d_1%s" % (dst_dir,save_set, status, valid_ok, format)
						SavePath_2="%s/clean_data/second/%s/%s/%08d_2%s" % (dst_dir,save_set, status, valid_ok, format)
						cv2.imwrite(SavePath_1,crop_1)
						cv2.imwrite(SavePath_2,crop_2)
						
					if(label=="1"):
						if(status=="PLUSIEURS_VEHICULES" and train_plus==num_train and valid_plus<num_valid):
							save_set="valid"
							valid_plus=valid_plus+1
							SavePath_1="%s/clean_data/second/%s/%s/%08d_1%s" % (dst_dir,save_set, status, valid_plus, format)
							SavePath_2="%s/clean_data/second/%s/%s/%08d_2%s" % (dst_dir,save_set, status, valid_plus, format)
							cv2.imwrite(SavePath_1,crop_1)
							cv2.imwrite(SavePath_2,crop_2)
			
					if(label=="0" and train_ok<num_train):
						train_ok=train_ok+1
						save_set="train"
						SavePath_1="%s/clean_data/second/%s/%s/%08d_1%s" % (dst_dir,save_set, status, train_ok, format)
						SavePath_2="%s/clean_data/second/%s/%s/%08d_2%s" % (dst_dir,save_set, status, train_ok, format)
						
						cv2.imwrite(SavePath_1,crop_1)
						cv2.imwrite(SavePath_2,crop_2)
					if(label=="1"): 						
						if(status=="PLUSIEURS_VEHICULES" and train_plus<num_train):
							train_plus=train_plus+1
							save_set="train"
							SavePath_1="%s/clean_data/second/%s/%s/%08d_1%s" % (dst_dir,save_set, status, train_plus, format)
							SavePath_2="%s/clean_data/second/%s/%s/%08d_2%s" % (dst_dir,save_set, status, train_plus, format)
							
							cv2.imwrite(SavePath_1,crop_1)
							cv2.imwrite(SavePath_2,crop_2)
				else:
					print("the files do not exist")


					
					
			if(resolution=="1392_900"):
				#test the presence of a file in the system 
				presence_1=os.path.isfile(path+first)
				presence_2=os.path.isfile(path+second)
				if(presence_1 and presence_2):
					save_set="train"
					#we define the width and the height of the resize image
					#here the width will be 128 and the height will be 82
					# print ("we are dealing with  ------1392x900------ resolution images")
					# print(" the files that you are looking for exist")
					image_1=cv2.imread(path+first)
					image_2=cv2.imread(path+second)
					crop_1=image_1[0:66,0:128]
					crop_2=image_2[0:66,0:128]
					#we then consider a stack that we will be appending to the resized image
					stack=np.zeros((10,128,3),dtype=np.uint8)
					crop_1=np.vstack((crop_1,stack))
					crop_2=np.vstack((crop_2,stack))
					
					#valid images
					if(label=="0" and train_ok==num_train and valid_ok<num_valid):
						save_set="valid"
						valid_ok=valid_ok+1
						SavePath_1="%s/clean_data/second/%s/%s/%08d_1%s" % (dst_dir,save_set, status, valid_ok, format)
						SavePath_2="%s/clean_data/second/%s/%s/%08d_2%s" % (dst_dir,save_set, status, valid_ok, format)
						cv2.imwrite(SavePath_1,crop_1)
						cv2.imwrite(SavePath_2,crop_2)
						
					if(label=="1"):						
						if(status=="PLUSIEURS_VEHICULES" and train_plus==num_train and valid_plus<num_valid):
							save_set="valid"
							valid_plus=valid_plus+1
							SavePath_1="%s/clean_data/second/%s/%s/%08d_1%s" % (dst_dir,save_set, status, valid_plus, format)
							SavePath_2="%s/clean_data/second/%s/%s/%08d_2%s" % (dst_dir,save_set, status, valid_plus, format)
							cv2.imwrite(SavePath_1,crop_1)
							cv2.imwrite(SavePath_2,crop_2)
			
					if(label=="0" and train_ok<num_train):
						train_ok=train_ok+1
						save_set="train"
						SavePath_1="%s/clean_data/second/%s/%s/%08d_1%s" % (dst_dir,save_set, status, train_ok, format)
						SavePath_2="%s/clean_data/second/%s/%s/%08d_2%s" % (dst_dir,save_set, status, train_ok, format)
						
						cv2.imwrite(SavePath_1,crop_1)
						cv2.imwrite(SavePath_2,crop_2)
					if(label=="1"): 
						if(status=="PLUSIEURS_VEHICULES" and train_plus<num_train):
							train_plus=train_plus+1
							save_set="train"
							SavePath_1="%s/clean_data/second/%s/%s/%08d_1%s" % (dst_dir,save_set, status, train_plus, format)
							SavePath_2="%s/clean_data/second/%s/%s/%08d_2%s" % (dst_dir,save_set, status, train_plus, format)
							
							cv2.imwrite(SavePath_1,crop_1)
							cv2.imwrite(SavePath_2,crop_2)
					
				else:
					print("the files do not exist")

					
					
			if(resolution=="2240_1488"):
				#test the presence of a file in the system
				presence_1=os.path.isfile(path+first)
				presence_2=os.path.isfile(path+second)
				if(presence_1 or presence_2):
					save_set="train"
					#we define the width and the height of the resize image
					#here the width will be 128 and the height will be 85
					# print ("we are dealing with  ------2240x1488------ resolution images")
					# print(" the files that you are looking for exist")
					image_2=cv2.imread(path+second)
					crop_2=image_2[0:72,0:128]
					stack=np.zeros((4,128,3),dtype=np.uint8)
					crop_2=np.vstack((crop_2,stack))
					#valid images
					if(label=="0" and train_ok==num_train and valid_ok<num_valid):
						save_set="valid"
						valid_ok=valid_ok+1
						SavePath_2="%s/clean_data/second/%s/%s/%08d%s" % (dst_dir,save_set, status, valid_ok, format)
						cv2.imwrite(SavePath_2,crop_2)
						
					if(label=="1"):						
						if(status=="PLUSIEURS_VEHICULES" and train_plus==num_train and valid_plus<num_valid):
							save_set="valid"
							valid_plus=valid_plus+1
							SavePath_2="%s/clean_data/second/%s/%s/%08d%s" % (dst_dir,save_set, status, valid_plus, format)
							cv2.imwrite(SavePath_2,crop_2)
			
					if(label=="0" and train_ok<num_train):
						train_ok=train_ok+1
						save_set="train"
						SavePath_2="%s/clean_data/second/%s/%s/%08d%s" % (dst_dir,save_set, status, train_ok, format)
						cv2.imwrite(SavePath_2,crop_2)
					if(label=="1"):				
						if(status=="PLUSIEURS_VEHICULES" and train_plus<num_train):
							train_plus=train_plus+1
							save_set="train"
							SavePath_2="%s/clean_data/second/%s/%s/%08d%s" % (dst_dir,save_set, status, train_plus, format)
							cv2.imwrite(SavePath_2,crop_2)
				else:
					print("the files do not exist")
	
				
			if(resolution=="3216_2136"):
				#test the presence of a file in the system 
				presence_1=os.path.isfile(path+first)
				presence_2=os.path.isfile(path+second)
				if(presence_1 or presence_2):
					save_set="train"
					#we define the width and the height of the resize image
					#here the width will be 128 and the height will be 85
					# print ("we are dealing with  ------3216x2136------ resolution images")
					# print(" the files that you are looking for exist")
					image_2=cv2.imread(path+second)
					crop_2=image_2[0:76,0:128]
					# print("###########YOUR FILES WERE SAVED IN %s ###########"%(SavePath_2))
					
					#valid images
					if(label=="0" and train_ok==num_train and valid_ok<num_valid):
						save_set="valid"
						valid_ok=valid_ok+1
						SavePath_2="%s/clean_data/second/%s/%s/%08d%s" % (dst_dir,save_set, status, valid_ok, format)
						cv2.imwrite(SavePath_2,crop_2)
						
					if(label=="1"):					
						if(status=="PLUSIEURS_VEHICULES" and train_plus==num_train and valid_plus<num_valid):
							save_set="valid"
							valid_plus=valid_plus+1
							SavePath_2="%s/clean_data/second/%s/%s/%08d%s" % (dst_dir,save_set, status, valid_plus, format)
							cv2.imwrite(SavePath_2,crop_2)
			
					if(label=="0" and train_ok<num_train):
						train_ok=train_ok+1
						save_set="train"
						SavePath_2="%s/clean_data/second/%s/%s/%08d%s" % (dst_dir,save_set, status, train_ok, format)
						cv2.imwrite(SavePath_2,crop_2)
					if(label=="1"): 						
						if(status=="PLUSIEURS_VEHICULES" and train_plus<num_train):
							train_plus=train_plus+1
							save_set="train"
							SavePath_2="%s/clean_data/second/%s/%s/%08d%s" % (dst_dir,save_set, status, train_plus, format)
							cv2.imwrite(SavePath_2,crop_2)
				else:
					print("the files do not exist")
					
			if(resolution=="4496_3000"):
				#test the presence of a file in the system 
				presence_1=os.path.isfile(path+first)
				presence_2=os.path.isfile(path+second)
				if(presence_1 or presence_2):
					save_set="train"
					#we define the width and the height of the resize image
					#here the width will be 128 and the height will be 85
					# print ("we are dealing with  ------4496x3000------ resolution images")
					# print(" the files that you are looking for exist")
					image_2=cv2.imread(path+second)
					crop_2=image_2[0:76,0:128]
					# print("###########YOUR FILES WERE SAVED IN %s ###########"%(SavePath_2))
					
					#valid images
					if(label=="0" and train_ok==num_train and valid_ok<num_valid):
						save_set="valid"
						valid_ok=valid_ok+1
						SavePath_2="%s/clean_data/second/%s/%s/%08d%s" % (dst_dir,save_set, status, valid_ok, format)
						cv2.imwrite(SavePath_2,crop_2)
						
					if(label=="1"):						
						if(status=="PLUSIEURS_VEHICULES" and train_plus==num_train and valid_plus<num_valid):
							save_set="valid"
							valid_plus=valid_plus+1
							SavePath_2="%s/clean_data/second/%s/%s/%08d%s" % (dst_dir,save_set, status, valid_plus, format)
							cv2.imwrite(SavePath_2,crop_2)
			
					if(label=="0" and train_ok<num_train):
						train_ok=train_ok+1
						save_set="train"
						SavePath_2="%s/clean_data/second/%s/%s/%08d%s" % (dst_dir,save_set, status, train_ok, format)
						cv2.imwrite(SavePath_2,crop_2)
					if(label=="1"): 						
						if(status=="PLUSIEURS_VEHICULES" and train_plus<num_train):
							train_plus=train_plus+1
							save_set="train"
							SavePath_2="%s/clean_data/second/%s/%s/%08d%s" % (dst_dir,save_set, status, train_plus, format)
							cv2.imwrite(SavePath_2,crop_2)
				
				else:
					print("the files do not exist")
			
			if(resolution=="6000_4000"):
				#test the presence of a file in the system 
				presence_1=os.path.isfile(path+first)
				presence_2=os.path.isfile(path+second)
				if(presence_1 or presence_2):
					save_set="train"
					#we define the width and the height of the resize image
					#here the width will be 128 and the height will be 85
					# print ("we are dealing with  ------6000x4000------ resolution images")
					# print(" the files that you are looking for exist")
					image_2=cv2.imread(path+second)
					crop_2=image_2[0:76,0:128]
					# print("###########YOUR FILES WERE SAVED IN %s ###########"%(SavePath_2))
					
					#valid images
					if(label=="0" and train_ok==num_train and valid_ok<num_valid):
						save_set="valid"
						valid_ok=valid_ok+1
						SavePath_2="%s/clean_data/second/%s/%s/%08d%s" % (dst_dir,save_set, status, valid_ok, format)
						cv2.imwrite(SavePath_2,crop_2)
						
					if(label=="1"):						
						if(status=="PLUSIEURS_VEHICULES" and train_plus==num_train and valid_plus<num_valid):
							save_set="valid"
							valid_plus=valid_plus+1
							SavePath_2="%s/clean_data/second/%s/%s/%08d%s" % (dst_dir,save_set, status, valid_plus, format)
							cv2.imwrite(SavePath_2,crop_2)
			
					if(label=="0" and train_ok<num_train):
						train_ok=train_ok+1
						save_set="train"
						SavePath_2="%s/clean_data/second/%s/%s/%08d%s" % (dst_dir,save_set, status, train_ok, format)
						cv2.imwrite(SavePath_2,crop_2)
					if(label=="1"):						
						if(status=="PLUSIEURS_VEHICULES" and train_plus<num_train):
							train_plus=train_plus+1
							save_set="train"
							SavePath_2="%s/clean_data/second/%s/%s/%08d%s" % (dst_dir,save_set, status, train_plus, format)
							cv2.imwrite(SavePath_2,crop_2)
								
				else:
					print("the files do not exist")
					
			