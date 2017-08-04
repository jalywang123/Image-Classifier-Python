import classifier as classif
import functions as func
import cv2
import os
import numpy as np

#we launch our neural net with saved weights
classif.launch_model()
valid_files=4000

filenames=["./OK/%08d.tiff"%(i) for i in range(1,valid_files+1)]
filenames.extend(["./PLUSIEURS_VEHICULES/%08d.tiff"%(k) for k in range(1,valid_files+1)])
np.random.shuffle(filenames)
counter_file=0

#full evaluation
test_all=False
if(test_all):
	misclassified_ok=0
	misclassified_plus=0


for file in filenames:
	counter_file+=1
	
	image=cv2.imread(file)
	
	label=file.split("/")[1]
	
	if(not(test_all)):
		if(counter_file==10):
			break
		
	prob=classif.classify(image)
	
	if(not(test_all)):
		image=cv2.resize(image,(896,266))
	
	#reseau de neurones : plusieurs vehicules et accord avec l'operateur
	if(prob[0][0][1]>prob[0][0][0] and label=="PLUSIEURS_VEHICULES"):
		if(not(test_all)):
			cv2.putText(image,"OP : Plusieurs Vehicules", (0, 20),
			cv2.FONT_HERSHEY_DUPLEX  , 1.0, (0, 255, 0), 1)
			
			cv2.putText(image, "Plusieurs Vehicules : "+str(prob[0][0][1])+"%", (450, 20),
			cv2.FONT_HERSHEY_DUPLEX  , 1.0, (0, 255, 0), 1)
		
	#desaccord plusieurs vehicules
	if(prob[0][0][1]>prob[0][0][0] and label!="PLUSIEURS_VEHICULES"):
		if(test_all):
			misclassified_plus+=1
		else:
			cv2.putText(image,"OP : Vehicule Unique", (0, 20),
			cv2.FONT_HERSHEY_DUPLEX  , 1.0, (0, 255, 0), 1)
			
			cv2.putText(image, "Plusieurs Vehicules : "+str(prob[0][0][1])+"%", (450, 20),
			cv2.FONT_HERSHEY_DUPLEX  , 1.0, (0, 0, 255), 1)
	
	#reseau de neurones : Vehicule Unique et accord avec l'operateur
	if(prob[0][0][1]<prob[0][0][0] and label=="OK"):
		if(not(test_all)):
			cv2.putText(image,"OP : Vehicule Unique", (0, 20),
			cv2.FONT_HERSHEY_DUPLEX  , 1.0, (0, 255, 0), 1)
			
			cv2.putText(image, "Vehicule Unique : "+str(prob[0][0][0])+"%", (450, 20),
			cv2.FONT_HERSHEY_DUPLEX  , 1.0, (0, 255, 0), 1)	
	
	#desaccord Vehicule Unique
	if(prob[0][0][1]<prob[0][0][0] and label!="OK"):
		if(test_all):
			misclassified_ok+=1
		else:
			cv2.putText(image,"OP : Plusieurs Vehicules", (0, 20),
			cv2.FONT_HERSHEY_DUPLEX  , 1.0, (0, 255, 0), 1)
			
			cv2.putText(image, "Vehicule Unique : "+str(prob[0][0][0])+"%", (450, 20),
			cv2.FONT_HERSHEY_DUPLEX  , 1.0, (0, 0, 255), 1)
	"""#reseau de neurones : plusieurs vehicules	
	else:
		cv2.putText(image, "Vehicule unique : "+str(prob[0][0][0])+"%", (15, 15),
		cv2.FONT_HERSHEY_DUPLEX  , 1.0, (0, 255, 0), 1)"""
		
	if(not(test_all)):
		cv2.imshow("image",image)
		cv2.waitKey(0)

if(test_all):
	print("We misclassified %d OK Images"%(misclassified_ok))
	print("We misclassified %d Multiple Vehicle Images"%(misclassified_plus))

