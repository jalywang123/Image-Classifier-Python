import classifier as classif
import functions as func
import cv2
import os

#we launch our neural net with saved weights
classif.launch_model()

dirs=["./tiff"]

for dir in dirs:
	files=os.listdir(dir)
	counter_false=0
	new_false=False
	counter_file=0

	for file in files:
		counter_file+=1
		image=cv2.imread(dir+"/"+file)
		if(counter_file==100):
			break
			
		prob=classif.classify(image)
		
		image=cv2.resize(image,(600,300))	
		if(prob[0][0][1]>prob[0][0][0]):
			cv2.putText(image, "Plusieurs Vehicules : "+str(prob[0][0][1])+"%", (15, 15),
			cv2.FONT_HERSHEY_TRIPLEX  , 0.8, (0, 0, 255), 1)
			new_false=True
		else:
			cv2.putText(image, "Vehicule unique : "+str(prob[0][0][0])+"%", (15, 15),
			cv2.FONT_HERSHEY_TRIPLEX  , 0.8, (0, 255, 0), 1)
			
		
		cv2.imshow("image",image)
		cv2.waitKey(0)
		if(counter_false%100==0 and new_false):
			print(counter_false," images were misclassified")
			new_false=False
	# print("we misclassified %d ok images "%(counter_false))

