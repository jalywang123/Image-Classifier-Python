from __future__ import division
import cv2
import numpy as np
import os

set="train_merged"
average_ok=0
squared_average_ok=0
average_motos=0
squared_average_motos=0
average_plus=0
squared_average_plus=0
max_images=3000

for i in range(1,1001):
	if(i%4000==0):
		print("we added %f files in OK in %s "%(i,set))
	path="./%s/OK/%08d.tiff"%(set,i)
	image=cv2.imread(path,0)
	image=image.astype(float)
	squared_image=np.square(image)
	squared_mean=squared_image.mean()
	squared_average_ok=squared_average_ok+squared_mean
	ave_image=image.mean()
	average_ok=average_ok+ave_image
	
average_ok=average_ok/max_images
squared_average_ok=squared_average_ok/max_images
print("the whole average of oks is %f"%(average_ok))
print("the whole average of squared oks is %f"%(squared_average_ok))

for i in range(1,1001):
	if(i%4000==0):
		print("we added %f files in MOTOS"%(i))
	path="./%s/MOTOS/%08d.tiff"%(set,i)
	image=cv2.imread(path,0)
	image=image.astype(float)
	squared_image=np.square(image)
	squared_mean=squared_image.mean()
	squared_average_motos=squared_average_motos+squared_mean
	ave_image=image.mean()
	average_motos=average_motos+ave_image
	
average_motos=average_motos/max_images
squared_average_motos=squared_average_motos/max_images
print("the whole average of motos is %f"%(average_motos))
print("the whole average of squared motos is %f"%(squared_average_motos))

for i in range(1,1001):
	if(i%4000==0):
		print("we added %f files in PLUSIEURS VEHICULES"%(i))
	path="./%s/PLUSIEURS_VEHICULES/%08d.tiff"%(set,i)
	image=cv2.imread(path,0)
	image=image.astype(float)
	squared_image=np.square(image)
	squared_mean=squared_image.mean()
	squared_average_plus=squared_average_plus+squared_mean
	ave_image=image.mean()
	average_plus=average_plus+ave_image
	

average_plus=average_plus/max_images
squared_average_plus=squared_average_plus/max_images
print("the whole average of plusieurs vehicules is %f"%(average_plus))
print("the whole average of squared plusieurs vehicules is %f"%(squared_average_plus))

average=average_ok+average_motos+average_plus
squared_average=squared_average_ok+squared_average_motos+squared_average_plus
print("the average is %f"%(average))
print("the squared average is %f"%(squared_average))
print("the variance is then %f"%(squared_average-average*average))