import cv2



#first thing we do is create a large list with all of the paths for the training files
filenames=["./OK/%08d.tiff"%(i) for i in range(1,8001)]
# filenames.extend(["./PLUSIEURS_VEHICULES/%08d.tiff"%(k) for k in range(1,20001)])

#We initialize a file counter to keep track of our process
file_counter=0

for image_path in filenames:
    
    file_counter=file_counter+1
    if(file_counter%1000==0):
        print("%d out of 16 000 files were processed"%(file_counter))
    #open the grayscale* image, and generate it's one-hot vector label
    image=cv2.imread(image_path)
    image=cv2.flip(image,1)
    cv2.imwrite("./OK/%08d.tiff"%(file_counter+8000),image)
    