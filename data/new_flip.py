import cv2


existing=96800
new=28000
#first thing we do is create a large list with all of the paths for the training files
filenames_plus=["./PLUSIEURS_VEHICULES/%08d.tiff"%(i) for i in range((existing+1),(existing+new+1))]
# filenames.extend(["./PLUSIEURS_VEHICULES/%08d.tiff"%(k) for k in range(1,20001)])

#We initialize a file counter to keep track of our process
file_counter_plus=0

for image_path_plus in filenames_plus:
    
    file_counter_plus=file_counter_plus+1
    # if(file_counter_plus%1000==0):
        # print("%d files were processed"%(file_counter_plus))
    #open the grayscale* image, and generate it's one-hot vector label
    image=cv2.imread(image_path_plus)
    image=cv2.flip(image,1)
    cv2.imwrite("./PLUSIEURS_VEHICULES/%08d.tiff"%(file_counter_plus+(existing+new)),image)
    
#first thing we do is create a large list with all of the paths for the training files
filenames_ok=["./OK/%08d.tiff"%(i) for i in range((existing+1),(existing+new+1))]

#We initialize a file counter to keep track of our process
file_counter_ok=0

for image_path_ok in filenames_ok:
    
    file_counter_ok=file_counter_ok+1
    # if(file_counter_ok%1000==0):
        # print("%d files were processed"%(file_counter_ok))
    #open the grayscale* image, and generate it's one-hot vector label
    image=cv2.imread(image_path_ok)
    image=cv2.flip(image,1)
    cv2.imwrite("./OK/%08d.tiff"%(file_counter_ok+(existing+new)),image)