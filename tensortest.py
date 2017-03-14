import tensorflow as tf
import cv2
#conversion of data for the model testing
for i in range(1,51):
    dog="./dog/dog_%03d.png"%(i)
    cat="./cat/cat_%03d.png"%(i)
    image_dog=cv2.imread(dog,0)
    image_cat=cv2.imread(cat,0)
    image_dog=cv2.resize(image_dog,(128,85))
    image_cat=cv2.resize(image_cat,(128,85))
    cv2.imwrite("./data_dog/%08d.png"%(i),image_dog)
    cv2.imwrite("./data_cat/%08d.png"%(i),image_cat)
    print("done %d"%(i))