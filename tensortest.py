import tensorflow as tf
import cv2

# conversion of data for the model testing
# for i in range(1,51):
    # dog="./dog/dog_%03d.png"%(i)
    # cat="./cat/cat_%03d.png"%(i)
    # image_dog=cv2.imread(dog,0)
    # image_cat=cv2.imread(cat,0)
    # image_dog=cv2.resize(image_dog,(128,85))
    # image_cat=cv2.resize(image_cat,(128,85))
    # cv2.imwrite("./data_dog/%08d.png"%(i),image_dog)
    # cv2.imwrite("./data_cat/%08d.png"%(i),image_cat)
    # print("done %d"%(i))
    
image=cv2.imread("./data_dog/%08d.png"%(10),0)
cv2.imshow("This is the image",image)
print("this is the value of one of the pixels : ",image[1][12])
print("this is the type of one of the pixels : ",type(image[1][12]))
cv2.waitKey(0)

sess = tf.InteractiveSession()
a = tf.constant(5,dtype=tf.uint8)
b = tf.constant(6,dtype=tf.uint8)
c=a*b
print(a)
print(b)
print(c)
print(c.eval())