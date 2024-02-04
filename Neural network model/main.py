print("Ideal learning rate",10** -3) 
import cv2
img = cv2.imread("/Users/yogesh/pythoncode/Tensorflow/tensorflow/Neural network model/image_data_set/coat1.jpeg")

img_min =img.min()
img_max=img.max()
# Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# resize the image the 28*28
img = cv2.resize(img_gray,dsize=(28,28))

img_norm = img/img_max

print(img_norm)

img_reshape = img.reshape(1,28,28)