import cv2 
import numpy as np 

image = cv2.imread('F:\Deep Learning Hub\CNN\how to convert image into numerical array\cat-8015038_1280.jpg')

#to convert into numerical array
image_array = np.array(image)
# print(image_array.shape)

#to flatten a matrix X of shape (a,b,c,d)
flattened_array = image_array.reshape(-1)
# print(flattened_array.shape)

#Note:
'''
-1 tells NumPy to infer the total number of elements automatically.
The shape (720, 1280, 3) has 720 × 1280 × 3 = 2,764,800 elements, so the output shape becomes (2764800,).

EXAMPLE 2:
if there is shape like image_array.reshape(50,-1).T = (50, 72, 72, 3) = (72 x 72 x 3, 50) = (15552, 50) 
here 50(training examples)
72(width) and 3 dimention

to standarized flatten array
# numpy_arr = np.array([2,4,5])
# std_myarr = numpy_arr/ 255.0
# # print(std_myarr)

'''

#To Standarized and centralized data for image dataset . 255 maximum number of pixels value

train_set_x = flattened_array/ 255.
print(train_set_x)

