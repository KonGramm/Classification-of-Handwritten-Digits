#---------------------------------------- Preamble -------------------------------------------------------------
#
# Purpose: Classification of Handwritten digits.
# In this project, we address the challenge of classifying handwritten digits 
# using a dataset of images from the US Postal Service Database. 
# The data includes training and test sets of 16x16 pixel images, represented as 256-element vectors.
#
# We construct an algorithm for classification by computing the Singular Value Decomposition (SVD)
# of the matrix corresponding to each digit class. We use the first few (5-20) singular vectors as 
# the basis for each class and classify unknown test digits based on how well they can be represented 
# in terms of these bases, using the residual vector in the least squares problem as a measure.
#
# In the code below, we tune the classification algorithm for accuracy, analyze how easy or difficult 
# different digits are to classify, and explore the effect of varying the number of basis vectors.
#
# Tasks:
# 1. Tune the algorithm and provide a graph of classification accuracy as a function of the number of basis vectors.
# 2. Investigate which digits are more difficult to classify and analyze the misclassified digits.
# 3. Explore the possibility of using different numbers of basis vectors for different classes.
#
# Data: 
# - azip.txt (training images), dzip.txt (training labels),
# - testzip.txt (test images), dtest.txt (test labels)
# 
# In order to run this code, download the corresponding excel data and ensure that it is stored in a folder
# on your PC. 
#
# Author: Konstantinos Grammenos

#Import the necessary libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import svd, norm
from sklearn.metrics import accuracy_score,  classification_report


#load the data
x_train = pd.read_excel(r'C:\Users\kosti\OneDrive\Desktop\Masters\Python\Homework_3\data.xlsx','azip', header=None)
y_train = pd.read_excel(r'C:\Users\Kosti\OneDrive\Desktop\masters\Python\Homework_3\data.xlsx','dzip', header=None)
x_test = pd.read_excel(r'C:\Users\Kosti\OneDrive\Desktop\masters\Python\Homework_3\data.xlsx','testzip', header=None)
y_test=pd.read_excel(r'C:\Users\Kosti\OneDrive\Desktop\masters\Python\Homework_3\data.xlsx','dtest', header=None)
x_train.head()

digit_image=x_train[0]
plt.imshow(digit_image.to_numpy().reshape(16,16),cmap='binary')       
  
def display_image(x):
    plt.imshow(x_train[x].values.reshape(16,16),cmap='Greys')
    plt.show()

alpha_matrices={}
for i in range(10):
    alpha_matrices.update({"A"+str(i):x_train.loc[:,list(y_train.loc[0,:]==i)]})
#print(alpha_matrices['A1'].shape)

#Calculate the SVDs for each A
left_singular={}
singular_matix={}
right_singular={}
for i in range(10):
    u, s, v_t = svd(alpha_matrices['A'+str(i)], full_matrices=False)
    left_singular.update({"u"+str(i):u})
    singular_matix.update({"s"+str(i):s})
    right_singular.update({"v_t"+str(i):v_t})
#print(left_singular['u0'].shape)


#I πίνακας με 1 στην διαγωνιο και 0 οπουδήποτε αλλού
I = np.eye(x_test.shape[0])
#Ta k παίρνουν τιμές από 5 μέχρι 20
k=np.arange(5,21)
len_test=x_test.shape[1]
predictions=np.empty((y_test.shape[1],0), dtype = int)

for t in list(k):
    prediction = []
    for i in range(len_test):
        residuals = []
        for j in range(10):
            u=left_singular["u"+str(j)][:,0:t]
            res=norm( np.dot(I-np.dot(u,u.T), x_test[i]  ))
            residuals.append(res)
        index_min = np.argmin(residuals)
        prediction.append(index_min)        
    prediction=np.array(prediction)
    #print(prediction)
    #print(prediction.reshape(-1,1))
    #το reshape(-1,1) τον μετατρέπει σε ένα πίνακα (n,1)
    predictions=np.hstack((predictions,prediction.reshape(-1,1)))

#τα Accuracy scores για όλες τις διαφορετικές τιμές των k
scores=[]
for i in range(len(k)):
    score=accuracy_score(y_test.loc[0,:],predictions[:,i])
    scores.append(score)
#k=15 gave the best results

data={"Number of basis vectors":list(k), "accuracy_score":scores}
df=pd.DataFrame(data).set_index("Number of basis vectors")
plt.plot(list(k), scores)
print(predictions[:,7])
print(classification_report(y_test.loc[0,:],predictions[:,7]))


#task 2:Check if all digits are equally easy or difficult to classify. 
#Also look at some of the difficult ones and see that in many cases they are very badly written.
misclassified = np.where(y_test.loc[0,:] != predictions[:,7])
plt.figure(figsize=(25,10))
columns = 5
rows = math.ceil(10 / columns)  # Make sure the number of rows is an integer
for i in range(2,12):
    misclassified_id=misclassified[0][i]
    image=x_test[misclassified_id]
    plt.subplot(rows, columns, i-1)
    plt.imshow(image.to_numpy().reshape(16,16),cmap='binary')
    plt.title("True label:"+str(y_test.loc[0,misclassified_id]) + '\n'+ "Predicted label:"+str(predictions[misclassified_id,15]))
 

