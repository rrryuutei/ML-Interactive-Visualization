#This project does multi-class classification for face images using kernel methods
import numpy
import scipy
import matplotlib.pyplot as plt

#Here we define our kernel function between two images
#Gaussian kernel as the building block of our kernel
def k_rbf(u,v,sigma):
    r = numpy.subtract(u,v)
    r2 = numpy.dot(r,r)    #distance between the two means
    k = scipy.exp(-r2/(sigma**2))
    return k
#Kernel between two images
def K_img(I1,I2,sigma_p,sigma_l):
    #I1, I2 are bags of patches (as a matrix, each entry is a patch vector) of the two images.
    [M,N,p2] = numpy.shape(I1)
    if M != numpy.shape(I2)[0] or N != numpy.shape(I2)[1] or p2 != numpy.shape(I2)[2]:
        print 'I1 and I2 must be of the same dimension!!!'
    p = (numpy.sqrt(p2)+1)/2
    k = 0
    for i1 in range(M):
        for j1 in range(N):
            for i2 in range(M):
                for j2 in range(N):
                    p1 = I1[i1,j1,:]/numpy.linalg.norm(I1[i1,j1,:])
                    p2 = I2[i2,j2,:]/numpy.linalg.norm(I2[i2,j2,:])
                    h_patch = k_rbf(p1,p2,sigma_p)    #similarity between two patches
                    h_loc = k_rbf(numpy.array([i1*p,j1*p]),numpy.array([i2*p,j2*p]),sigma_l) #closeness of two patches
                    k += h_patch*h_loc
    return k

#We use kernel Nearest-mean classifier for multiple classes of images
def NearestMean(test,train,sigma_p,sigma_l):
    #input a list of training sets (each set is one class). Specify parameter for the RBF kernel.
    #Test set is also a list. Each element is a matrix of pathes
    size = len(test)    #size of the test data we need to predict for
    n_class = len(train)
    predict = numpy.zeros(size)    #prediction for each of the test trial
    b = numpy.zeros(n_class)    #the second term b in h(x), only depends on the training data
    for c in range(n_class):
        train_set = train[c]
        n = len(train_set)    #size of the set of class
        for i in range(n):
            for j in range(i):
                b[c] += -K_img(train_set[i],train_set[j],sigma_p,sigma_l)/(n**2)
        for i in range(n):
            b[c] += -K_img(train_set[i],train_set[i],sigma_p,sigma_l)/(2*n**2)
    for trial in range(size):    #calculate <w,phi(x)> for every single x in test set
        w_phi = numpy.zeros(n_class)    #the first term <w,phi(x)> in h(x) for each class 
        for c in range(n_class):      
            train_set = train[c]
            n = len(train_set)    #size of the set of class
            for i in range(n):
                w_phi[c] += K_img(train_set[i], test[trial], sigma_p,sigma_l) / n
        h = w_phi + b
        predict[trial] = numpy.argmax(h)
    return predict

#Getting the LFW dataset and implement the algorithm
from sklearn.datasets import fetch_lfw_people
LFW = fetch_lfw_people(min_faces_per_person=10)    #requiring the min faces, we get three classes
label = LFW.target
n_img = len(label)
images = LFW.images
images = images[:,:-1,:] - images[:,1:,:]  #get gradient images
height = numpy.shape(images)[1] #image height
width = numpy.shape(images)[2]  #image width
#transform images into matrices of patches
Patches = []
p = 5   #square region size
step = (p+1)/2
for im in images:
    p_ind0 = 0 #initial starting corner point of a patch
    while p_ind0 <= height - p:
        p_ind1 = 0
        while p_ind1 <= width - p:
            patch = im[p_ind0:p_ind0+p,p_ind1:p_ind1+p]
            Patches.append(numpy.reshape(patch,(p**2,)))
            p_ind1 += step
        p_ind0 += step
BagSize = len(Patches)/n_img
#normalize the data
from sklearn import preprocessing
Patch_scaled = preprocessing.scale(numpy.array(Patches),axis=0)
#building matrices of patches
ImPatches = []
M = (height-p+step) / step
N = (width-p+step) / step
for i in range(n_img):
    patches = Patch_scaled[i*BagSize:(i+1)*BagSize]
    PatchMat = numpy.zeros((M,N,p**2))
    for j in range(BagSize):
        m = j / N
        n = j % N
        PatchMat[m,n,:] = patches[j]
    ImPatches.append(PatchMat)

#We test the algorithm error in this part
#first sort the images into three classes
index = numpy.random.permutation(len(label))    #randomize the order of the data index
Max_size = 20    #specify the maximual size of a class
class0 = []
class1 = []
class2 = []
for i in range(n_img):
    if label[i] == 0 and len(class0) < 20:
        class0.append(ImPatches[i])
    elif label[i] == 1 and len(class1) < 20:
        class1.append(ImPatches[i])
    elif label[i] == 2 and len(class2) < 20:
        class2.append(ImPatches[i])
Classes = [class0,class1,class2]
test_size = (len(class0)+len(class1)+len(class2))/(4*3)
test = class0[:test_size] + class1[:test_size] + class2[:test_size]
test_label = test_size*[0] + test_size*[1] + test_size*[2]
train0 = class0[test_size:]
train1 = class1[test_size:]
train2 = class2[test_size:]
train = [train0,train1,train2]

#Tuning the kernel hyper-parameter sigma, we can observe test error minimized
n_p = 3
sigma_p = numpy.array([0.1,1,10])    #set different sigma parameters
n_loc = 4
sigma_loc = numpy.array([0.1,1,10,100])
error = numpy.zeros((n_p,n_loc))
for p_index in range(n_p):
    for loc_index in range(n_loc):
        predict = NearestMean(test,train,sigma_p[p_index],sigma_loc[loc_index])   #predict test set
        for i in range(3*test_size):
            if abs(predict[i] - test_label[i]) > 0.01:
                error[p_index,loc_index] += 1.0/(3*test_size)
        print (p_index,loc_index)
#plot the test error for different parameter values
plt.imshow(error,cmap='hot')
plt.colorbar()
plt.ylabel('sigma_p index')
plt.xlabel('sigma_loc index')