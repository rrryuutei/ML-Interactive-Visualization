import numpy
#This project deals with a time series of face images, changing from one person to another.
#We use kernel based method to estimate when does the change happen.

import scipy
import matplotlib.pyplot as plt

#Getting the LFW dataset
from sklearn.datasets import fetch_lfw_people
LFW = fetch_lfw_people(min_faces_per_person=15)  #requiring the min faces, we get images of two people
n_img = len(LFW.target)
TS = []    #create time series
label = []
#sort the images
for i in range(n_img):
    if LFW.target[i] == 0:
        TS.insert(0,LFW.images[i,:,:])
        label.insert(0,0)
    elif LFW.target[i] == 1:
        TS.append(LFW.images[i,:,:])
        label.append(1)
#plot the series of the images
plt.figure(0)
for i in range(n_img):
    plt.subplot(6,11,i+1)
    im = TS[i]
    plt.imshow(im,cmap='gray')
    plt.axis('off')
plt.show

#In this part we define the kernel between two images. We split the images into small square patches,
#while keeping the spatial continuity of the images.

#Gaussian kernel as the building block of the image kernel
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
                    h_patch = k_rbf(p1,p2,sigma_p)   #similarity between the two patches
                    h_loc = k_rbf(numpy.array([i1*p,j1*p]),numpy.array([i2*p,j2*p]),sigma_l) #spatial peparation between the two patches
                    k += h_patch*h_loc
    return k

#Transform the raw images into matrices of patches to be fed to the kernel
Patches = []
p = 5   #square region size
step = (p+1)/2
for im in TS:
    im = im[:-1,:] - im[1:,:]
    [h,w] = numpy.shape(im)
    p_ind0 = 0     #initial starting corner point of a patch
    while p_ind0 <= h - p:
        p_ind1 = 0
        while p_ind1 <= w - p:
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
M = (h-p+step) / step
N = (w-p+step) / step
for i in range(n_img):
    patches = Patch_scaled[i*BagSize:(i+1)*BagSize]
    PatchMat = numpy.zeros((M,N,p**2))
    for j in range(BagSize):
        m = j / N
        n = j % N
        PatchMat[m,n,:] = patches[j]
    ImPatches.append(PatchMat)
#compute the kernel matrix for several values of kernel hyper-parameters
n_p = 5
sigma_p = numpy.array([0.01,0.1,1,10,100])    #set different sigma parameters
sigma_loc = 1
K_mats = []
for p_index in range(n_p):
    K = numpy.zeros((n_img,n_img))
    for i in range(n_img):
        for j in range(i+1):
            K[i,j] = K_img(ImPatches[i],ImPatches[j],sigma_p[p_index],sigma_loc)
            K[j,i] = K[i,j]
    K_mats.append(K)
#display the kernel matrix with sigma_p=1
plt.figure(1)
plt.imshow(K_mats[2])
plt.colorbar()

#This part formulate an estimator of the change point through the kernel matrix of the image series
#Function F_n gives the whole value of F_n(t) for t from 1 to n-1, given the kernel matrix K
def F_n(K):
    n = int(numpy.shape(K)[0])
    F = []
    for t in range(1,n):
        u_t = numpy.ones(t) / t
        u_nt = numpy.ones(n-t) / (t-n)
        u = numpy.hstack((u_t,u_nt))
        Ft = t*(n-t)/n*numpy.dot(u.T,numpy.dot(K,u))
        F.append(Ft)
    return F
#plot the F_n(t) function for several kernel hyper-parameter values
plt.figure(2)
for i in range(1,n_p):
    plt.subplot(2,2,i)
    plt.plot(F_n(K_mats[i]))
    plt.xlabel('t')
    plt.ylabel('Fn(t)')
    plt.title('sigma_p = ' + str(sigma_p[i]))
    plt.subplots_adjust(wspace=0.6,hspace=0.6)
#We estimate the argmax of F_n(t), and then plot the estimation error
t_star = 44    #the correct turing point
error = []
for i in range(n_p):
    t_hat = numpy.argmax(F_n(K_mats[i])) + 1
    error.append(numpy.abs(t_hat - t_star))
plt.figure(3)
plt.plot(sigma_p,error)
plt.xscale('log')
plt.xlabel('sigma_p')
plt.ylabel('Change-point estimate error')



