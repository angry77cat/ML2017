from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

base_dir= os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir,'faceExpressionDatabase')
alphabat= {0:"a", 1:"b", 2:"c", 3:"d",\
4:"e", 5:"f", 6:"g", 7:"h", 8:"i", 9:"j"}
# read_img(subject):
im_datas= []
for i1 in range(10):
	for i2 in range(10):
		dir1= os.path.join(data_dir, '{}.bmp'.format(alphabat[i1]+'0'+str(i2)))
		img= Image.open(dir1)
		im_datas.append(np.asarray(img, dtype= np.uint16))

# import data
im_datas= np.asarray(im_datas)
im_datas= im_datas.reshape((100,64,64))
x= im_datas.reshape((100,64*64))

# mean
summed= np.zeros((64,64))
for i in range(100):
	summed += im_datas[i]
mx= (summed/100).reshape((64*64,))

# origin-mean
lx= np.zeros((x.shape[0], 64*64))
for i in range(x.shape[0]):
	lx[i]= x[i]- mx

# plot mean
fig= plt.figure()
plt.imshow(mx.reshape((64,64)), cmap= 'gray')
plt.axis('off')
print('plot avg face')
plt.savefig('averageFace.png')

# SVD
U, S, V= np.linalg.svd(lx)
S= np.diag(S)

fig= plt.figure()
for i in range(9):
	eigface= V[i,:].reshape((64,64))
	eigfaces= fig.add_subplot(3,3,i+1)
	eigfaces.imshow(eigface, cmap= 'gray')
	plt.axis('off')
print('plot eig face')
plt.savefig('eigenFace.png')


# show original faces
fig= plt.figure()
for i in range(100):
	faces= fig.add_subplot(10,10,i+1)
	faces.imshow(im_datas[i], cmap= 'gray')
	plt.axis('off')
print('plot ori face')
plt.savefig('originalFace.png')

# reconstruction by top 5 eigenface

for n_com in [4]:
	Uk= U[:,0:n_com]
	Sk= S[0:n_com, 0:n_com]
	Vk= V[0:n_com,:]
	recon1= np.dot(np.dot(Uk,Sk),Vk)
	recon2= np.zeros((100,64*64))
	for i in range(recon1.shape[0]):
		recon2[i,:]+= (recon1[i,:]+ mx)

	fig= plt.figure()
	for i1 in range(100):
		reconfaces= fig.add_subplot(10,10,i1+1)
		reconfaces.imshow(recon2[i1,:].reshape((64,64)), cmap= 'gray')
		plt.axis('off')
	plt.savefig('reconstructedFace.png')
print('plotted recon face')

# dimensionality
for n_com in range(100):
	Uk= U[:,0:n_com]
	Sk= S[0:n_com, 0:n_com]
	Vk= V[0:n_com,:]
	recon1= np.dot(np.dot(Uk,Sk),Vk)
	recon2= np.zeros((100,64*64))
	for i in range(recon1.shape[0]):
		recon2[i,:]+= (recon1[i,:]+ mx)
	err = np.sqrt(np.average(((x-recon2)/256)**2))
	if err<0.01:
		print('error minor to 1%', 'at', n_com+1)
		break


