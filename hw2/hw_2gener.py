import numpy as np
from numpy.linalg import inv
import pandas as pd
import random as rd
import sys
import math

y_train= pd.read_csv(sys.argv[4], header= None).values
np.seterr( over='ignore' )
#csv
x_trainadd= pd.read_csv(sys.argv[1], header= None, encoding= 'big5',).values
x_trainadd= np.array(x_trainadd[:, 4].astype(float))
x_testadd= pd.read_csv(sys.argv[2], encoding= 'big5',).values
x_testadd= np.array(x_testadd[:, 4].astype(float))

drop_out_list= [' 10th',' 11th',' 12th',' 1st-4th',' 5th-6th',' 7th-8th'
	,' 9th',' Assoc-acdm',' Assoc-voc',' Bachelors',' Doctorate',' HS-grad',' Masters',
	' Preschool',' Prof-school',' Some-college']


x_train= pd.read_csv(sys.argv[3])
x_train_drop= x_train.drop(drop_out_list, axis= 1)
x_train= x_train_drop.values.astype(float)
x_train= np.insert(x_train, 6, x_trainadd, axis= 1) 
print(x_train.shape)

x_test= pd.read_csv(sys.argv[5])
x_test_drop= x_test.drop(drop_out_list, axis= 1)
x_test= x_test_drop.values.astype(float)
x_test= np.insert(x_test, 6, x_testadd, axis= 1) 
print(x_test.shape)

#normalize
comb= np.vstack((x_train,x_test))
for i in range(7): 
	maxm= comb[:,i].max()
	minm= comb[:,i].min()
	for n in range(len(x_train[:,i])):
		x_train[n, i]= (x_train[n, i]- minm)/(maxm-minm)
	for k in range(len(x_test[:,i])):
		x_test[k,i]= (x_test[k, i]- minm)/(maxm-minm)

def estimation(x_train, y_train, data_set):
	u1 = np.zeros((x_train.shape[1],)).astype(np.float)
	u2 = np.zeros((x_train.shape[1],)).astype(np.float)
	n1= 0
	n2= 0
	sigma1= np.zeros((x_train.shape[1], x_train.shape[1])).astype(np.float)
	sigma2= np.zeros((x_train.shape[1], x_train.shape[1])).astype(np.float)

	for n in data_set:
		if(y_train[n]== 1):
			u1 += x_train[n]
			n1= n1 + 1
		elif(y_train[n]== 0):
			u2 += x_train[n]
			n2= n2 + 1

	if(n1== 0):
		u2 = u2/ n2
		for n in data_set:
			sigma2 += np.dot(np.transpose([x_train[i] - u2]), (x_train- u2))
		sigma= sigma2/ n2
	elif(n2== 0):
		u1 = u1/ n1
		for n in data_set:
			sigma1 += np.dot(np.transpose([x_train[i] - u1]), (x_train- u1))
		sigma= sigma1/ n1
	else:
		u1 = u1/ n1
		u2 = u2/ n2 
		for n in data_set:
			sigma1 += np.dot(np.transpose([x_train[i] - u1]), [(x_train[i]- u1)])
			sigma2 += np.dot(np.transpose([x_train[i] - u2]), [(x_train[i]- u2)])
		sigma= (sigma1/ (n1+n2)) + (sigma2/ (n1+n2))

	print(sigma)
	return u1, u2, sigma, n1, n2

def sampleSelect(valid_size, seed):
	rd.seed(seed)
	valid_set= rd.sample(range(32561), valid_size)
	train_set= list(set(valid_set)^set(range(32561)))
	return valid_set, train_set

def sigmoid(z):
	res= 1/ (1.0 + np.exp(-z))
	return np.clip(res, 0.00000000000001, 0.99999999999999)

def print_result(x_train, y_train, val, b, w, valN, i):
	corr= 0
	for n in val:
		z= np.dot(w, x_train[n])+ b
		f= sigmoid(z)
		# print(f)
		if(np.abs(y_train[n] - f) < 0.5):
			corr= corr+1
	accr= corr/ valN
	print('i:', i,'accr:', accr)

def testOutput(b,w, x_test):
	result= []
	idName= []
	for n in range(16281):
		z= np.dot(w, x_train[n])+ b
		f= sigmoid(z)
		if(f > 0.5):
			result.append(1)
		else:
			result.append(0)
		idName.append(n+1)
	data= np.hstack((np.asarray(idName).reshape(16281,1), np.asarray(result).reshape(16281,1)))
	pd.DataFrame(data, columns= ['id', 'label']).to_csv(sys.argv[6], index= None)

i= 0
valN= 6400
u1, u2, sigma, n1, n2= estimation(x_train, y_train, range(32561))

inv_sigma= np.linalg.inv(sigma)
w= np.dot((u1-u2), inv_sigma)
b= (-0.5)* np.dot(np.dot([u1], inv_sigma) , u1) + (0.5)* np.dot(np.dot([u2], inv_sigma),u2) + np.log(n1/n2)
print((-0.5)* np.dot(np.dot([u1], inv_sigma) , u1) + (0.5)* np.dot(np.dot([u2], inv_sigma),u2))
print(np.dot(np.dot([u2], inv_sigma),u2))

val, train= sampleSelect(valN, 127)

print_result(x_train, y_train, val, b, w, valN, i)

testOutput(b,w,x_test)