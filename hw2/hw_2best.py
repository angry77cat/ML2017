import numpy as np
import pandas as pd
import random as rd
import sys
import math

y_train= pd.read_csv(sys.argv[4], header= None).values

x_trainadd= pd.read_csv(sys.argv[1], header= None, encoding= 'big5',).values
x_trainadd= np.array(x_trainadd[:, 4].astype(float))
x_testadd= pd.read_csv(sys.argv[2], encoding= 'big5',).values
x_testadd= np.array(x_testadd[:, 4].astype(float))



drop_null= []
drop_out_list= [' 10th',' 11th',' 12th',' 1st-4th',' 5th-6th',' 7th-8th'
	,' 9th',' Assoc-acdm',' Assoc-voc',' Bachelors',' Doctorate',' HS-grad',' Masters',
	' Preschool',' Prof-school',' Some-college']

drop_out_relationship= [' Husband',' Not-in-family',' Other-relative',
	' Own-child',' Unmarried',' Wife']
drop_out_nation= [' Cambodia',' Canada',' China',' Columbia',' Cuba',' Dominican-Republic',
	' Ecuador',' El-Salvador',' England',' France',' Germany',' Greece',' Guatemala',
	' Haiti',' Holand-Netherlands',' Honduras',' Hong',' Hungary',' India',' Iran',' Ireland',
	' Italy',' Jamaica', ' Japan',' Laos',' Mexico',' Nicaragua',' Outlying-US(Guam-USVI-etc)',
	' Peru',' Philippines',' Poland',' Portugal',' Puerto-Rico',' Scotland',' South',
	' Taiwan',' Thailand',' Trinadad&Tobago',' United-States',' Vietnam',' Yugoslavia',
	'?_native_country']
drop_out_race=[' Amer-Indian-Eskimo',' Asian-Pac-Islander',' Black',' Other',' White']

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

#scaling
# comb= np.vstack((x_train,x_test))
# for i in range(7): 
# 	maxm= comb[:,i].max()
# 	minm= comb[:,i].min()
# 	for n in range(len(x_train[:,i])):
# 		x_train[n, i]= (x_train[n, i]- minm)/(maxm-minm)
# 	for k in range(len(x_test[:,i])):
# 		x_test[k,i]= (x_test[k, i]- minm)/(maxm-minm)

#normalize
for i in range(7):
	mean= x_train[:,i].mean()
	var= np.sqrt(x_train[:,i].var())
	for n in range(len(x_train[:,i])):
		x_train[n,i]= (x_train[n,i]-mean)/var
	for k in range(len(x_test[:,i])):
		x_test[k,i]= (x_test[k,i]-mean)/var

for i in range(7):
	x_trainadd2= x_train[:,i]*x_train[:,i]
	x_train= np.insert(x_train, i+7, x_trainadd2, axis= 1)
	x_testadd2= x_test[:,i]*x_test[:,i]
	x_test= np.insert(x_test, i+7, x_testadd2, axis= 1)

# for i in range(7):
# 	x_trainadd3= x_train[:,i]*x_train[:,i]*x_train[:,i]
# 	x_train= np.insert(x_train, i+14, x_trainadd3, axis= 1)
# 	x_testadd3= x_test[:,i]*x_test[:,i]*x_test[:,i]
# 	x_test= np.insert(x_test, i+14, x_testadd3, axis= 1)

def sampleSelect(valid_size, seed):
	rd.seed(seed)
	valid_set= rd.sample(range(32561), valid_size)
	train_set= list(set(valid_set)^set(range(32561)))
	return valid_set, train_set

def cont(preEnd, size, max, train):
	data = []
	for n in range(size):
		preEnd= preEnd + 1 
		if(preEnd>= max):
			preEnd = preEnd % max
		data.append(train[preEnd])
	return preEnd, data

def grad(data_set, x_data, y_data, b, w, b_grad, w_grad, lam):
	lossGrad= 0
	for n in data_set:
		z= np.sum(w * x_data[n]) + b
		# print(z)
		f= 1 / (1 + np.exp(-z))
		# print(y_data[n])
		lossGrad= -(y_data[n] - f) #Scalar
		b_grad= b_grad + lossGrad
		w_grad= w_grad + lossGrad * x_data[n]
	w_grad= w_grad + 2 * lam * w
	return b_grad, w_grad

def train_process(b_grad, w_grad, p, reNew, x_train, y_train, b, w, b_lr, w_lr, lr, valN, train, lam):
	p, input_set = cont(p, reNew, 32561-valN, train)
	b_grad, w_grad= grad(input_set, x_train, y_train, b, w, b_grad, w_grad, lam)
	b_lr= b_lr+ b_grad**2
	w_lr= w_lr+ w_grad**2
	b= b - (lr/ np.sqrt(b_lr)) * b_grad
	w= w - (lr/ np.sqrt(w_lr)) * w_grad
	return b_grad, w_grad, p, b, w, b_lr, w_lr

def print_result(x_train, y_train, val, train,  b, w, valN, i):
	corr= 0
	for n in val:
		z= np.sum(w * x_train[n]) + b
		f= 1 / (1 + np.exp(-z))
		if(np.abs(y_train[n] - f) < 0.5):
			corr= corr+1
	accr1= corr/ valN

	corr= 0
	for n in train:
		z= np.sum(w * x_train[n]) + b
		f= 1 / (1 + np.exp(-z))
		if(np.abs(y_train[n] - f) < 0.5):
			corr= corr+1
	accr2= corr/ (32561-valN)
	print('i:', i,'Val_accr:', accr1, 'Tra_accr:', accr2)
	return accr

# def print_parameter(b, w, x_train):
# 	para= np.zeros((x_train.shape[1],2)).astype(np.float)
# 	para[0,0]= b
# 	for i in range(x_train.shape[1]):
# 		para[i,1]= w[0,i]
# 	pd.DataFrame(para).to_csv(sys.argv[7], index= None)

# def store_parameter(w, w_history, ite, x_train):
# 	for n in range(x_train.shape[1]):
# 		w_history[n, ite]= w[0,n]
# 	pd.DataFrame(w_history).to_csv(sys.argv[8], index= None)

def testOutput(b,w, x_test):
	result= []
	idName= []
	for n in range(16281):
		z= np.sum(w * x_test[n]) + b
		f= 1 / (1 + np.exp(-z))
		if(f > 0.5):
			result.append(1)
		else:
			result.append(0)
		idName.append(n+1)
	data= np.hstack((np.asarray(idName).reshape(16281,1), np.asarray(result).reshape(16281,1)))
	pd.DataFrame(data, columns= ['id', 'label']).to_csv(sys.argv[6], index= None)


b= 0
w= np.zeros((1,x_train.shape[1]))
w_history= np.zeros((x_train.shape[1],1000))
b_lr= 1/1000000000000000
w_lr= np.ones((1,x_train.shape[1]))/1000000000000000
seed= 127
valN= 6400
lr= 0.01
reNew= 5
epoch= 50
iteration= 1000000
lam= 0.00001
p= 0
accr= 0.1
ite= 0

val, train= sampleSelect(valN, seed)
trainAll = range(32561)

for i in range(iteration):
	if(i%500001==0 and i<1000000):
		seed= seed + 1
		val, train= sampleSelect(valN, seed)

	# elif(i%1000== 0 and i<= 103000):
	# 	seed= seed + 1
	# 	val, train= sampleSelect(valN, seed)
	# elif(i%100== 0 and i<= 104000):
	# 	seed= seed + 1
	# 	val, train= sampleSelect(valN, seed)



	if(i%5== 0 and i<=200000):
		w_grad= np.zeros((1,x_train.shape[1])).astype(np.float)
		b_grad= 0
	elif(i>= 200000):
		w_grad= np.zeros((1,x_train.shape[1])).astype(np.float)
		b_grad= 0

	b_grad, w_grad, p, b, w, b_lr, w_lr= train_process(b_grad, w_grad, p, reNew, x_train, y_train, b, w, b_lr, w_lr, lr, valN, train, lam)
	

	if(i%10000==0):
		accr= print_result(x_train, y_train, val, train, b, w, valN, i)
		ite= ite +1

		# store_parameter(w, w_history, ite, x_train)

	# if(i<= 100000):
	# 	b_grad, w_grad, p, b, w, b_lr, w_lr= train_process(b_grad, w_grad, p, reNew, x_train, y_train, b, w, b_lr, w_lr, lr, 0, trainAll)
	# 	if(i%10000==0):
	# 		accr= print_result(x_train, y_train, val, b, w, valN, i)
	# 		ite= ite +1
	# 		store_parameter(w, w_history, ite)
	# elif(i>= 100000 and i< 103000):
	# 	reNew= 2500
	# 	b_grad, w_grad, p, b, w, b_lr, w_lr= train_process(b_grad, w_grad, p, reNew, x_train, y_train, b, w, b_lr, w_lr, lr, 0, trainAll)
	# 	if(i%100==0):
	# 		accr= print_result(x_train, y_train, val, b, w, valN, i)
	# 		ite= ite +1
	# 		store_parameter(w, w_history, ite)
	# elif(i>= 103000):
	# 	reNew= 25000
	# 	b_grad, w_grad, p, b, w, b_lr, w_lr= train_process(b_grad, w_grad, p, reNew, x_train, y_train, b, w, b_lr, w_lr, lr, 0, trainAll)
	# 	if(i%10==0):
	# 		accr= print_result(x_train, y_train, val, b, w, valN, i)
	# 		ite= ite +1
	# 		store_parameter(w, w_history, ite)




	# if(i%5000==0 and i<=200000):
	# 	seed= seed + 1
	# 	val, train= sampleSelect(valN, seed)

	# if(i%5== 0 and i<= 150000):
	# 	w_grad= np.zeros((1,106)).astype(np.float)
	# 	b_grad= 0
	# elif(i>= 200000):
	# 	w_grad= np.zeros((1,106)).astype(np.float)
	# 	b_grad= 0

	# if(accr< 0.8 and i>200000):
	# 	reNew= 25
	# 	b_grad, w_grad, p, b, w, b_lr, w_lr= train_process(b_grad, w_grad, p, reNew, x_train, y_train, b, w, b_lr, w_lr, lr, valN, train)
	# 	if(i%1000==0):
	# 		accr= print_result(x_train, y_train, val, b, w, valN, i)
	# elif(accr>= 0.8 and accr< 0.807 and i>200000):
	# 	# w_grad= np.zeros((1,106)).astype(np.float)
	# 	# b_grad= 0
	# 	reNew= 2500
	# 	b_grad, w_grad, p, b, w, b_lr, w_lr= train_process(b_grad, w_grad, p, reNew, x_train, y_train, b, w, b_lr, w_lr, lr, valN, train)
	# 	if(i%20==0):
	# 		accr= print_result(x_train, y_train, val, b, w, valN, i)
	# elif(accr>= 0.807 and accr< 0.815 and i>200000):
	# 	# w_grad= np.zeros((1,106)).astype(np.float)
	# 	# b_grad= 0
	# 	reNew= 6250
	# 	b_grad, w_grad, p, b, w, b_lr, w_lr= train_process(b_grad, w_grad, p, reNew, x_train, y_train, b, w, b_lr, w_lr, lr, valN, train)
	# 	if(i%4==0):
	# 		accr= print_result(x_train, y_train, val, b, w, valN, i)
	# elif(i>200000):
	# 	# w_grad= np.zeros((1,106)).astype(np.float)
	# 	# b_grad= 0
	# 	reNew= 25000
	# 	b_grad, w_grad, p, b, w, b_lr, w_lr= train_process(b_grad, w_grad, p, reNew, x_train, y_train, b, w, b_lr, w_lr, lr, valN, train)
	# 	if(i%1==0):
	# 		accr= print_result(x_train, y_train, val, b, w, valN, i)
	# else:
	# 	reNew= 25
	# 	b_grad, w_grad, p, b, w, b_lr, w_lr= train_process(b_grad, w_grad, p, reNew, x_train, y_train, b, w, b_lr, w_lr, lr, valN, train)
	# 	if(i%1000==0):
	# 		accr= print_result(x_train, y_train, val, b, w, valN, i)

testOutput(b,w, x_test)
# print_parameter(b,w, x_train)

exit()
