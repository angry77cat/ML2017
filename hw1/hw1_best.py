import numpy as np
import pandas as pd
import random as rd
import sys


#input data
trainarray= pd.read_csv(sys.argv[1], encoding= 'big5').values
train01= trainarray[:,3:27]
train02= np.vsplit(train01, 12)

train03= []
for i in range(12):
	if i==0:
		train03= np.vsplit(train02[i],20)
	elif i!=0:
		for j in range(20):
			train03.append(np.vsplit(train02[i],20)[j])

for i in range(240):
	if i== 0:
		train04= train03[0]
	else:
		train04= np.hstack((train04, train03[i]))

# Convert 'NR' to '0'
np.place(train04, train04=='NR', ['0.0'])

# Convert String to Float
train04= train04.astype(np.float)

x_train= []
for i in range(5760):
	if i%480<= 470: #Do not include the final 9hrs of the month 
		x_train.append(train04[:,i:i+9].reshape(162,1))


#preprocess y data
y_train= []
for i in range(5760):
	if i%480>= 9:
		y_train.append(train04[9,i])
#len(y_train)= 5652

###preprocessing stage end
def testOutput(b,w):
	testset = pd.read_csv(sys.argv[2], header= None,encoding= 'big5')
	testarray= testset.values
	test01= testarray[:,2:12]
	np.place(test01, test01=='NR', ['0.0'])
	test02= test01.astype(np.float)
	test04= np.vsplit(test02, 240)
	result= []
	for i in range(len(test04)):
		test04[i]= test04[i].reshape(162,1)
		result.append(sum(w * test04[i]) + b)
	idName= []
	ttest01= testarray[:,0]
	for i in range(len(ttest01)):
		if i%18== 0:
			idName.append(ttest01[i])
	data= np.hstack((np.asarray(idName).reshape(240,1),np.asarray(result).reshape(240,1)))
	pd.DataFrame(data, columns=['id','value'] ).to_csv(sys.argv[3],index= None)

def grad1st(data_set, x_data, y_data, b, w, b_grad, w_grad, lamda):
	lossGrad = 0
	for n in data_set:
		lossGrad= y_data[n] - b - sum(w * x_data[n])
		b_grad= b_grad - 2.0 * lossGrad * 1.0
		w_grad= w_grad - 2.0 * lossGrad * x_train[n] + 2 * lamda * w
	return b_grad, w_grad

def sampleSelect(valid_size, seed):
	rd.seed(seed)
	valid_set= rd.sample(range(5652), valid_size)
	train_set= list(set(valid_set)^set(range(5652)))
	return valid_set, train_set  ## pointer

def cont(preEnd, size, max):
	data = []
	for n in range(size):
		preEnd= preEnd + 1 
		if(preEnd>= max):
			preEnd = preEnd % max
		data.append(preEnd)
	return preEnd, data

def train_process(b_grad, w_grad, p, reNew, x_train, y_train, b, w, b_lr, w_lr, lamda):
	p, input_set = cont(p, reNew, 5652)
	b_grad, w_grad = grad1st(input_set, x_train, y_train, b, w, b_grad, w_grad, lamda)
	b_lr = b_lr + b_grad**2
	w_lr = w_lr + w_grad**2 
	b = b - lr/ np.sqrt(b_lr) * b_grad
	w = w - (lr/ np.sqrt(w_lr)) * w_grad
	return b_grad, w_grad, p, b, w, b_lr, w_lr

def print_result(x_train, y_train, b, w, valid_size, seed, premse):
	# print(i)
	premse= premse
	loss= 0
	mse= 0
	for n in range(1):
		val1, train1= sampleSelect(valid_size, seed+n)
		loss= 0
		for k in val1[:]:
			loss= loss + (y_train[k] - b - sum(w * x_train[k]))**2 + lamda * sum(w * w)
		mse= mse + np.sqrt(loss/len(val1))
	mse = mse / 1
	loss_history.append(mse)
	ite_history.append(i)
	# print(mse)
	return premse, mse

def record(mse, b, w):
	pd.DataFrame(mse).to_csv('./best_minmse.csv' ,index= None)
	pd.DataFrame(b).to_csv('./best_b.csv' ,index= None)
	pd.DataFrame(w).to_csv('./best_w.csv' ,index= None)

###parameters
b= 0 #initial
w= np.zeros((162,1)) #initial
lr= 0.015 #learning rate
iteration= 4700
reNew= 70
b_lr= 0.0
w_lr= np.zeros((162,1))
seed= 123
valN= 1000
p= -1
mse= 10
minmse= 100
loss_history= []
ite_history= []
lamda= 0
premse= 10

### Train set, Validation set selection
val, train= sampleSelect(valN, seed)


for i in range(iteration):
	if(i%1000== 0):
		print(i, "of 4700")
	# if(i>3000 and i%100== 0):
	# 	seed= seed+1
	# 	val, train= sampleSelect(valN, seed)

	### Initialization
	# if(mse<7 and (mse- premse)> 0.01):
	# 	b_grad= 0.0 
	# 	w_grad= np.zeros((162,1)).astype(np.float)

	### Processing
	if(mse>7):
		b_grad= 0.0 
		w_grad= np.zeros((162,1)).astype(np.float)
		b_grad, w_grad, p, b, w, b_lr, w_lr= train_process(b_grad, w_grad, p, reNew, x_train, y_train, b, w, b_lr, w_lr, lamda) 
		if(i%100== 0):
			premse, mse= print_result(x_train, y_train, b, w, valN, seed, mse)
		# if(mse[0]< minmse):
		# 	minmse= mse[0]
		# 	testOutput(b,w)
			# record(mse, b, w)

	elif(mse>6.5):
		reNew= 700
		b_grad, w_grad, p, b, w, b_lr, w_lr= train_process(b_grad, w_grad, p, reNew, x_train, y_train, b, w, b_lr, w_lr, lamda) 
		if(i%10== 0):
			premse, mse= print_result(x_train, y_train, b, w, valN, seed, mse)
		# if(mse[0]< minmse):
		# 	minmse= mse[0]
		# 	testOutput(b,w)
			# record(mse, b, w)

	elif(mse>6):
		reNew= 1400
		b_grad, w_grad, p, b, w, b_lr, w_lr= train_process(b_grad, w_grad, p, reNew, x_train, y_train, b, w, b_lr, w_lr, lamda) 
		if(i%5== 0):
			premse, mse= print_result(x_train, y_train, b, w, valN, seed, mse)
		# if(mse[0]< minmse):
		# 	minmse= mse[0]
		# 	testOutput(b,w)
			# record(mse, b, w)

	elif(mse>5):
		reNew= 5652
		b_grad, w_grad, p, b, w, b_lr, w_lr= train_process(b_grad, w_grad, p, reNew, x_train, y_train, b, w, b_lr, w_lr, lamda) 
		if(i%1== 0):
			premse, mse= print_result(x_train, y_train, b, w, valN, seed, mse)
		# if(mse[0]< minmse):
		# 	minmse= mse[0]
		# 	testOutput(b,w)
			# record(mse, b, w)

#Store parameter
testOutput(b,w)

exit()


