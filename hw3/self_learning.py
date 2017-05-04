from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.callbacks import Callback
from numpy.random import normal
from keras.utils import np_utils
import matplotlib.pyplot as plt
import read
import themodel
import sys
class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]
        self.tr_accs=[]
        self.val_accs=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))


data_gen= ImageDataGenerator(rotation_range= 30, width_shift_range= 0.1,
	height_shift_range= 0.1, preprocessing_function= lambda X:X + normal(scale= 0.05, size=X.shape) )

early_stop= EarlyStopping(patience= 3, verbose= 1)
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
callbacks_list.append(early_stop)
model= themodel.cnn()
hist= History()
callbacks_list.append(hist)


x_train, y_train, x_val, y_val = read.train_split_data(sys.argv[1],127,5652)

split= 1000
x_train_split= []
y_train_split= []

print(x_train.shape)
# range(round(x_train.shape[0]/ split))
for i in range(23):
	print('self training: part ',i,' of ', round(x_train.shape[0]/ split))
	if i== 0:
		x_train_split.append(x_train[i*split: (i+1)*split])
		y_train_split.append(y_train[i*split: (i+1)*split])
		x_train_split= np.asarray(x_train_split).reshape((split, 48,48,1))
		y_train_split= np.asarray(y_train_split).reshape((split, 7))
		continue	

	print('training data size: ', x_train_split.shape[0])
	for n in range(1):
		print('for loop:', n)
		model.fit_generator(data_gen.flow(x_train_split,y_train_split, batch_size= 100),
			epochs= 2, steps_per_epoch= len(x_train_split)/100, validation_data= 
			data_gen.flow(x_val,y_val,batch_size= 100), validation_steps= 5652/100, 
			callbacks= callbacks_list)

	if i== 1:
		print(hist.tr_accs)
		plt_tr= hist.tr_accs
		plt_val= hist.val_accs
	elif i>1:
		plt_tr= np.hstack((plt_tr, hist.tr_accs))
		plt_val= np.hstack((plt_val, hist.val_accs))

	print('predicting the next set')
	pred= model.predict_classes(x_train[i*split: (i+1)*split])
	pred= to_categorical(np.asarray(pred),num_classes=7).reshape((split,7))

	if i+1 > (x_train.shape[0]/ split):
		break

	new_set= np.asarray(x_train[i*split: (i+1)*split])
	x_train_split= np.vstack((x_train_split, new_set))
	y_train_split= np.vstack((y_train_split, pred))

x= range(plt_tr[0]) #batch, for loop, epoch multipulate
plt.plot(x, plt_tr)
plt.xlabel('epoch')
plt.ylabel('accr')
plt.show()
