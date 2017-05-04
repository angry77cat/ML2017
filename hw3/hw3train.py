# import os
# os.environ["THEANO_FLAGS"] = "device=gpu"

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from numpy.random import normal
from keras.utils import np_utils
import read
import themodel
import sys

x_train, y_train, x_val, y_val = read.train_split_data(sys.argv[1],127,5652)

data_gen= ImageDataGenerator(width_shift_range= 0.1,height_shift_range= 0.1,
	rotation_range= 35,preprocessing_function= lambda X:X + normal(scale= 0.05, size=X.shape) )

early_stop= EarlyStopping(patience= 3, verbose= 1)
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
callbacks_list.append(early_stop)

model= themodel.cnn()

for i in range(30):
	print(i)
	model.fit_generator(data_gen.flow(x_train,y_train, batch_size= 75),
		epochs= 20, steps_per_epoch= len(x_train)/75, validation_data= 
		data_gen.flow(x_val,y_val,batch_size= 75), validation_steps= 5652/75, 
		callbacks= callbacks_list)


