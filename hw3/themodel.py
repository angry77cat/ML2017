from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization


def cnn():
	model = Sequential()
	model.add(Conv2D(30,kernel_size= (3,3), activation= 'relu', input_shape= (48,48,1)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size= (2,2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(60,kernel_size= (3,3), activation= 'relu', input_shape= (48,48,1)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size= (2,2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(120,kernel_size= (3,3), activation= 'relu', input_shape= (48,48,1)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size= (2,2)))
	model.add(Dropout(0.2))

	model.add(Flatten())

	model.add(Dense(units= 250, activation= 'relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(units= 250, activation= 'relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(units= 7, activation= 'softmax'))

	model.summary()
	model.compile(loss= 'categorical_crossentropy', optimizer= 'adam',
		metrics=['accuracy'])
	return model

def dnn():
	model = Sequential()
	model.add(Flatten(input_shape= (48,48,1)))
	model.add(Dense(units= 185, activation= 'relu'))
	model.add(Dense(units= 220, activation= 'relu'))
	model.add(Dense(units= 250, activation= 'relu'))
	model.add(Dense(units= 220, activation= 'relu'))
	model.add(Dense(units= 200, activation= 'relu'))
	# model.add(Dense(units= 200, activation= 'relu'))
	model.add(Dense(units= 7, activation= 'softmax'))

	model.summary()
	model.compile(loss= 'categorical_crossentropy', optimizer= 'adam',
		metrics=['accuracy'])
	return model


