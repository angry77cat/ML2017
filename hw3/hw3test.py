from keras.models import load_model
import sys
import os
import numpy as np
import pandas as pd
from utils import io
import read

base_dir = os.path.dirname(os.path.realpath(__file__))
# data_dir = os.path.join(base_dir,'storemodel')

x_test= read.test_data(sys.argv[1])

model= load_model(os.path.join('{}.hdf5'.format('weights-improvement-04-0.63bestK65')))

y_test= model.predict_classes(x_test)

id_name= []
for i in range(len(y_test)):
	id_name.append(i)
data= np.hstack((np.asarray(id_name).reshape(len(y_test),1), y_test.reshape(len(y_test),1)))
pd.DataFrame(data, columns= ['id', 'label']).to_csv(sys.argv[2], index= None)