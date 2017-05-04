from keras.utils import plot_model
from keras import layers
import sys
from keras.models import load_model

plot_model(load_model(sys.argv[1]), to_file='model.png',show_shapes= True)
layer= load_model(sys.argv[1])
config= layer.get_config()

# print(config[17])