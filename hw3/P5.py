import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from vis.utils import utils
from vis.visualization import visualize_saliency, visualize_activation
from utils import io
import sys


model= load_model(sys.argv[1])
print('Model loaded.')
l= 0
print('layer name: ', model.layers[l].name)

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = ''
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name]

# feats, labels, _= io.read_dataset('train')

# image_paths= []

# for i in range(1):
    # image_paths.append(feats[i].reshape((48,48)))

# fig= plt.figure()
acts=[]
fig= plt.figure()

for i in range(30):
    act= visualize_activation(model, l, filter_indices= i, seed_img=None, text=None, \
    act_max_weight=1, lp_norm_weight=10, tv_weight=10).reshape((48,48))
    acts= fig.add_subplot(5,10,i+1)
    acts.imshow(act,cmap= 'gray')
    plt.axis('off')
    
# plt.show()
print('plotting')
# plt.imshow(utils.stitch_images(acts, 1))
# # plt.imshow(ttt.reshape((48,48)))
# plt.colorbar()
# plt.title('Saliency map')
plt.show()
