import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from vis.utils import utils
from vis.visualization import visualize_saliency, visualize_activation
from utils import io
import sys


model= load_model(sys.argv[1])
print('Model loaded.')
l= 19
print('layer name: ', model.layers[l].name)

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = ''
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name]

feats, labels, _= io.read_dataset('train')

image_paths= []

for i in range(1):
    image_paths.append(feats[i].reshape((48,48)))

fig= plt.figure()
heatmaps = []
ttt=[]
i= 1
s= 0
for path in image_paths:
    x = path
    x = x.reshape((48,48,1))
    t = x.reshape((1,48,48,1))

    pred_class = np.argmax(model.predict(t))

    print(pred_class, labels[s])

    
    heatmap = visualize_saliency(model, l, filter_indices=[pred_class],
     seed_img= x)
    print(heatmap.shape)
    heatmaps.append(heatmap)

    i += 1
    s += 1



plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.colorbar()
plt.title('Saliency map')
plt.show()
