# from keras import applications
from keras.models import load_model
import sys
import numpy as np
from keras import backend as K

# build the VGG16 network
model = load_model('/Users/alxperlc/Desktop/ML_hw3/new2/weights-improvement-04-0.63bestK65.hdf5')

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

layer_name = 'conv2d_1'
input_img= model.input
filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer

# we start from a gray image with some noise
input_img_data = np.random.random((1, 1, 48, 48)) * 20 + 128.
layer_output = layer_dict[layer_name].output
loss = K.mean(layer_output[:,:,:,filter_index])
grads = K.gradients(loss, input_img)[0]
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
iterate = K.function([K.learning_phase()], [loss, grads])

# build a loss function that maximizes the activation
# of the nth filter of the layer considered
# layer_output = layer_dict[layer_name].output


# # compute the gradient of the input picture wrt this loss
# grads = K.gradients(loss, input_img)[0]

# # normalization trick: we normalize the gradient
# grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# # this function returns the loss and grads given the input picture
# iterate = K.function([input_img], [loss, grads])


# run gradient ascent for 20 steps
for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step

from scipy.misc import imsave

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

img = input_img_data[0]
img = deprocess_image(img)
imsave('%s_filter_%d.png' % (layer_name, filter_index), img)
