from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

base_dir= os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir,'hand')

# read_img(subject):
im_datas= []
for i in range(481):
	dir1= os.path.join(data_dir, '{}.png'.format('hand.seq'+str(i+1)))
	img= Image.open(dir1)
	img = img.resize((64,60),Image.ANTIALIAS)
	im_datas.append(np.asarray(img, dtype= np.uint16))
im_datas= np.asarray(im_datas).reshape((481,64*60))

np.savez('hand.npz', X=im_datas )

from sklearn.svm import LinearSVR as SVR
from p3_2_gen import get_eigenvalues

# Train a linear SVR
npzfile = np.load('p3_2_data.npz')
# print(npzfile)
X = npzfile['X']
y = npzfile['y']

# we already normalize these values in gen.py
# X /= X.max(axis=0, keepdims=True)

print('SVR')
svr = SVR(C=1)
svr.fit(X, np.log(y))

# svr.get_params() to save the parameters
# svr.set_params() to restore the parameters

print('Predicting')
# predict
testdata = np.load('hand.npz')
test_X = []
for i in range(1):
    data = testdata['X']
    vs = get_eigenvalues(data)
    test_X.append(vs)

test_X = np.array(test_X)
pred_y = svr.predict(test_X)

print('Output')
with open('p3_2_ans.csv', 'w') as f:
    print('SetId,LogDim', file=f)
    for i, d in enumerate(pred_y):
        print(f'{i},{d}', file=f)

# singular value
sing_vals = []

u, s, v = np.linalg.svd(im_datas - im_datas.mean(axis=0))
print(s)
# s /= s.max()
# sing_vals.append(s)
# sing_vals = np.array(sing_vals).mean(axis=0)
# return sing_vals
