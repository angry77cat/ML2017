from keras.models import load_model
import numpy as np
import itertools
import sys
from keras.utils.np_utils import to_categorical
from utils import io
import matplotlib.pyplot as plt

model_path = sys.argv[1]
emotion_classifier = load_model(model_path)
np.set_printoptions(precision=2)
dev_feats, dev_labels, _ = io.read_dataset('valid')

ite= 0
count= 0
target= 2
fig= plt.figure()
bins= np.asarray([0,1,2,3,4,5,6])
while count<= 5:
    if dev_labels[ite]== target:
    	# if count<=5 :
    	# 	h= 1
    	# 	count += 1
    	# else:
	    ax= fig.add_subplot(3,4,2*(count)+1)
	    ax.imshow(dev_feats[ite].reshape((48,48)))
	    plt.xlabel(dev_labels[ite])
	    plt.tight_layout()
	    prob= emotion_classifier.predict_proba(dev_feats[ite].reshape((1,48,48,1)))
	    print(prob.reshape((7)))
	    ax= fig.add_subplot(3,4,2*(count)+2)
	    plt.bar(bins, prob.reshape((7)))
	    plt.xticks(bins, [0,1,2,3,4,5,6])
	    plt.xlabel('emotion code')
	    plt.ylabel('prob.')
	    count += 1
    ite += 1
plt.show()