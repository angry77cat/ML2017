import numpy as np
import pickle
import os
# os.environ["THEANO_FLAGS"] = "device=gpu"
import string
import sys
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
# import matplotlib.pyplot as plt

np.random.seed(123)

train_path = 'train_data.csv'
test_path = 'test_data.csv'

#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 100
nb_epoch = 50
batch_size = 128


################
###   Util   ###
################
class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]
        self.tr_f1s=[]
        self.val_f1s=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_f1s.append(logs.get('f1_score'))
        self.val_f1s.append(logs.get('val_f1_score'))


def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r', encoding= 'ISO-8859-1') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def get_embedding_dict(path):
    embedding_dict = {}
    with open(path,'r', encoding= 'ISO-8859-1') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

###########################
###   custom metrices   ###
###########################
def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

#########################
###   Main function   ###
#########################
def main():

    ### read training and testing data
    (Y_data,X_data,tag_list) = read_data(train_path,True)
    (_, X_test,_) = read_data(test_path,False)
    all_corpus = X_data + X_test
    print ('Find %d articles.' %(len(all_corpus)))
    
    ### tokenizer for all data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_corpus)
    word_index = tokenizer.word_index
    print ('Tokenized')
    
    ### store tokenizer, tag_list
    with open('rnn_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle)
    with open('rnn_tag_list.pickle', 'wb') as handle:
        pickle.dump(tag_list, handle)
    print ('Store tokenizer, tag_list')

    ### convert word sequences to index sequence
    print ('Convert to index sequences.')
    train_sequences = tokenizer.texts_to_sequences(X_data)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    ### padding to equal length
    print ('Padding sequences.')
    train_sequences = pad_sequences(train_sequences)
    max_article_length = train_sequences.shape[1]
    test_sequences = pad_sequences(test_sequences, maxlen=max_article_length)
    
    print ('multi_categorical')
    train_tag = to_multi_categorical(Y_data,tag_list) 
    
    ### split data into training set and validation set
    (X_train,Y_train),(X_val,Y_val) = split_data(train_sequences,train_tag,split_ratio)
    
    ### get mebedding matrix from glove
    print ('Get embedding dict from glove.')
    embedding_dict = get_embedding_dict('glove.6B.%dd.txt'%embedding_dim)
    print ('Found %s word vectors.' % len(embedding_dict))
    num_words = len(word_index) + 1
    print ('Create embedding matrix.')
    embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)

    
    ### build model
    print ('Building model.')
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_article_length,
                        trainable=False))
    model.add(GRU(128,activation='tanh',dropout=0.2))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(38,activation='sigmoid'))
    model.summary()

    adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=[f1_score])
   
    earlystopping = EarlyStopping(monitor='val_f1_score', patience = 15, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath='rnn_best.h5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_f1_score',
                                 mode='max')

    hist= History()
    model.fit(X_train, Y_train, 
                 validation_data=(X_val, Y_val),
                 epochs=nb_epoch, 
                 batch_size=batch_size,
                 verbose= 1,
                 callbacks=[earlystopping,checkpoint, hist])

    model_json = model.to_json()
    with open("rnn_model.json", "w") as json_file:
        json_file.write(model_json)
    print("Saved model_json to disk")

    # plt.figure()
    # x_axis = range(1, nb_epoch+ 1)
    # plt.plot(x_axis, hist.tr_losses, 'r')
    # plt.plot(x_axis, hist.val_losses, 'b')
    # plt.xlabel('epoch')
    # plt.ylabel('losses')
    # plt.savefig('loss.png')

    # plt.figure()
    # x_axis = range(1, nb_epoch+ 1)
    # plt.plot(x_axis, hist.tr_f1s, 'r')
    # plt.plot(x_axis, hist.val_f1s, 'b')
    # plt.xlabel('epoch')
    # plt.ylabel('f1_score')
    # plt.savefig('f1s.png')

if __name__=='__main__':
    main()