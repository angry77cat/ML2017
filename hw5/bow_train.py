import numpy as np
# import os
# os.environ["THEANO_FLAGS"] = "device=gpu"
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
import pickle
import matplotlib.pyplot as plt
import json
import nltk
nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

train_path = 'train_data.csv'
test_path = 'test_data.csv'
output_path = 'pred.csv'

#####################
###   parameter   ###
#####################
np.random.seed(123)
split_ratio = 0.1
embedding_dim = 47905
nb_epoch = 1000
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
    for i, word in enumerate(word_index):
        # print(word)
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

    ### nltk stop words
    stop_words = set(stopwords.words('english'))
    word_tokens= []
    for _, string in enumerate(all_corpus):
        word_token= word_tokenize(string) 
        filtered_sentence = [w for w in word_token if not w in stop_words]
        f_sens= ''
        for _, string2 in enumerate(filtered_sentence):
            f_sens += string2 + ' '
        word_tokens.append(f_sens)
    print ('filtered out stop words')

    ### tokenizer for all data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(word_tokens)
    word_index = tokenizer.word_index
    print ('Tokenized')

    ### store tokenizer, tag_list
    with open('bow_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle)
    with open('bow_tag_list.pickle', 'wb') as handle:
        pickle.dump(tag_list, handle)
    print ('Store tokenizer, tag_list')

    ### convert word texts to matrix
    all_sequences = tokenizer.texts_to_sequences(word_tokens)
    all_bw= tokenizer.sequences_to_matrix(all_sequences,mode= 'tfidf')
    print ('Bag of words')

    print ('multi_categorical')
    train_tag = to_multi_categorical(Y_data,tag_list) 

    x_data = all_bw[:np.asarray(X_data).shape[0], :]
    X_test = all_bw[-np.asarray(X_test).shape[0]:, :]
    (X_train,Y_train),(X_val,Y_val) = split_data(x_data,train_tag,split_ratio)


    # with open('bw_word_index.pickle', 'wb') as handle:
    #     pickle.dump(word_index, handle, protocol= pickle.HIGHEST_PROTOCOL)
    # with open('bw_tag_list.pickle', 'wb') as handle:
    #     pickle.dump(tag_list, handle)
    # with open('bw_word_docs.pickle', 'wb') as handle:
    #     pickle.dump(word_docs, handle)
    ### build model
    print ('Building model.')
    model = Sequential()
    model.add(Dense(256,activation='relu', input_shape= (embedding_dim, )))
    model.add(Dropout(0.2))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(38,activation='sigmoid'))
    model.summary()

    adam = Adam(lr=0.00125,decay=1e-4,clipvalue=0.2)
    # adam= Adam(lr= 0.0005, decay= 1e-3, clipvalue= 0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer= adam,
                  metrics=[f1_score])
   
    earlystopping = EarlyStopping(monitor='val_f1_score', patience = 15, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath='bow_best.h5',
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
                 verbose= 2,
                 callbacks=[earlystopping,checkpoint, hist])

    model_json = model.to_json()
    with open("bow_model.json", "w") as json_file:
        json_file.write(model_json)
    print("Saved model to disk")

    # plt.figure()
    # x_axis = range(1, nb_epoch+ 1)
    # plt.plot(x_axis, hist.tr_losses, 'r')
    # plt.plot(x_axis, hist.val_losses, 'b')
    # plt.xlabel('epoch')
    # plt.ylabel('losses')
    # plt.savefig('bw_loss.png')

    # plt.figure()
    # x_axis = range(1, nb_epoch+ 1)
    # plt.plot(x_axis, hist.tr_f1s, 'r')
    # plt.plot(x_axis, hist.val_f1s, 'b')
    # plt.xlabel('epoch')
    # plt.ylabel('f1_score')
    # plt.savefig('bw_f1s.png')

if __name__=='__main__':
    main()