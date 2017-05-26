import numpy as np
import pickle
import sys
import nltk
nltk.download("stopwords")
nltk.download("punkt")
from keras.models import model_from_json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

test_path= sys.argv[1]
output_path= sys.argv[2]

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


def main():
    with open('bow_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        print("Loaded bow_tokenizer from disk")

    with open('bow_tag_list.pickle', 'rb') as handle:
        tag_list= pickle.load(handle)
        print("Loaded bow_tag_list from disk")

    with open('bow_model.json', 'rb') as json:
        loaded_model = json.read()
        print("Loaded bow_model from disk")

    model = model_from_json(loaded_model)
    model.load_weights('bow_best.h5')
    print("Loaded bow_model weight from disk")

    (_, X_test,_) = read_data(test_path,False)

    ### nltk stop words
    stop_words = set(stopwords.words('english'))
    word_tokens= []
    for _, string in enumerate(X_test):
        word_token= word_tokenize(string) 
        filtered_sentence = [w for w in word_token if not w in stop_words]
        f_sens= ''
        for _, string2 in enumerate(filtered_sentence):
            f_sens += string2 + ' '
        word_tokens.append(f_sens)
    print ('filtered out stop words')

    ### convert word texts to matrix
    test_sequences = tokenizer.texts_to_sequences(word_tokens)
    test_sequences = tokenizer.sequences_to_matrix(test_sequences,mode= 'tfidf')
    print ('Bag of words')

    print("Predicting")
    Y_pred = model.predict(test_sequences)
    thresh = 0.4
    with open(output_path,'w') as output:
        print ('\"id\",\"tags\"',file=output)
        Y_pred_thresh = (Y_pred > thresh).astype('int')
        for index,labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

if __name__=='__main__':
    main()

