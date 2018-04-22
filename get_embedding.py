from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np

count = 0 
def genMatrix(paragraph, model, glove_model, modelVocab , WWords) :
    word_tokens = word_tokenize(paragraph)

    global count 

    matrix = []
    leng = len(word_tokens)
    print("length of para ", leng)
    count = count+leng
    for i in range(0,leng) :
        if word_tokens[i] in modelVocab :
            matrix.append(glove_model[word_tokens[i]])
        elif word_tokens[i] in WWords:
            matrix.append(model[word_tokens[i]])
        else :
            leftout = np.random.uniform(low=-0.9, high=0.9, size=50)
            matrix.append(leftout) 
    return matrix


def main () :
    glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors_50d.txt", binary = False)
    glove_modelVocab = list(glove_model.wv.vocab)
    glove_modelVocab = set(glove_modelVocab)

    model = Word2Vec.load("word2vec_vectors_50d")
    WWords = list(model.wv.vocab)
    WWords = set(WWords)


    print("Glove Word2Vec model loaded...")

    data = open('test1.txt', 'r', encoding='utf-8').read()
    trainData = data.split('\n')
  
    paramax = 396
    ques_max = 33
    # for i in range(len(trainData)) :
    for i in range(1) :
        temp = []
        temp = trainData[i].split('\t')
        mat_paragraph = genMatrix(temp[0], model, glove_model, glove_modelVocab,WWords)
        
        addition = [0]*50

        diff = paramax - len(mat_paragraph)
        for j in range(0,diff) :
            mat_paragraph.append(addition)
        
        mat_question = genMatrix(temp[1], model, glove_model, glove_modelVocab,WWords)

        diff = ques_max - len(mat_question)
        for j in range(0,diff) :
            mat_question.append(addition)
        addition = []
        yield np.asarray(mat_paragraph) , np.asarray(mat_question)


if __name__ == "__main__" :
    main()