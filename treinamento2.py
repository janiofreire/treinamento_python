from keras.preprocessing.text import *
from keras.models import *
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

GLOBAL_MAP = {}
GLOBAL_FILE_W2V_PATH = ""
GLOBAL_VEC_COLUMNS_SIZE=0

def pearson(y_true, y_pred):
    fsp = y_pred - K.mean( y_pred)
    fst = y_true - K.mean(y_true)

    devP = K.std(y_pred)
    devT = K.std(y_true)

    return K.mean(fsp * fst) / (devP * devT)


def cleanNumber(x):
    return x[3].replace("\n","")


def loadWordToVec(path, quantColumns):
    file = open(path, mode="r")

    global GLOBAL_MAP
    global GLOBAL_VEC_COLUMNS_SIZE

    file.readline()

    for line in file:
        split_data = line.split()
        index = 1
        word = split_data[0]
        quant_columns_vec = len(split_data) -1

        if quant_columns_vec > GLOBAL_VEC_COLUMNS_SIZE:
            local_dif = quant_columns_vec-GLOBAL_VEC_COLUMNS_SIZE
            index += local_dif
            for w in split_data[1:index]:
                word += " " + w

        GLOBAL_MAP[word] = np.asarray(split_data[index:], dtype='float32')

    file.close()

    return


def openFileSentencePairSimilarity(filePath):
    file = open(filePath, mode="r")
    lines = file.readlines()
    sentences_result = [i.split("\t")[1:] for i in lines]

    for i in sentences_result:
        i[2] = float(i[2].replace("\n",""))
        i[2] = i[2] / 5

    file.close()

    return sentences_result


def sentenceToVecDefault(sentence):
    if len(GLOBAL_MAP) < 1:
        global GLOBAL_FILE_W2V_PATH
        global GLOBAL_VEC_COLUMNS_SIZE
        loadWordToVec(GLOBAL_FILE_W2V_PATH, GLOBAL_VEC_COLUMNS_SIZE)

    list_words = text_to_word_sequence(sentence, lower=True)
    vec_result = []

    for word in list_words:
        vect_values = GLOBAL_MAP.get(word)
        if vect_values is not None:
            for i in range(len(vect_values)):
                if i > len(vec_result)-1:
                    vec_result.append(0)
                vec_result[i] += vect_values[i]

    return [v/len(list_words) for v in vec_result]


def mergeVector(s1, s2):
    vec_dif = [s1[i]-s2[i] for i in range(len(s1))]
    vec_multi = np.multiply(s1, s2)
    return [vec_dif[i]-vec_multi[i] for i in range(len(vec_dif))]


def vectorizeSentence(dados_param, sentenceToVec, mergeVector):
    merge_result=[]
    results=[]

    for dado in dados_param:
        s1 = sentenceToVec(dado[0])
        s2 = sentenceToVec(dado[1])
        merge = mergeVector(s1, s2)
        merge_result.append(merge)
        if len(dado) > 2:
            results.append(dado[2])
    return merge_result, results


def execute_experiment(pathTreino, pathValidation, pathWordEmbemding,numColumnVector):
    training_data = openFileSentencePairSimilarity(pathTreino)
    validate_data = openFileSentencePairSimilarity(pathValidation)
    global GLOBAL_FILE_W2V_PATH
    global GLOBAL_VEC_COLUMNS_SIZE
    GLOBAL_FILE_W2V_PATH = pathWordEmbemding
    GLOBAL_VEC_COLUMNS_SIZE = numColumnVector
    vec_sentence_training, result_t = vectorizeSentence(training_data, sentenceToVecDefault, mergeVector)
    vec_sentence_validate, result_v = vectorizeSentence(validate_data, sentenceToVecDefault, mergeVector)

    traini_data(vec_sentence_training, result_t, vec_sentence_validate, result_v)

    return


def traini_data(vec_sentence_training, result_t, vec_sentence_validate, result_v):
    global GLOBAL_VEC_COLUMNS_SIZE
    sequence_input = Input(shape=(GLOBAL_VEC_COLUMNS_SIZE,), dtype='float32')
   # x = Conv1D(128, 5, activation='relu')
    #x = MaxPooling1D(5)(x)
    #x = Conv1D(128, 5, activation='relu')(x)
    #x = MaxPooling1D(5)(x)
    #x = Conv1D(128, 5, activation='relu')(x)
    #x = MaxPooling1D(35)(x)  # global max pooling
    #x = Flatten()(x)
    #x = Dense(128, activation='relu')(x)
    #preds = Dense(1, activation='softmax')(x)

    #preds = Dense(1, activation='softmax')

    #model = Model(sequence_input, preds)

    model = Sequential()
    model.add(Dense(1, input_dim=GLOBAL_VEC_COLUMNS_SIZE, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam'
                  #,metrics=[pearson]
                  )

    x_train = pad_sequences(vec_sentence_training, maxlen=GLOBAL_VEC_COLUMNS_SIZE)
    y_train = to_categorical(np.asarray(result_t))
    x_val = pad_sequences(vec_sentence_validate, maxlen=GLOBAL_VEC_COLUMNS_SIZE)
    y_val = to_categorical(np.asarray(result_v))

    # happy learning!
    model.fit(x_train, y_train, validation_data=(x_val, y_val)
              #,epochs=2, batch_size=128
              )
    return


def main():
    execute_experiment("/media/janio/cdd58d38-6ee5-49db-aeca-f74319d6e461/dados/Propor/treino_2016/treino.csv",
     "/media/janio/cdd58d38-6ee5-49db-aeca-f74319d6e461/dados/Propor/gold_2016/assin-test-gold/propor_gold_ptbr.csv",
     "/media/janio/cdd58d38-6ee5-49db-aeca-f74319d6e461/dados/Dados_Treinamento/cbow_s300.txt",300)


if __name__ == '__main__':
   main()