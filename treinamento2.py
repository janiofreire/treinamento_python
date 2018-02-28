from keras.preprocessing.text import *
from keras.models import *
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from scipy.stats import pearsonr
from tensorflow.contrib.metrics import streaming_pearson_correlation
import numpy as np

GLOBAL_MAP = {}
GLOBAL_FILE_W2V_PATH = ""
GLOBAL_VEC_COLUMNS_SIZE=0


def pearson(y_true, y_pred):
    result = streaming_pearson_correlation(y_true, y_pred)
    return result[0]


def pearson3(y_true, y_pred):
    fsp = y_pred - K.mean(y_pred)
    fst = y_true - K.mean(y_true)

    devP = K.std(y_pred)
    devT = K.std(y_true)

    return (fsp * fst) / (devP * devT)


def pearson2(y_true, y_pred):
  fsp = y_pred - K.mean(y_pred)
  fst = y_true - K.mean(y_true)

  devP = K.std(y_pred)
  devT = K.std(y_true)

  return K.mean(fsp * fst) / (devP * devT)


def clean_number(x):
    return x[3].replace("\n","")


def load_word_to_vec(path, quant_columns):
    file = open(path, mode="r")

    global GLOBAL_MAP

    file.readline()

    for line in file:
        split_data = line.split()
        index = 1
        word = split_data[0]
        quant_columns_vec = len(split_data) - 1

        if quant_columns_vec > quant_columns:
            local_dif = quant_columns_vec - quant_columns
            index += local_dif
            for w in split_data[1:index]:
                word += " " + w

        GLOBAL_MAP[word] = np.asarray(split_data[index:], dtype='float32')

    file.close()

    return


def open_file_sentence_pair_similarity(filePath):
    file = open(filePath, mode="r")
    lines = file.readlines()
    sentences_result = [i.split("\t")[1:] for i in lines]

    for i in sentences_result:
        i[2] = float(i[2].replace("\n",""))
        i[2] = i[2] / 5

    file.close()

    return sentences_result


def sentence_to_vec_default(sentence):
    if len(GLOBAL_MAP) < 1:
        global GLOBAL_FILE_W2V_PATH
        global GLOBAL_VEC_COLUMNS_SIZE
        load_word_to_vec(GLOBAL_FILE_W2V_PATH, GLOBAL_VEC_COLUMNS_SIZE)

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


def merge_vector(s1, s2):
    vec_dif = [s1[i]-s2[i] for i in range(len(s1))]
    vec_multi = np.multiply(s1, s2)
    return [vec_dif[i]-vec_multi[i] for i in range(len(vec_dif))]


def vectorize_sentence(dados_param, sentence_to_vec, merge_vector):
    merge_result=[]
    results=[]

    for dado in dados_param:
        s1 = sentence_to_vec(dado[0])
        s2 = sentence_to_vec(dado[1])
        merge = merge_vector(s1, s2)
        merge_result.append(merge)
        if len(dado) > 2:
            results.append(dado[2])
    return merge_result, results


def create_default_model(num_column_vector):
    model = Sequential()
    model.add(Dense(1, input_dim=num_column_vector, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='softmax'))

    return model

def crete_maia_layer():
    model = Sequential()
    model.add(LSTM(return_sequences=False))
    model.add(Dense(output_dim=100, activation='relu', init='glorot_uniform', name='module3'))
    return model


def traini_data(vec_sentence_training, result_t, vec_sentence_test, result_v, num_column_vector, mode_creator, function_evaluation):
    sequence_input = Input(shape=(num_column_vector,), dtype='float32')
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

    model = None

    if mode_creator is not None:
        model = mode_creator()
    else:
        model = create_default_model(num_column_vector)

    model.compile(loss='mean_squared_error',
                  optimizer='adam'
                  , metrics=[pearson])

    x_train = pad_sequences(vec_sentence_training, maxlen=num_column_vector, dtype='float32')
   # y_train = to_categorical(np.asarray(result_t), num_classes=None)
    y_train = np.asarray(result_t)
    x_val = pad_sequences(vec_sentence_test, maxlen=num_column_vector, dtype='float32')
 #   y_val = to_categorical(np.asarray(result_v), num_classes=None)
    y_val = np.asarray(result_v)

    # happy learning!
    model.fit(x_train, y_train
              #, validation_data=(x_val, y_val)
          , epochs=100, batch_size=128)
    if function_evaluation is None:
        scores = model.evaluate(x_val, y_val, verbose=1)
    else:
        scores = function_evaluation(model, x_val,y_val)

    print("Person: " + str(scores[1]))
    return


def pearson_evaluation(model, x_val, y_val):
    values = []
    x_list = x_val.tolist()
    y_list = y_val.tolist()

    [values.append(model.predict(x)) for x in x_list]

    return pearsonr(values, y_list)


def execute_experiment(path_treino, path_validation, path_word_embemding, num_column_vector, model_creation, function_evaluation):
    training_data = open_file_sentence_pair_similarity(path_treino)
    validate_data = open_file_sentence_pair_similarity(path_validation)
    global GLOBAL_FILE_W2V_PATH
    global GLOBAL_VEC_COLUMNS_SIZE

    GLOBAL_FILE_W2V_PATH = path_word_embemding
    GLOBAL_VEC_COLUMNS_SIZE = num_column_vector

    vec_sentence_training, result_t = vectorize_sentence(training_data, sentence_to_vec_default, merge_vector)
    vec_sentence_validate, result_v = vectorize_sentence(validate_data, sentence_to_vec_default, merge_vector)

    traini_data(vec_sentence_training, result_t, vec_sentence_validate, result_v, num_column_vector, model_creation, function_evaluation)

    return


def main():
    execute_experiment("/media/janio/cdd58d38-6ee5-49db-aeca-f74319d6e461/dados/Propor/treino_2016/treino.csv",
     "/media/janio/cdd58d38-6ee5-49db-aeca-f74319d6e461/dados/Propor/trial/trial-500.csv",
    #"/media/janio/cdd58d38-6ee5-49db-aeca-f74319d6e461/dados/Propor/gold_2016/assin-test-gold/propor_gold_ptbr.csv",
     "/media/janio/cdd58d38-6ee5-49db-aeca-f74319d6e461/dados/Dados_Treinamento/glove_s300.txt", 300, None, None)


if __name__ == '__main__':
   main()