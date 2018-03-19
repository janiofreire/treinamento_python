from keras.preprocessing.text import *
from keras.models import *
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from scipy.stats import pearsonr
from tensorflow.contrib.metrics import streaming_pearson_correlation
import numpy as np
import tensorflow as tf
import json
import pickle

GLOBAL_MAP = {}
GLOBAL_FILE_W2V_PATH = ""
GLOBAL_VEC_COLUMNS_SIZE = 0


def pearson(y_true, y_pred):
    result = streaming_pearson_correlation(y_true, y_pred)
    return result[0]


def pearson3(y_true, y_pred):
    fsp = y_pred - K.mean(y_pred)
    fst = y_true - K.mean(y_true)

    dev_p = K.std(y_pred)
    dev_t = K.std(y_true)

    return (fsp * fst) / (dev_p * dev_t)


def pearson2(y_true, y_pred):
    fsp = y_pred - K.mean(y_pred)
    fst = y_true - K.mean(y_true)

    dev_p = K.std(y_pred)
    dev_t = K.std(y_true)

    return K.mean(fsp * fst) / (dev_p * dev_t)


def clean_number(x):
    return x[3].replace("\n", "")


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


def is_pair_file(sentences_results):
    line = next(iter(sentences_results or []), None)
    return len(line) == 6


def open_file_sentence_pair_similarity(file_path):
    file = open(file_path, mode="r")
    lines = file.readlines()
    sentences_result = [i.split("\t")[1:] for i in lines]
    index = 4 if is_pair_file(sentences_result) else 2

    for i in sentences_result:
        i[index] = float(i[index].replace("\n", ""))
        i[index] = i[index] / 5

    file.close()

    return sentences_result


def load_global_data():
    if len(GLOBAL_MAP) < 1:
        global GLOBAL_FILE_W2V_PATH
        global GLOBAL_VEC_COLUMNS_SIZE
        load_word_to_vec(GLOBAL_FILE_W2V_PATH, GLOBAL_VEC_COLUMNS_SIZE)
    return


def sentence_to_vec_default(line, index):
    sentence = line[index]

    load_global_data()

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


def create_pair_vector(pairs, sentence_to_vec_function, merge_vector_function):
    vector_result = []

    load_global_data()

    for pair in pairs:
        vec1 = GLOBAL_MAP.get(pair['c1'].lower())
        vec2 = GLOBAL_MAP.get(pair['c2'].lower())
        if vec1 is None:
            vec1 = np.zeros(GLOBAL_VEC_COLUMNS_SIZE)
        if vec2 is None:
            vec2 = np.zeros(GLOBAL_VEC_COLUMNS_SIZE)
        vector_result.append(merge_vector_function(vec1, vec2))

    return vector_result


def vectorize_sentence_pair(dados_param, sentence_to_vec_function, merge_vector_function):
    merge_result = []
    results = []

    for dado in dados_param:
        pairs = json.loads(dado[5])
        merge_result.append(create_pair_vector(pairs, sentence_to_vec_function, merge_vector_function))
        results.append(dado[4])

    return merge_result, results


def merge_vector(s1, s2):
    vec_dif = [s1[i]-s2[i] for i in range(len(s1))]
    vec_multi = np.multiply(s1, s2)
    return [vec_dif[i]-vec_multi[i] for i in range(len(vec_dif))]


def merge_vector_combine(s1, s2):
    return np.append(s1, s2)


def vectorize_sentence(dados_param, sentence_to_vec_function, merge_vector_function):
    merge_result = []
    results = []

    for dado in dados_param:
        s1 = sentence_to_vec_function(dado, 0)
        s2 = sentence_to_vec_function(dado, 1)
        merge = merge_vector_function(s1, s2)
        merge_result.append(merge)
        if len(dado) > 2:
            results.append(dado[2])

    return merge_result, results


def create_default_model(num_column_vector):
    model = Sequential()
    model.add(Dense(1, input_dim=num_column_vector, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='softmax'))

    return model


def crete_maia_layer(num_column_vector, train_data):
    model = Sequential()
    model.add(LSTM(output_dim=num_column_vector,
                   return_sequences=False,
                   input_shape=(train_data.shape[1], train_data.shape[2])))
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(1, init='normal'))

    return model


def traini_data(x_train, y_train, x_val, y_val, num_column_vector, mode_creator, function_evaluation, is_pair):
    if is_pair is False:
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_val = np.reshape(x_val, (x_val.shape[0], 1, x_val.shape[1]))

    if mode_creator is not None:
        model = mode_creator(num_column_vector, x_train)
    else:
        model = create_default_model(num_column_vector)

    model.compile(loss='mean_squared_error',
                  optimizer='adam'
                  , metrics=[
            'mse'
            #            pearson
        ])

    model.fit(x_train, y_train
              # , validation_data=(x_val, y_val)
              , epochs=30, batch_size=128)

    if function_evaluation is None:
        scores = model.evaluate(x_val, y_val, verbose=1)
    else:
        scores = function_evaluation(model, x_val, y_val)
    print("Person: " + str(scores[0]))

    return model


def pearson_evaluation(model, x_val, y_val):
    values2 = []
    values = model.predict(x_val)
    values = values.tolist()
    [values2.append(x[0]) for x in values]

    return pearsonr(values2, y_val.tolist())


def execute_experiment(path_treino, path_validation, path_word_embemding, num_column_vector, model_creation,
                       function_evaluation, slice_teste, vectoriz_function, sentence_to_vec_function, merge_vector_function, is_pair):
    training_data = open_file_sentence_pair_similarity(path_treino)
    validate_data = open_file_sentence_pair_similarity(path_validation)
    global GLOBAL_FILE_W2V_PATH
    global GLOBAL_VEC_COLUMNS_SIZE

    GLOBAL_FILE_W2V_PATH = path_word_embemding
    GLOBAL_VEC_COLUMNS_SIZE = num_column_vector

    vec_sentence_training, result_t = vectoriz_function(training_data, sentence_to_vec_function, merge_vector_function)
    vec_sentence_validate, result_v = vectoriz_function(validate_data, sentence_to_vec_function, merge_vector_function)

    vec_sentence_training = pad_sequences(vec_sentence_training, maxlen=num_column_vector, dtype='float32')
    result_t = np.asarray(result_t)
    vec_sentence_validate = pad_sequences(vec_sentence_validate, maxlen=num_column_vector, dtype='float32')
    result_v = np.asarray(result_v)

    model = None

    if slice_teste is not None and slice_teste is True:
        x_slice = np.split(vec_sentence_validate, 2)
        y_slice = np.split(result_v, 2)
        for i in range(len(x_slice)):
            vec_sentence_training = np.concatenate([vec_sentence_training, x_slice[i]])
            result_t = np.concatenate([result_t, y_slice[i]])
            index = 0 if i == 1 else 1
            vec_sentence_validate = np.concatenate([vec_sentence_validate, x_slice[index]])
            result_v = np.concatenate([result_v, y_slice[index]])
            traini_data(vec_sentence_training, result_t, vec_sentence_validate, result_v, num_column_vector,
                        model_creation, function_evaluation, is_pair)
    else:
        model = traini_data(vec_sentence_training, result_t, vec_sentence_validate, result_v, num_column_vector, model_creation,
                    function_evaluation, is_pair)

    return model


def main():
    tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    path_corpus = "/media/janio/cdd58d38-6ee5-49db-aeca-f74319d6e461/dados/Resultados/"
    model =  execute_experiment(path_corpus+"result_dice_ptbr_propor_train_2016.csv",
    #path_corpus+"assin-ptbr-test.csv",
    path_corpus+"result_dice_ptbr_propor_dev_2016.csv",
     "/media/janio/cdd58d38-6ee5-49db-aeca-f74319d6e461/dados/Dados_Treinamento/glove_s300.txt",
                       300,
                       crete_maia_layer,
                       pearson_evaluation,
                       False,
#                       vectorize_sentence,
                       vectorize_sentence_pair,
                       sentence_to_vec_default,
#                       merge_vector
                       merge_vector_combine,
                       True
                       )
    output = open('/media/janio/cdd58d38-6ee5-49db-aeca-f74319d6e461/dados/modelos/modelo.pkl', 'wb')
    pickle.dump(model, output)

if __name__ == '__main__':
   main()