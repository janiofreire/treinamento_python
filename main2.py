from keras.preprocessing.text import *
from keras.models import *
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
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

import numpy as np

def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    mult = norma*normb
    cos = dot_product/mult
    return cos

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

def create_pair_sep_vector(pairs, sentence_to_vec_function, merge_vector_function):
    vector_result = np.zeros((GLOBAL_VEC_COLUMNS_SIZE,2,GLOBAL_VEC_COLUMNS_SIZE))

    load_global_data()

    for i,pair in enumerate(pairs):
        vec1 = GLOBAL_MAP.get(pair['c1'].lower())
        vec2 = GLOBAL_MAP.get(pair['c2'].lower())
        if vec1 is None:
            vec1 = np.zeros(GLOBAL_VEC_COLUMNS_SIZE)
        if vec2 is None:
            vec2 = np.zeros(GLOBAL_VEC_COLUMNS_SIZE)
        vector_result[i][0] = vec1
        vector_result[i][1] = vec2

    return vector_result



def create_pair_cosine(pairs):
    vector_result_cos = np.zeros((300))
    load_global_data()
   
    for i,pair in enumerate(pairs):
        vec1 = GLOBAL_MAP.get(pair['c1'].lower())
        vec2 = GLOBAL_MAP.get(pair['c2'].lower())
      
        if vec1 is None:
            vec1 = np.zeros(GLOBAL_VEC_COLUMNS_SIZE)
        if vec2 is None:
            vec2 = np.zeros(GLOBAL_VEC_COLUMNS_SIZE)
        vector_result = abs(cos_sim(vec1,vec2))
        print(vector_result) 
        if not(vector_result is np.nan):
            #vector_result[i] = pair['r']
            if float(vector_result) >= 0.5:
                vector_result_cos[i] = 1.0
            else:
                vector_result_cos[i] = 0.0
        #vec2 = GLOBAL_MAP.get(pair['c2'].lower())
        #if vec is None:
        #    vec1 = np.zeros(GLOBAL_VEC_COLUMNS_SIZE)
        #vector_result.append(vec)
    
    return vector_result_cos




def create_pair_dice(pairs):
    vector_result = np.zeros((300))

    load_global_data()
   
    for i,pair in enumerate(pairs):
        if not(pair['r'] is np.nan):
            #vector_result[i] = pair['r']
            if float(pair['r']) >= 0.66:
                vector_result[i] = "1.0"
            else:
                vector_result[i] = "0.0"
        #vec2 = GLOBAL_MAP.get(pair['c2'].lower())
        #if vec is None:
        #    vec1 = np.zeros(GLOBAL_VEC_COLUMNS_SIZE)
        #vector_result.append(vec)
    
    return vector_result


def vectorize_sentence_pair(dados_param, sentence_to_vec_function, merge_vector_function):
    merge_result = []
    results = []
   
    for dado in dados_param:
        pairs = json.loads(dado[5])
        merge_result.append(create_pair_vector(pairs, sentence_to_vec_function, merge_vector_function))
        results.append(dado[4])

    return merge_result, results


def vectorize_sentence_pair_dice(dados_param, sentence_to_vec_function, merge_vector_function):
    result_dice = []
    merge_result = []
    results = []
   
    for dado in dados_param:
        pairs = json.loads(dado[5])
        #merge_result.append(create_pair_vector(pairs, sentence_to_vec_function, merge_vector_function))
        merge_result.append(create_pair_vector(pairs, sentence_to_vec_function, merge_vector_function))
        result_dice.append(create_pair_dice(pairs))
        results.append(dado[4])

    return merge_result,result_dice, results  

def vectorize_sentence_pair_cos(dados_param, sentence_to_vec_function, merge_vector_function):
    result_cos = []
    merge_result = []
    results = []
   
    for dado in dados_param:
        pairs = json.loads(dado[5])
        #merge_result.append(create_pair_vector(pairs, sentence_to_vec_function, merge_vector_function))
        merge_result.append(create_pair_sep_vector(pairs, sentence_to_vec_function, merge_vector_function))
        result_cos.append(create_pair_cosine(pairs))
        results.append(dado[4])

    return merge_result,result_cos, results  
    
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

def crete_maia_layer(num_column_vector, train_data,dice_layer):
    model1 = Sequential()
    model1.add(LSTM(output_dim=num_column_vector,
                   return_sequences=True,
                   input_shape=(train_data.shape[1], train_data.shape[2])))
    model1.add(LSTM(output_dim=num_column_vector,
                   return_sequences=False,
                   input_shape=(train_data.shape[1], train_data.shape[2])))
    model1.add(Dense(output_dim=num_column_vector,input_dim=train_data.shape[2],activation='relu'))
    #model1.add(LSTM(output_dim=num_column_vector,
    #               return_sequences=False,
    #               input_shape=(train_data.shape[1], train_data.shape[2])))   
    #model1.add(TimeDistributed(LSTM(100, batch_input_shape=(None, train_data.shape[1], train_data.shape[2],train_data.shape[3]), return_sequences=False), batch_input_shape=(None, train_data.shape[1], train_data.shape[2], train_data.shape[3])))

            #model.add(LSTM(100, batch_input_shape=(None, clauses.shape[1],clauses.shape[2],clauses.shape[3])))

    #model1.add(LSTM(100, batch_input_shape=(None, train_data.shape[1],train_data.shape[2])))

    model2 = Sequential()
    model2.add(Dense(num_column_vector, input_dim=dice_layer.shape[1], init='normal', activation='relu'))
    #model2.add(Dense(num_column_vector, activation='relu'))
    
    final_model = Merge([model1,model2], mode='concat')
    model = Sequential()
    model.add(final_model)
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(1, init='normal',activation='sigmoid'))
    print(model.summary())

    return model


def traini_data(x_train,dice_train, y_train, x_val,dice_val, y_val,x_test,dice_test, y_test, num_column_vector, mode_creator, function_evaluation, is_pair):
    #train_data_size = round(len(x_train)*0.8)
    if is_pair is False:

        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_val = np.reshape(x_val, (x_val.shape[0], 1, x_val.shape[1]))

    if mode_creator is not None:
        model = mode_creator(num_column_vector,x_train,dice_train)
    else:
        model = create_default_model(num_column_vector)
    
    model.compile(loss='mean_squared_error',
                  optimizer='adam'
                  , metrics=[
            'mse'
            #            pearson
        ])

    early_stopping = EarlyStopping(monitor='val_loss', patience=40)
    checkpointer = ModelCheckpoint(filepath='graph_ee_kfold.hdf5',
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True,
                    mode='min')

    model.fit([x_train,dice_train], y_train
              , validation_data=([x_val,dice_val], y_val)
              , nb_epoch=70, batch_size=400
              , callbacks=[early_stopping,checkpointer]
               )

    model.load_weights('graph_ee_kfold.hdf5')
    X_test = [x_test,dice_test]
    if function_evaluation is None:
       
        scores = model.evaluate([x_test,dice_test], y_test, verbose=1)
    else:
        scores = function_evaluation(model, dice_test, y_test)
    print("Person: " + str(scores[0]))

    return model



#def traini_data(x_train, y_train, x_val, y_val, num_column_vector, mode_creator, function_evaluation, is_pair):
#    if is_pair is False:
#        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
#        x_val = np.reshape(x_val, (x_val.shape[0], 1, x_val.shape[1]))

#    if mode_creator is not None:
#        model = mode_creator(num_column_vector, x_train)
#    else:
#        model = create_default_model(num_column_vector)
    
#     model.compile(loss='mean_squared_error',
#                  optimizer='adam'
#                  , metrics=[
#            'mse'
            #            pearson
#        ])
#    early_stopping = EarlyStopping(monitor='loss', patience=50)

#                checkpointer = ModelCheckpoint(filepath='graph_ee_kfold.hdf5',
#                    monitor='loss',
#                    verbose=1,
#                    save_best_only=True,
#                    mode='min')

#    model.fit(x_train, y_train
#              , validation_data=(x_val, y_val)
#              , epochs=100, batch_size=128,)

#    if function_evaluation is None:
#        scores = model.evaluate(x_val, y_val, verbose=1)
#    else:
#        scores = function_evaluation(model, x_val, y_val)
#    print("Person: " + str(scores[0]))

#    return model


def pearson_evaluation(model, x_val, y_val):
    values2 = []
    values = model.predict(x_val)
    values = values.tolist()
    
    [values2.append(float("{0:.2f}".format(x[0]))) for x in values]
    print(values2)
    print(y_val)
    return pearsonr(values2, y_val.tolist())


def execute_experiment(path_treino, path_validation,path_testing, path_word_embemding, num_column_vector, model_creation,
                       function_evaluation, slice_teste, vectoriz_function, sentence_to_vec_function, merge_vector_function, is_pair):
    training_data = open_file_sentence_pair_similarity(path_treino)
    validate_data = open_file_sentence_pair_similarity(path_validation)
    test_data = open_file_sentence_pair_similarity(path_validation)

    global GLOBAL_FILE_W2V_PATH
    global GLOBAL_VEC_COLUMNS_SIZE

    GLOBAL_FILE_W2V_PATH = path_word_embemding
    GLOBAL_VEC_COLUMNS_SIZE = num_column_vector

    vec_sentence_training, vector_dice_train, result_t = vectoriz_function(training_data, sentence_to_vec_function, merge_vector_function)

    vec_sentence_validate, vector_dice_validate, result_v = vectoriz_function(validate_data, sentence_to_vec_function, merge_vector_function)
    
    vec_sentence_test, vector_dice_test, result_te = vectoriz_function(test_data, sentence_to_vec_function, merge_vector_function)
    
    vec_sentence_training = pad_sequences(vec_sentence_training, maxlen=num_column_vector, dtype='float32')
    result_t = np.asarray(result_t)
    
    vec_sentence_validate = pad_sequences(vec_sentence_validate, maxlen=num_column_vector, dtype='float32')
    result_v = np.asarray(result_v)
    
    vec_sentence_test = pad_sequences(vec_sentence_test, maxlen=num_column_vector, dtype='float32')
    result_te = np.asarray(result_te)

    
    vector_dice_train = np.array(vector_dice_train)
    vector_dice_validate = np.array(vector_dice_validate)   
    vector_dice_test = np.array(vector_dice_test)
    model = None
    print("training embedding vectors shape")
    print(vec_sentence_training.shape)
    print("training dice vectors shape")
    print(vector_dice_train.shape)
    print("training data output")
    print(result_t.shape)

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
    #    model = traini_data(vec_sentence_training, result_t, vec_sentence_validate, result_v, num_column_vector, model_creation,
    #                function_evaluation, is_pair)
         model = traini_data(vec_sentence_training,vector_dice_train, result_t, vec_sentence_validate,vector_dice_validate, result_v, vec_sentence_test,vector_dice_test, result_te, num_column_vector, model_creation,
                    function_evaluation, is_pair)
    return model


def main():
    tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    path_corpus = "/home/sousa/janio_project/"
    model =  execute_experiment(path_corpus+"result_max(dice_nc)_ptbr_propor_train_2016.csv",
    #path_corpus+"assin-ptbr-test.csv",
    path_corpus+"result_max(dice_nc)_ptbr_propor_dev_2016.csv",
    path_corpus+"result_dice_ptbr_propor_test_2016.csv",
    path_corpus+"glove_s300.txt",
                       300,
                       crete_maia_layer,
                       pearson_evaluation,
                       False,
#                       vectorize_sentence,
                       #vectorize_sentence_pair_dice,
                       vectorize_sentence_pair_dice,
                       sentence_to_vec_default,
#                       merge_vector
                       merge_vector_combine,
                       True
                       )
    output = open(path_corpus+'modelo.pkl', 'wb')
    pickle.dump(model, output)

if __name__ == '__main__':
   main()
