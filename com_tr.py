import collections
import csv
import numpy as np
import json
import string
import os
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

#motion_dic = {0:'go',1:'bring',2:'put',3:'find',4:'move',5:'search',6:'take',7:'think',8:'drop',9:'remove',10:'switch',11:'grab',12:'want',13:'let',14:'look',15:'control',16:'release',17:'send',18:'listen',19:'close',20:'turn',21:'clean',22:'feed',23:'lift'}
word_dic_path = './com_tr/command_dictionary'
motion_dic_path = './com_tr/motion_dictionary'
train_path = './com_tr/command_train3.tsv'
test_path = './com_tr/command_test.tsv'
com_model_path = './com_tr/command_model'
enable_debug = False
os.environ['PYTHONHASHSEED'] = '0'

def load_command_dataset(tsv_path):
    #Load the command dataset from a TSV file

    messages = []
    labels = []

    with open(tsv_path, 'r', newline='', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        i = 0

        for label, message in reader:
#            if enable_debug:
#                print("label, message",label, message)
            messages.append(message)
            labels.append(1 if label == 'com' else 0)
            i=i+1
    tsv_file.close()
    if enable_debug:
        print("total training data",i)
        print("labels",labels)
        print("positive samples", len(np.where(np.array(labels)==1)[0]))
    return messages, np.array(labels)

def load_com_dataset(tsv_path):
    #Load the command_motion dataset from a TSV file

    messages = []
    labels = []

    with open(tsv_path, 'r', newline='', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')

        for label, message in reader:
#            print("label, message",label, message)
            messages.append(message)
            labels.append(label)
    tsv_file.close()
    return messages, labels

def write_json(filename, value):
    """Write the provided value as JSON to the given filename"""
    with open(filename, 'w') as f:
        json.dump(value, f)

def write_tsv(tsv_path, value):
    """Write the provided value as JSON to the given filename"""

    data = value[1]
    input_msg = data.split(' ')
    if enable_debug:
        print("input_msg",input_msg)

    word_dict = load_word_dic(word_dic_path)
    word_dic_values = list(word_dict.values())
    for i in range(len(input_msg)):
        if input_msg[i] not in word_dic_values:
            add_dic(word_dict,input_msg[i])
            word_dic_values = list(word_dict.values())

    write_json(word_dic_path, word_dict)
    train_com_detect(train_path,test_path)

    with open(tsv_path, 'a', newline='') as tsv_file:
        tsv_w = csv.writer(tsv_file,delimiter='\t')
        tsv_w.writerow(value)
    tsv_file.close()

def error_analysis(prediction, labels, test_messages):
    """To verify which item was predicted wrong and print that item"""
    diff = prediction - labels
    idx = np.where(diff!=0)[0]
    if enable_debug:
        print("idx",idx)
        for i in range(len(idx)):
            print("the predicted wrong items:", idx, test_messages[int(idx[i])])

def predict_analysis(prediction, labels, test_messages):
    """To verify which item was predicted wrong and print that item"""
    if enable_debug:
        print("labels",len(labels),labels)
        print("prediction",len(prediction),prediction)
        for i in range(len(labels)):
            if prediction[i]!=labels[i]:
                print("the predicted wrong items:",i,test_messages[i])

def get_words(message):
    #Get the normalized list of words from a message string.

    msg = message.strip()
    msg = msg.split(' ')
    
    return [s.lower() for s in msg]


def create_dictionary(messages):
    #Create a dictionary mapping words to integer indices.

#    occ_count = 5
    occ_count = 1
    new_msg = []
    dic_key = 0
    dict = {}

    for i in range(len(messages)):
        new_msg.append(get_words(messages[i]))

    for i in range(len(new_msg)):
        for j in range(len(new_msg[i])):
            string = new_msg[i][j]

            #check if the string exists in at least 5 messages
            counts = 0
            for k in range(len(new_msg)):
                counts = counts + new_msg[k].count(string)

#            print("counts",counts)

            #if exists in at least 5 messages, then it should be in dictionary
            if counts >= occ_count:
                if dic_key == 0:
                    #first item
                    dict[dic_key] = string
                    dic_key = dic_key + 1
                else:
                    #Check if this sting already in dictionary, if not, add it in dictionary
                    if list(dict.values()).count(string) == 0:
                        dict[dic_key] = string
                        dic_key = dic_key + 1
#    print("dict",len(dict),dict)

    return dict


def transform_text(messages, word_dictionary):
    #Transform a list of text messages into a numpy array for further processing.

    dimi = len(messages)
    dimj = len(word_dictionary)
    matrix = np.zeros((dimi,dimj))
    dic_values = list(word_dictionary.values())
#    print("word_dictionary",dic_values)

    for i, message in enumerate(messages):
#        print("message",message,get_words(message))
        for word in get_words(message):
            if word in dic_values:
                key = dic_values.index(word)
#                print("word,key",word,key)
                matrix[i,key]+=1

#    for i in range(dimi):
#        message = get_words(messages[i])
#        print("message",message)
#        for j in range(len(message)):
#            print("len(message)",len(message),message[j])
#            if message[j] in word_dictionary.values():
#                key = list(word_dictionary.keys())[list (word_dictionary.values()).index(message[j])]
#                count = message.count(message[j])
#                matrix[i,key] = count

    return matrix

    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    n = labels.shape[0]
    Fi_y = len(np.where(labels==1)[0])/n

    m = matrix.shape[1]
    Fi_y_pos = np.zeros((m))
    Fi_y_neg = np.zeros((m))

    idx_pos = np.where(labels==1)[0]
    idx_neg = np.where(labels==0)[0]

#    Fi_y_pos1 = (matrix[labels==0]).sum(axis=0)+1
#    Fi_y_neg1 = (matrix[labels==1]).sum(axis=0)+1
#    Fi_y_pos1 /=Fi_y_pos1.sum()
#    Fi_y_neg1 /=Fi_y_neg1.sum()

    for i in range(m):

        pos=0
        maxp = 0
        for j in range(len(idx_pos)):
            if matrix[idx_pos[j],i] > 0:
                pos = pos + matrix[idx_pos[j],i]
                if matrix[idx_pos[j],i] > maxp:
                    maxp = matrix[idx_pos[j],i]
#        print("maxp",maxp)
        neg=0
        maxg=0
        for k in range(len(idx_neg)):
            if matrix[idx_neg[k],i] > 0:
                neg = neg + matrix[idx_neg[k],i]
                if matrix[idx_neg[k],i]>maxg:
                    maxg = matrix[idx_neg[k],i]
#        print("maxg",maxg)
        Fi_y_pos[i] = (1+pos)/(len(idx_pos)+maxp+1)
        Fi_y_neg[i] = (1+neg)/(len(idx_neg)+maxg+1)
#        Fi_y_pos[i] = (1+pos)/(len(idx_pos)+m)
#        Fi_y_neg[i] = (1+neg)/(len(idx_neg)+m)

    model = []
    model.append(Fi_y)
    model.append(Fi_y_pos)
    model.append(Fi_y_neg)

    com_model = {}
    com_model['Fi_y'] = Fi_y
    com_model['Fi_y_pos'] = Fi_y_pos.tolist()
    com_model['Fi_y_neg'] = Fi_y_neg.tolist()
#    com_model['Fi_y'] = np.log(Fi_y)
#    com_model['Fi_y_pos'] = np.log(Fi_y_pos).tolist()
#    com_model['Fi_y_neg'] = np.log(Fi_y_neg).tolist()

#    print("Fi_y_pos,Fi_y_neg",Fi_y_pos.shape,Fi_y_neg.shape)
    write_json('./com_tr/com_model', com_model)

    return model

    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix,training=True):
    #Use a Naive Bayes model to compute predictions for a target matrix.

    if training:
        Fi_y = model[0]
        Fi_y_pos = model[1]
        Fi_y_neg = model[2]
    else:
#        print("model",model)
        Fi_y = model['Fi_y']
        Fi_y_pos = np.array(model['Fi_y_pos'],dtype=object)
        Fi_y_neg = np.array(model['Fi_y_neg'],dtype=object)

#        print("Fi_y_pos,Fi_y_neg",Fi_y_pos.shape,Fi_y_neg.shape)
#        print(test)


    prediction = np.zeros((matrix.shape[0]))
#    print("matrix",matrix.shape)
    for i in range(matrix.shape[0]):
#        print("matrix[i]",matrix[i])
        idx = np.where(matrix[i]>0)[0]
#        print("idx",idx)

        for j in range(len(idx)):
            if j ==0:
                Fi_y_posp = matrix[i,idx[j]]*Fi_y_pos[idx[j]]
                Fi_y_negp = matrix[i,idx[j]]*Fi_y_neg[idx[j]]
            else:
                Fi_y_posp = Fi_y_posp*(matrix[i,idx[j]]*Fi_y_pos[idx[j]])

                Fi_y_negp = Fi_y_negp*(matrix[i,idx[j]]*Fi_y_neg[idx[j]])
        
        predict = Fi_y_posp*Fi_y / (Fi_y_posp*Fi_y + Fi_y_negp*(1-Fi_y))
#        predict = np.log(Fi_y_posp)*Fi_y / (np.log(Fi_y_posp)*Fi_y + np.log(Fi_y_negp)*(1-Fi_y))
#        predict = np.log(Fi_y_posp*Fi_y / (Fi_y_posp*Fi_y + Fi_y_negp*(1-Fi_y)))
        
        if predict>0.5:
            prediction[i] = 1
        else:
            prediction[i] = 0
#        print("prediction[i]",prediction[i])

#    probs_0 = (matrix*Fi_y_neg).sum(axis=1) + (1-Fi_y)
#    probs_1 = (matrix*Fi_y_pos).sum(axis=1) + Fi_y

#    output = (probs_1>probs_0).astype(int)
#    print("output",output)

    return prediction


def load_word_dic(path):
    tf = open(path, "r")
    word_dictionary = json.load(tf)
#    print("word_dictionary",word_dictionary)

    word_dict = {}
    for key, value in word_dictionary.items():
        word_dict[int(key)] = value

    return word_dict

def add_dic(dictionary,data):
    idx = len(dictionary)
    dictionary[idx] = data

    return dictionary

def fit_motion_matrix():

    train_messages, train_labels = load_com_dataset('./com_tr/motion_train2.tsv')
    if enable_debug:
        print("train_messages",train_messages)
        print("train_labels",train_labels)

    word_dict = load_word_dic(word_dic_path)
    motion_dic = load_word_dic(motion_dic_path)

    dimi = len(motion_dic)
    dimj = len(word_dict)
    word_dic_values = list(word_dict.values())
    mon_dic_values = list(motion_dic.values())
    revise_dic = False

    mo_obj_matrix = np.zeros((dimi,dimj))
    mo_matrix = np.zeros((dimi))
    obj_matrix = np.zeros((dimj))

    for i in range(len(train_messages)):
        message = get_words(train_messages[i])
        if enable_debug:
            print("message",len(message),message)

        for j in range(len(message)):
            if message[j] in mon_dic_values:
                keyi = mon_dic_values.index(message[j])
                if enable_debug:
                    print("message[j],train_labels[i]",message[j],train_labels[i])
                if train_labels[i] in word_dic_values:
                    keyj = word_dic_values.index(train_labels[i])
                else:
                    revise_dic = True
                    add_dic(word_dict,train_labels[i])
                    word_dic_values = list(word_dict.values())
                    keyj = word_dic_values.index(train_labels[i])

                mo_obj_matrix[keyi,keyj]+=1
                mo_matrix[keyi] +=1
                obj_matrix[keyj] +=1

    mo_obj_model = {}
    mo_obj_model['Fi_mo'] = mo_matrix.tolist()
    mo_obj_model['Fi_obj'] = obj_matrix.tolist()
    mo_obj_model['matrix'] = mo_obj_matrix.tolist()

#    print("Fi_y_pos,Fi_y_neg",Fi_y_pos.shape,Fi_y_neg.shape)
    write_json('./com_tr/mo_obj_model', mo_obj_model)

    if revise_dic:
        write_json(word_dic_path, word_dict)
        train_com_detect(train_path,test_path)

def predict_motion_objective(input_msg='',training=True):
    
    tf = open("./com_tr/mo_obj_model", "r")
    mo_obj_model = json.load(tf)

    mo_matrix = np.array(mo_obj_model['Fi_mo'],dtype=object)
    obj_matrix = np.array(mo_obj_model['Fi_obj'],dtype=object)
    mo_obj_matrix = np.array(mo_obj_model['matrix'],dtype=object)

    if training:
        test_messages, test_labels = load_com_dataset('./com_tr/motion_test3.tsv')
        if enable_debug:
            print("test_messages",test_messages)
            print("test_labels",test_labels)
    else:
        test_messages = input_msg

    word_dict = load_word_dic(word_dic_path)
    motion_dic = load_word_dic(motion_dic_path)

    dimi = len(motion_dic)
    dimj = len(word_dict)
    word_dic_values = list(word_dict.values())
    mon_dic_values = list(motion_dic.values())
    dic_idx = len(word_dict)
    is_motion = False
    is_obj = False
    value = []
    if training:
        for i in range(len(test_messages)):
            message = get_words(test_messages[i])
            if enable_debug:
                print("message",message)
            for j in range(len(message)):
                if message[j] in mon_dic_values:
                    keyi = mon_dic_values.index(message[j])
                    is_motion = True
                    if enable_debug:
                        print("message[j],keyi",message[j],keyi)
                    prob = np.zeros((dimj))
                    for k in range(len(message)):
                        if enable_debug:
                            print("k,message[k]",k,message[k])
                        if message[k] in word_dic_values:
                            keyj = word_dic_values.index(message[k])
                            if enable_debug:
                                print("mo_obj_matrix[keyi,keyj],obj_matrix[keyj],mo_matrix[keyi]",mo_obj_matrix[keyi,keyj],obj_matrix[keyj],mo_matrix[keyi])
#                            prob[keyj] = mo_obj_matrix[keyi,keyj] * obj_matrix[keyj] / mo_matrix[keyi]
                            prob[keyj] = mo_obj_matrix[keyi,keyj] * obj_matrix[keyj]
                        else:
                            prob[keyj] = 0.
                            

                    if enable_debug:
                        print("prob",prob)
                    idx = np.argmax(prob)

                    if prob[idx] > 0:
                        is_obj = True
                        if enable_debug:
                            print("word_dict[idx]",word_dict[idx])
                        value.append(word_dict[idx])
                    else:
                        value.append(" ")
                    break
                else:
                    if j == (len(message)-1):
                        value.append(" ")
                        break

    else:
        for i in range(len(test_messages)):
            message = get_words(test_messages[i])
            if enable_debug:
                print("message",message)
            motion_word = []
            obj_word = []
            for j in range(len(message)):
                if enable_debug:
                    print("message[j]",message[j])
                if message[j] in mon_dic_values:
                    keyi = mon_dic_values.index(message[j])
                    is_motion = True
                    motion_word.append(message[j])
                    if enable_debug:
                        print("message[j],keyi",message[j],keyi)
                    prob = np.zeros((dimj))
                    for k in range(len(message)):
                        if enable_debug:
                            print("k,message[k]",k,message[k])
                        if message[k] in word_dic_values:
                            keyj = word_dic_values.index(message[k])
                            if enable_debug:
                                print("mo_obj_matrix[keyi,keyj],obj_matrix[keyj],mo_matrix[keyi]",mo_obj_matrix[keyi,keyj],obj_matrix[keyj],mo_matrix[keyi])
#                            prob[keyj] = mo_obj_matrix[keyi,keyj] * obj_matrix[keyj] / mo_matrix[keyi]
                            prob[keyj] = mo_obj_matrix[keyi,keyj] * obj_matrix[keyj]

                    if enable_debug:
                        print("prob",prob)
                    idx = np.argmax(prob)

                    if prob[idx] > 0:
                        is_obj = True
                        obj_word.append(word_dict[idx])
                        if enable_debug:
                            print("word_dict[idx]",word_dict[idx])
                    else:
                        obj_word.append('')

    if training:
 #       if enable_debug:
 #           print("value",value)
        predict_analysis(value, test_labels, test_messages)
        return value
    else:
        if enable_debug:
            print("motion_word,obj_word",motion_word,obj_word)
        return is_motion, motion_word, is_obj, obj_word,input_msg

def predict_motion_objective_fd(input_msg,pred,fd_msg):
    
    tf = open("./com_tr/mo_obj_model", "r")
    mo_obj_model = json.load(tf)

    mo_matrix = np.array(mo_obj_model['Fi_mo'],dtype=object)
    obj_matrix = np.array(mo_obj_model['Fi_obj'],dtype=object)
    mo_obj_matrix = np.array(mo_obj_model['matrix'],dtype=object)

    fd_msg = fd_msg.strip()
    fd_msg = fd_msg.split(' ')
    test_messages = input_msg

    word_dict = load_word_dic(word_dic_path)
    motion_dic = load_word_dic(motion_dic_path)

    dimi = len(motion_dic)
    dimj = len(word_dict)
    word_dic_values = list(word_dict.values())
    mon_dic_values = list(motion_dic.values())
    revise_motiondic = False
    is_motion = False
    is_obj = False
    revise_dic = False
    if pred == 1:

        if enable_debug:
            print("input_msg",input_msg)
            print("fd_msg",fd_msg)
        message = get_words(test_messages[0])
        obj_word = []
        motion_word = []
        for m in range(len(fd_msg)):
            if fd_msg[m] in mon_dic_values:
                is_motion = True
                motion_word.append(fd_msg[m])
                keyi = mon_dic_values.index(fd_msg[m])
                prob = np.zeros((dimj))
                for k in range(len(message)):
                    if enable_debug:
                        print("k,message[k]",k,message[k])
                    if message[k] in word_dic_values:
                        keyj = word_dic_values.index(message[k])
                        prob[keyj] = mo_obj_matrix[keyi,keyj] * obj_matrix[keyj] / mo_matrix[keyi]
                        prob[keyj] = mo_obj_matrix[keyi,keyj] * obj_matrix[keyj]
                    else:
                        revise_dic = True
                        add_dic(word_dict,message[k])
                        word_dic_values = list(word_dict.values())

                idx = np.argmax(prob)
                if enable_debug:
                    print("prob, idx", prob,idx)

                if prob[idx] > 0:
                     is_obj = True
                     obj_word.append(word_dict[idx])
                     if enable_debug:
                         print("word_dict[idx]",word_dict[idx])
                else:
                    obj_word.append('')
#                return is_motion, fd_msg, is_obj, word_dict[idx]
            else:
                revise_motiondic = True
                add_dic(motion_dic,fd_msg[m])
                mon_dic_values = list(motion_dic.values())
                if fd_msg[m] not in word_dic_values:
                    revise_dic = True
                    add_dic(word_dict,fd_msg[m])
                    word_dic_values = list(word_dict.values())
    else:
        pass

    if revise_dic:
        write_json(word_dic_path, word_dict)
        train_com_detect(train_path,test_path)

    if revise_motiondic:
        write_json(motion_dic_path, motion_dic)
        fit_motion_matrix()

    return is_motion, motion_word, is_obj, obj_word,input_msg

def train_com_detect(train_path,test_path):
    train_messages, train_labels = load_command_dataset(train_path)
    test_messages, test_labels = load_command_dataset(test_path)

#    dictionary = create_dictionary(train_messages)

#    print('Size of dictionary: ', len(dictionary))
#    write_json(word_dic_path, dictionary)
    dictionary = load_word_dic(word_dic_path)
    train_matrix = transform_text(train_messages, dictionary)

#    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    np.savetxt(com_model_path, naive_bayes_model, fmt='%s')
    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

#    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)
#    np.savetxt('./com_tr/spam_naive_bayes_predictions', naive_bayes_predictions)
    if enable_debug:
        print("naive_bayes_predictions",naive_bayes_predictions)
        print("naive_bayes_predictions",np.where(naive_bayes_predictions==1))
        print("test_labels",np.where(test_labels==1))

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)
    error_analysis(naive_bayes_predictions, test_labels, test_messages)
    if enable_debug:
        print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))



def main():

    train_com_detect(train_path,test_path)

    fit_motion_matrix()
    predict_motion_objective()
#    write_json(motion_dic_path, motion_dic)


if __name__ == "__main__":
    main()

