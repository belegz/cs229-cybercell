import collections
import csv
import numpy as np
import json
import string

motion_dic = {0:'go',1:'bring',2:'put',3:'find',4:'move',5:'search',6:'take',7:'think',8:'drop',9:'remove',10:'switch',11:'grab',12:'want',13:'let',14:'look',15:'control',16:'release',17:'need',18:'listen',19:'be',20:'turn',21:'clean',22:'feed',23:'lift'}
word_dic_path = './com_tr/command_dictionary'
enable_debug = False

def load_command_dataset(tsv_path):
    """Load the spam dataset from a TSV file

    Args:
         csv_path: Path to TSV file containing dataset.

    Returns:
        messages: A list of string values containing the text of each message.
        labels: The binary labels (0 or 1) for each message. A 1 indicates spam.
    """

    messages = []
    labels = []

    with open(tsv_path, 'r', newline='', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')

        for label, message in reader:
#            print("label, message",label, message)
            messages.append(message)
            labels.append(1 if label == 'com' else 0)

    return messages, np.array(labels)

def load_com_dataset(tsv_path):
    """Load the spam dataset from a TSV file

    Args:
         csv_path: Path to TSV file containing dataset.

    Returns:
        messages: A list of string values containing the text of each message.
        labels: The binary labels (0 or 1) for each message. A 1 indicates spam.
    """

    messages = []
    labels = []

    with open(tsv_path, 'r', newline='', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')

        for label, message in reader:
#            print("label, message",label, message)
            messages.append(message)
            labels.append(label)

    return messages, labels


def write_json(filename, value):
    """Write the provided value as JSON to the given filename"""
    with open(filename, 'w') as f:
        json.dump(value, f)

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    msg = message.strip()
    msg = msg.split(' ')
 #   print("msg",msg)
#    msg = msg.replace('\r\nham\t',' ')
#    msg = message.split(' ')
#    msg = message.translate(str.maketrans('', '', string.punctuation))
#    msg = [s.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~')) for s in msg]
#    print("msg",len(msg),msg)
    
    return [s.lower() for s in msg]

    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
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
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***

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

#    print("Fi_y_pos,Fi_y_neg",Fi_y_pos.shape,Fi_y_neg.shape)
    write_json('./com_tr/com_model', com_model)

    return model

    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix,training=True):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """

    if training:
        Fi_y = model[0]
        Fi_y_pos = model[1]
        Fi_y_neg = model[2]
    else:
#        print("model",model)
        Fi_y = model['Fi_y']
        Fi_y_pos = np.array(model['Fi_y_pos'])
        Fi_y_neg = np.array(model['Fi_y_neg'])

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

    return prediction


def load_word_dic():
    tf = open(word_dic_path, "r")
    word_dictionary = json.load(tf)
#    print("word_dictionary",word_dictionary)

    word_dict = {}
    for key, value in word_dictionary.items():
        word_dict[int(key)] = value

    return word_dict

def fit_motion_matrix():

    train_messages, train_labels = load_com_dataset('./com_tr/motion_train2.tsv')
    if enable_debug:
        print("train_messages",train_messages)
        print("train_labels",train_labels)

    word_dict = load_word_dic()

    dimi = len(motion_dic)
    dimj = len(word_dict)
    word_dic_values = list(word_dict.values())
    mon_dic_values = list(motion_dic.values())

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

def predict_motion_objective(input_msg='',training=True):
    
    tf = open("./com_tr/mo_obj_model", "r")
    mo_obj_model = json.load(tf)

    mo_matrix = np.array(mo_obj_model['Fi_mo'])
    obj_matrix = np.array(mo_obj_model['Fi_obj'])
    mo_obj_matrix = np.array(mo_obj_model['matrix'])

    if training:
        test_messages, test_labels = load_com_dataset('./com_tr/motion_test2.tsv')
        if enable_debug:
            print("test_messages",test_messages)
            print("test_labels",test_labels)
    else:
        test_messages = input_msg

    word_dict = load_word_dic()

    dimi = len(motion_dic)
    dimj = len(word_dict)
    word_dic_values = list(word_dict.values())
    mon_dic_values = list(motion_dic.values())

    is_motion = False
    is_obj = False

    if training:
        value = []
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
                            prob[keyj] = mo_obj_matrix[keyi,keyj] * obj_matrix[keyj] / mo_matrix[keyi]
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
        for i in range(len(test_messages)):
            message = get_words(test_messages[i])
            if enable_debug:
                print("message",message)
            for j in range(len(message)):
                if enable_debug:
                    print("message[j]",message[j])
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
                            prob[keyj] = mo_obj_matrix[keyi,keyj] * obj_matrix[keyj] / mo_matrix[keyi]
                        else:
                            prob[keyj] = 0.

                    if enable_debug:
                        print("prob",prob)
                    idx = np.argmax(prob)

                    if prob[idx] > 0:
                        is_obj = True
                        if enable_debug:
                            print("word_dict[idx]",word_dict[idx])

                    return is_motion, motion_dic[keyi], is_obj, word_dict[idx],input_msg

    if training:
        print("value",value)
        return value
    else:
        return is_motion, '', is_obj, '',input_msg

def predict_motion_objective_fd(input_msg,pred,fd_msg):
    
    tf = open("./com_tr/mo_obj_model", "r")
    mo_obj_model = json.load(tf)

    mo_matrix = np.array(mo_obj_model['Fi_mo'])
    obj_matrix = np.array(mo_obj_model['Fi_obj'])
    mo_obj_matrix = np.array(mo_obj_model['matrix'])

    test_messages = input_msg

    word_dict = load_word_dic()

    dimi = len(motion_dic)
    dimj = len(word_dict)
    word_dic_values = list(word_dict.values())
    mon_dic_values = list(motion_dic.values())
    
    is_motion = False
    is_obj = False
    if pred == 1:

        if enable_debug:
            print("input_msg",input_msg)
            print("fd_msg",fd_msg)
        message = get_words(test_messages[0])
        if fd_msg in mon_dic_values:
            keyi = mon_dic_values.index(fd_msg)
            prob = np.zeros((dimj))
            for k in range(len(message)):
                if enable_debug:
                    print("k,message[k]",k,message[k])
                if message[k] in word_dic_values:
                    keyj = word_dic_values.index(message[k])
                    prob[keyj] = mo_obj_matrix[keyi,keyj] * obj_matrix[keyj] / mo_matrix[keyi]
                else:
                    prob[keyj] = 0.

            idx = np.argmax(prob)
            if enable_debug:
                print("prob, idx", prob,idx)

            if prob[idx] > 0:
                 is_obj = True
                 if enable_debug:
                     print("word_dict[idx]",word_dict[idx])

            is_motion = True
            return is_motion, fd_msg, is_obj, word_dict[idx]
    else:
        pass


    return is_motion, '', is_obj, ''


def main():

    train_messages, train_labels = load_command_dataset('./com_tr/command_train4.tsv')

#    print("train_messages",len(train_messages), train_messages)
#    print("train_labels",len(train_labels), train_labels)

#    val_messages, val_labels = load_spam_dataset('./com_tr/command_val.tsv')
    test_messages, test_labels = load_command_dataset('./com_tr/command_test.tsv')
#    print("train_labels", train_labels,train_labels.shape[0],np.where(train_labels==1)[0])

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    write_json('./com_tr/command_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

#    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    np.savetxt('./com_tr/command_model', naive_bayes_model, fmt='%s')
    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

#    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)
#    np.savetxt('./com_tr/spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    fit_motion_matrix()
    predict_motion_objective()

if __name__ == "__main__":
    main()

