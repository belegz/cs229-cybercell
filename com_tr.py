import collections
import csv
import numpy as np
import json
import string

def load_spam_dataset(tsv_path):
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

    msg = message.split(' ')
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
    occ_count = 2
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
    # *** START CODE HERE ***
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

    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    Fi_y = model[0]
    Fi_y_pos = model[1]
    Fi_y_neg = model[2]

    prob = np.zeros((len(dictionary)))
    top5word = []
    for i in range(len(dictionary)):
        prob[i] = np.log(Fi_y_pos[i]/Fi_y_neg[i])
#        print("i,prob[i]",i,prob[i])

    idx = np.argsort(-prob)
#    print("idx",idx)

    for i in range(5):
#        print("dictionary[idx[i]]",dictionary[idx[i]])
        top5word.append(dictionary[idx[i]])

    return top5word

    # *** END CODE HERE ***


def main():

    train_messages, train_labels = load_spam_dataset('./com_tr/command_train.tsv')

#    print("train_messages",len(train_messages), train_messages)
#    print("train_labels",len(train_labels), train_labels)

#    val_messages, val_labels = load_spam_dataset('./com_tr/command_val.tsv')
    test_messages, test_labels = load_spam_dataset('./com_tr/command_test.tsv')
#    print("train_labels", train_labels,train_labels.shape[0],np.where(train_labels==1)[0])

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    write_json('./com_tr/command_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

#    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])
    np.savetxt('./com_tr/spam_sample_train_matrix', train_matrix[:100,:])

#    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    np.savetxt('./com_tr/command_model', naive_bayes_model, fmt='%s')
    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

#    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)
#    np.savetxt('./com_tr/spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

#    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

#    write_json('./com_tr/spam_top_indicative_words', top_5_words)

if __name__ == "__main__":
    main()

