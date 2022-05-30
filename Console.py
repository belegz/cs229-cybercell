import numpy as np
import random as rn
import string

import os
from Predict import predict_command
from com_tr import *
import json
from get_branch import *

#savepath_full = "command/Test/command.abstr"
savepath = "command/Test/"
savefolder = "command"
temp_msg = 'never mind; never mind'
current_message = ""
MAX_CYCLE_NUM = 10 # when it reach 10 cycles, the robot quits
# os.environ['PYTHONHASHSEED'] = '0'

status = 0 #0:human input 1: communicate 2: output
dic_path = './com_tr/myDictionary.pkl'
motion_dic = {0:'go',1:'bring',2:'put',3:'find',4:'move',5:'search',6:'take',7:'think',8:'drop',9:'remove',10:'switch',11:'grab',12:'want',13:'let',14:'look',15:'control',16:'release',17:'need',18:'listen',19:'be',20:'turn',21:'',}

def command_detect(message,dic_path,path):

    for i in range(len(message)):
        if message[i][0] == ' ':
            removed = message[i][1:]
            # print(removed)
            message[i] = removed
    # print("message",message)

    tf = open("./com_tr/command_dictionary", "r")
    word_dictionary = json.load(tf)
#    print("word_dictionary",word_dictionary)

    word_dict = {}

    for key, value in word_dictionary.items():
        word_dict[int(key)] = value

#    print("word_dict",word_dict)
    # print("in message ", message)
    matrix = transform_text(message, word_dict)

    # print("matrix",matrix)
    tf = open("./com_tr/com_model", "r")
    com_model = json.load(tf)
    prediction = predict_from_naive_bayes_model(com_model, matrix,training=False)

    print("prediction",prediction)

#    new_msg = []
    com_list = []
    for i in range(len(prediction)):
        if prediction[i] > 0:
            # print(message[i])
#            new_msg.append(message[i]) 
            # print("path",path)
            savepath = path + 'command' + str(i) + '.abstr'
            np.savetxt(savepath,np.asarray([message[i]]),fmt='%s')
            savepath2 = path + 'command' + str(i) + '.uncontr'
            np.savetxt(savepath2,np.asarray([temp_msg]),fmt='%s')
            com_list.append('command' + str(i))

    return com_list

def msg_pre_procss(message):
#    message = message + '\n'

    msg = message.split('.')
    # print("msg",len(msg),msg)
    msg_list = []
    for i in range(len(msg)):
        new_msg = msg[i].translate(str.maketrans('', '', string.punctuation))
#        out=out.split(' ')
#        print("new_msg",new_msg)
        msg_list.append(new_msg)

    return msg_list

def msg_analysis(message):
#    message = message + '\n'
#    print("message",len(message),message)

    new_msg = 'Do you mean:'

#     for i in range(len(message)):
# #        print("len(message[i])",len(message[i]))
#         list_msg = message[i]
#         for j in range (len(list_msg)):
#             new_msg = new_msg + list_msg[j] + ' '
#    print("len(message)",len(message))
    for k in range(len(message)):
        msg_item = message[k]
        for i in range(len(msg_item)):
#            print("len(msg_item)",len(msg_item))
            list_msg = msg_item[i]
            for j in range (len(list_msg)):
                new_msg = new_msg + list_msg[j] + ' '

    new_msg = new_msg + '? (yes/no)' + '\n'
    return new_msg

def msg_separate(old_message, action_word_list):
    msg_list = old_message.split(' ')
    # for msg in msg_list:
        # print(msg)
    indexes = []
    # msg_str_list = list()
    for i in range(len(msg_list)):
        if msg_list[i]in action_word_list:
        # index = msg_list.index(action_word)
        # if msg_list.count(action_word) > 1:
        # if action word not in msg_list
        # print("action_word", action_word, index)
            indexes.append(i)
    if not indexes:
        return
    # print("action indexes", indexes)
    for i in range(len(indexes)):
        if i+1 < len(indexes):
            # print("action index", indexes[i])
            from_idx = int(indexes[i])
            to_idx = int(indexes[i+1])
            # print(from_idx)
            # print(to_idx)
            # messages.append(msg_list[from_idx:to_idx])
            msg_list.insert(to_idx, ';')
            if msg_list[to_idx-1] == "and" or msg_list[to_idx-1] == "or":
                msg_list.pop(to_idx-1)
            # i = to_idx - 1
    msg_str = ' '.join(msg_list)
    print("Separated message", msg_str)
    msg_str_list = msg_str.split(';')
    return msg_str_list

if __name__ == '__main__':

    while True:
        #exact the key words
        if status == 0:
            message = input("User, please input your command: \n")
            current_message = str(message.lower())
            # print(current_message)
            if message.lower()=="exit":
                print("Goodbye User, until next time.")
                break
            message = message.lower()
            message = msg_pre_procss(message)
            # print(message)
            # com_list = command_detect(message,dic_path,savepath)
            com_list = command_detect(message,dic_path,savepath)
            print("com_list",com_list)
            # current_message = com_list
#            print(test)

#            print("test_doc_str",test_doc_str)
#            np.savetxt(savepath,np.asarray([message]),fmt='%s')
            if(len(com_list) == 0):
                print("This is not recognized as a command. Please try another. ")
                continue
            keywords = predict_command(savefolder,com_list)
            # print("keywords",keywords)
            new_msg = msg_analysis(keywords)

            msg_feedback = input(new_msg)
            status = 1
#        predict_command(test_doc_str)
#            break
        elif status == 1:
            #communication step
            # print("msg_feedback",msg_feedback.lower())
            if(msg_feedback.lower()=="yes"):
                current_message = ""
                status = 0
            elif(msg_feedback.lower()=="no"):
                # print("Start communication")
                status = 2
                action_words = input("Please provide a few action key words in your command, separate with space if you have more than one: \n")
                if action_words.lower()=="exit":
                    print("Goodbye User, until next time.")
                    exit()
                print("Get Key words: "+action_words)
                print("Message: "+current_message)
                action_word_list = action_words.split(" ")
                # for a_word in action_word_list:
                    # TODO: if is threshold problem, tune the com predict threshold
                    # print("action word", a_word)
                msg_str_list = msg_separate(current_message, action_word_list)
                if not msg_str_list:
                    print("This keyword is not in this sentence. ")
                    status = 0
                    current_message = ""
                    continue
                keywords_list = list()
                for i in range(len(msg_str_list)):
                    new_msg = msg_str_list[i]
                    new_msg = msg_pre_procss(new_msg)
                    com_list = command_detect(new_msg,dic_path,savepath)
                    keywords = predict_command(savefolder,com_list)
                    keywords_list.append(keywords)
                    # print(keywords)
                # print("keywords list",keywords_list)
                final_msg = ""
                all_success = True
                # pop_index = list()
                for i in range(len(keywords_list)):
                    new_msg = msg_analysis(keywords_list[i])
                    msg_feedback = input(new_msg)
                    # get good result
                    if(msg_feedback.lower()=="yes"):
                        # print(keywords_list[i][0])
                        final_list_add = keywords_list[i][0]
                        # print(map(' '.join,final_list_add))
                        final_msg_add = ' '.join(map(' '.join,final_list_add))
                        # print("add message",final_msg_add)
                        final_msg += final_msg_add
                        final_msg += ". "
                    else:
                        all_success = False
                if all_success:
                    status = 0
                    current_message = ""
                    continue
                else:
                    current_message = ""
                    print("The correctly understood command(s): ", final_msg)
                    print("Sorry, We cannot analysis your command successfully. Please try another command. ")
                    status = 0
            else:
                print("Sorry, I do not understand your input. Please enter yes or no. ")
                new_msg = msg_analysis(keywords)
                msg_feedback = input(new_msg)
