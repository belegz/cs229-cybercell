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

#    print("message",message)
    tf = open("./com_tr/command_dictionary", "r")
    word_dictionary = json.load(tf)
#    print("word_dictionary",word_dictionary)

    word_dict = {}

    for key, value in word_dictionary.items():
        word_dict[int(key)] = value

#    print("word_dict",word_dict)
    matrix = transform_text(message, word_dict)

#    print("matrix",matrix)
    tf = open("./com_tr/com_model", "r")
    com_model = json.load(tf)
    prediction = predict_from_naive_bayes_model(com_model, matrix,training=False)

    print("prediction",prediction)

#    new_msg = []
    for i in range(len(prediction)):
        if prediction[i] > 0:
            print(message[i])
#            new_msg.append(message[i]) 
            print("path",path)
            savepath = path + 'command' + str(i) + '.abstr'
            np.savetxt(savepath,np.asarray([message[i]]),fmt='%s')
            savepath2 = path + 'command' + str(i) + '.uncontr'
            np.savetxt(savepath2,np.asarray([temp_msg]),fmt='%s')

def msg_pre_procss(message):
#    message = message + '\n'

    msg = message.split('.')
#    print("msg",len(msg),msg)
    msg_list = []
    for i in range(len(msg)-1):
        new_msg = msg[i].translate(str.maketrans('', '', string.punctuation))
#        out=out.split(' ')
#        print("new_msg",new_msg)
        msg_list.append(new_msg)

    return msg_list

def msg_analysis(message):
#    message = message + '\n'
#    print("message",len(message),message)

    new_msg = 'Robot:\nDo you mean:'

    for i in range(len(message)):
#        print("len(message[i])",len(message[i]))
        list_msg = message[i]
        for j in range (len(list_msg)):
            new_msg = new_msg + list_msg[j] + ' '

    new_msg = new_msg + '? (yes/no)' + '\n'
    return new_msg

if __name__ == '__main__':

    while True:
        #exact the key words
        if status == 0:
            message = input("User, please input your command: \n")
            current_message = str(message)
            print(current_message)
            if message.lower()=="exit":
                print("Goodbye User, until next time.")
                break
            message = msg_pre_procss(message)
            new_msg = command_detect(message,dic_path,savepath)
#            print(test)

#            print("test_doc_str",test_doc_str)
#            np.savetxt(savepath,np.asarray([message]),fmt='%s')

            keywords = predict_command(savefolder)
            print("keywords",keywords)
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
            else:
                # print("Start communication")
                status = 2
                for i in range(MAX_CYCLE_NUM):
                    action_words = input("Please provide a few action key words in your command, separate with space if you have more than one: \n")
                    if action_words.lower()=="exit":
                        print("Goodbye User, until next time.")
                        exit()
                    print("Get Key words: "+action_words)
                    print("Message: "+current_message)
                    action_word_list = action_words.split(" ")
                    for a_word in action_word_list:
                        print("TODO")
                    new_msg = msg_analysis(keywords)
                    msg_feedback = input(new_msg)
                    # get good result
                    if(msg_feedback.lower()=="yes"):
                        status = 0
                        current_message = ""
                        break
                    if(i == MAX_CYCLE_NUM - 1):
                        current_message = ""
                        print("Sorry, We cannot analysis your command successfully. Please try another command. ")
                status = 0

