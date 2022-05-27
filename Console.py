import numpy as np
import random as rn
import string

import os
from Predict import predict_command
from com_tr import *
import json

#savepath_full = "command/Test/command.abstr"
savepath = "command/Test/"
savefolder = "command"
temp_msg = 'never mind; never mind'
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
    com_list = []
    for i in range(len(prediction)):
        if prediction[i] > 0:
            print(message[i])
#            new_msg.append(message[i]) 
#            print("path",path)
            savepath = path + 'command' + str(i) + '.abstr'
            np.savetxt(savepath,np.asarray([message[i]]),fmt='%s')
            savepath2 = path + 'command' + str(i) + '.uncontr'
            np.savetxt(savepath2,np.asarray([temp_msg]),fmt='%s')
            com_list.append('command' + str(i))

    return com_list

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
#    print("len(message)",len(message))
    for k in range(len(message)):
        msg_item = message[k]
        for i in range(len(msg_item)):
#            print("len(msg_item)",len(msg_item))
            list_msg = msg_item[i]
            for j in range (len(list_msg)):
                new_msg = new_msg + list_msg[j] + ' '

    new_msg = new_msg + '?' + '\n'
    return new_msg

if __name__ == '__main__':

    while True:

        #exact the key words
        if status == 0:
            message = input("Robot:\nHuman, please input your command: \n")

            message = msg_pre_procss(message)
            com_list = command_detect(message,dic_path,savepath)
            print("com_list",com_list)

#            print("test_doc_str",test_doc_str)
#            np.savetxt(savepath,np.asarray([message]),fmt='%s')

            keywords = predict_command(savefolder,com_list)
            print("keywords",keywords)
            new_msg = msg_analysis(keywords)

            msg_feedback = input(new_msg)
            status = 1
#        predict_command(test_doc_str)
#            break
        elif status == 1:
            #communication step
            print("msg_feedback",msg_feedback)

