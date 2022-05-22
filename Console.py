import numpy as np
import random as rn
import string

import os
from Predict import predict_command

savepath = "command/Test/command.abstr"
savefolder = "command"
# os.environ['PYTHONHASHSEED'] = '0'

status = 0 #0:idle 1: listening 2: action 

def msg_pre_procss(message):
#    message = message + '\n'
    out = message.translate(str.maketrans('', '', string.punctuation))

    test_doc_str = {}
    test_doc_str['1'] = out

    out=out.split(' ')

    return out, test_doc_str

def msg_analysis(message):
#    message = message + '\n'
#    print("message",len(message),message)

    new_msg = 'Robot:\nDo you mean:'

    for i in range(len(message)):
#        print("len(message[i])",len(message[i]))
        list_msg = message[i]
        for j in range (len(list_msg)):
            new_msg = new_msg + list_msg[j] + ' '

    new_msg = new_msg + '?' + '\n'
    return new_msg

if __name__ == '__main__':

    while True:
        #exact the key words
        if status == 0:
            message = input("Robot:\nHuman, please input your command: \n")

            message,test_doc_str = msg_pre_procss(message)

#            print("test_doc_str",test_doc_str)
            np.savetxt(savepath,np.asarray([message]),fmt='%s')

            keywords = predict_command(savefolder)
#            print("keywords",keywords)
            new_msg = msg_analysis(keywords)

            msg_feedback = input(new_msg)
            status = 1
#        predict_command(test_doc_str)
#            break
        elif status == 1:
            #communication step
            print("msg_feedback",msg_feedback)