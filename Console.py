import numpy as np
import random as rn
import string

import os
from Predict import predict_command

savepath = "command/Test/command.abstr"
savefolder = "command"
# os.environ['PYTHONHASHSEED'] = '0'

status = 0 #0:idle 1: listening 2: action 
MAX_CYCLE_NUM = 10 # when it reach 10 cycles, the robot quits

def msg_pre_procss(message):
#    message = message + '\n'
    out = message.translate(str.maketrans('', '', string.punctuation))
    print(out)
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

    new_msg = new_msg + '? (yes/no)' + '\n'
    return new_msg

if __name__ == '__main__':

    print("Welcome to Cyber-Cell! This robot will help you to analysis your command. ")
    while True:
        #exact the key words
        if status == 0:
            message = input("Cyber-Cell:\nUser, please input your command: \n")

            message,test_doc_str = msg_pre_procss(message)

#            print("test_doc_str",test_doc_str)
            np.savetxt(savepath,np.asarray([message]),fmt='%s')
            status = 2
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
            if(msg_feedback.lower=="yes"):
               status = 0 
            else:
                status = 2
                action_words = input("Cyber-Cell:\nPlease provide a key word in your command: \n")
                for i in range(MAX_CYCLE_NUM):
                    # np.savetxt(savepath,np.asarray([message]),fmt='%s')
                    # TODO: implement what we want to do with action words
                    keywords = predict_command(savefolder)
                    new_msg = msg_analysis(keywords)
                    # get good result
                    if(msg_feedback.lower=="yes"):
                        status = 0
                        break
                print("Sorry, We cannot analysis your command successfully. Please try another command. ")
                status = 0

