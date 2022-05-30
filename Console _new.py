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
enable_debug = True
# os.environ['PYTHONHASHSEED'] = '0'

status = 0 #0:human input 1: communicate 2: exit
word_dic_path = './com_tr/command_dictionary'

def command_detect(message):

#    print("message",message)
#    tf = open("./com_tr/command_dictionary", "r")
#    word_dictionary = json.load(tf)
#    print("word_dictionary",word_dictionary)

#    word_dict = {}
#    for key, value in word_dictionary.items():
#        word_dict[int(key)] = value

    word_dict = load_word_dic()
#    print("word_dict",word_dict)
    matrix = transform_text(message, word_dict)

#    print("matrix",matrix)
    tf = open("./com_tr/com_model", "r")
    com_model = json.load(tf)
    prediction = []
    prediction = predict_from_naive_bayes_model(com_model, matrix,training=False)

    if enable_debug:
        print("prediction",prediction)

#    new_msg = []
    com_list = []
    command_list = []
    for i in range(len(prediction)):
        if prediction[i] > 0:
            if enable_debug:
                print(message[i])
#            new_msg.append(message[i]) 
#            print("path",path)
#            savepath = path + 'command' + str(i) + '.abstr'
#            np.savetxt(savepath,np.asarray([message[i]]),fmt='%s')
#            savepath2 = path + 'command' + str(i) + '.uncontr'
#            np.savetxt(savepath2,np.asarray([temp_msg]),fmt='%s')
#            com_list.append('command' + str(i))
            command_list.append(message[i])

    return command_list

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

#    return msg_list
    return [s.lower() for s in msg_list]

def msg_post_procss(is_motion, motion, is_obj, obj):

    need_feedback = 0
    if is_motion and is_obj:

        new_msg = '\nDo you mean:\n'
        new_msg = new_msg + 'motion is: ' + motion + '\n'
        new_msg = new_msg + 'objective is: ' + obj + '\n'
    elif is_motion:
        new_msg = '\nDo you mean:\n'
        new_msg = new_msg + 'motion is: ' + motion + '\n'
        new_msg = new_msg + 'What is the objective?' + '\n'
        need_feedback = 2
    else:
        new_msg = '\nNot quite sure what you mean.\nWhat is the motion?\n'
        need_feedback = 1

    return need_feedback, new_msg

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
            message = input("Robot:\nUser, please input your command: \n")

            message = msg_pre_procss(message)
            com_list = command_detect(message)
            if enable_debug:
                print("com_list",com_list)

            while len(com_list) == 0 or len(com_list) > 1:
                if len(com_list) == 0:
                    message = input("Robot:\nThis is not a command. User, please input your command or call administrator: \n")
                else:
                    message = input("Robot:\nUser, please input one command at a time: \n")
                message = msg_pre_procss(message)
                com_list = command_detect(message)

#            print("test_doc_str",test_doc_str)
#            np.savetxt(savepath,np.asarray([message]),fmt='%s')

#            keywords = predict_command(savefolder,com_list)
#            print("keywords",keywords)
#            new_msg = msg_analysis(keywords)

#            msg_feedback = input(new_msg)
#            print("msg_feedback",msg_feedback)

            is_motion, motion, is_obj, obj,org_msg = predict_motion_objective(com_list,training=False)
            if enable_debug:
                print("is_motion, motion, is_obj, obj",is_motion, motion, is_obj, obj)

            status = 1
#        predict_command(test_doc_str)
#            break
        elif status == 1:
            #communication step
            need_fb, msg_feedback = msg_post_procss(is_motion, motion, is_obj, obj)
#            print("need_fb,msg_feedback",need_fb,msg_feedback)

            if need_fb == 0:
                user_fb = input(msg_feedback)

                if user_fb == 'yes':
                    print("Thank you for your confirmation. we will excute this command for you! Please input your next command or exit. ")
                    status = 0
                else:
                    need_fb = 1
                    new_motion = input("User,please confirm the motion word.")
                    is_motion, motion, is_obj, obj = predict_motion_objective_fd(org_msg,need_fb,new_motion)
            elif need_fb == 1:
                #did not find motion word and objective
                new_motion = input(msg_feedback)
                is_motion, motion, is_obj, obj = predict_motion_objective_fd(org_msg,need_fb,new_motion)

                if not is_motion:
                    print("Sorry, I dont have this feature.Please read the manual and input valid command.")
                    status = 0

            elif need_fb == 2:
                #found motion but did not find objective
                obj = input(msg_feedback)
                is_obj = True
                if enable_debug:
                    print("is_motion, motion",is_motion, motion)
            else:
                status = 0

 #           print(test)
