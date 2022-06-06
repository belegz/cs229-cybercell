import numpy as np
import random as rn
import string

import os
from Predict import predict_command
from com_tr import *
import json

word_dic_path = './com_tr/command_dictionary'
save_train_path = './com_tr/motion_train2.tsv'
enable_debug = False
# os.environ['PYTHONHASHSEED'] = '0'

status = 0 #0:human input 1: communicate 2: exit
word_dic_path = './com_tr/command_dictionary'

def command_detect(message):

    word_dict = load_word_dic(word_dic_path)
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
    msg = message.replace("?", ".")
    msg = msg.split('.')
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
        new_msg = new_msg + 'motion is: ' + ' '.join(motion) + '\n'
        new_msg = new_msg + 'objective is: ' + ' '.join(obj) + '\n'
    elif is_motion:
        new_msg = '\nDo you mean:\n'
        new_msg = new_msg + 'motion is: ' + ' '.join(motion) + '\n'
        new_msg = new_msg + 'What is the object?' + '\n'
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

    update_data = False
    while True:
        #exact the key words
        if status == 0:
            message = input("Robot:\nUser, please input your command: \n")

            if message == 'exit':
                status = 2 # exit the program
                print("Thank you for use! Bye bye!")
                break

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
                    if update_data:
                        update_value = []
                        update_value.append(obj[0])
                        update_value.append(org_msg[0])
                        if enable_debug:
                            print("obj,org_msg",obj,obj[0],org_msg,update_value)
                        write_tsv(save_train_path, update_value)
                        fit_motion_matrix()
                        update_data = False

                else:
                    need_fb = 1
                    user_fb2 = input("Action word is correct?\n")
                    if user_fb2 == 'yes':
                        user_fb3 = input("Object word is correct?\n")
                        if user_fb3 == 'yes':
                            print("Thank you for the confirmation.We will excute the command.")
                            status = 0
                        else:
                            obj = []
                            input_obj = input("please input the correct object words.\n")
                            obj.append(input_obj)
                            is_obj = True
                            if enable_debug:
                                print("is_motion, motion",is_motion, motion)
                                print("input_obj,org_msg",input_obj,org_msg)
                            if input_obj in (org_msg[0].strip()).split(' '):
                                update_data = True
                    else:
                        new_motion = input("User,please confirm the action word.\n")
                        is_motion, motion, is_obj, obj,org_msg = predict_motion_objective_fd(org_msg,need_fb,new_motion)
            elif need_fb == 1:
                #did not find motion word and objective
                new_motion = input(msg_feedback)
                is_motion, motion, is_obj, obj,org_msg = predict_motion_objective_fd(org_msg,need_fb,new_motion)

                if not is_motion:
#                    print("Sorry, I dont have this feature. Please read the manual and input valid command.")
                    new_motion = input("Sorry, I dont have this feature. Please give the action word.\n")
                    is_motion, motion, is_obj, obj,org_msg = predict_motion_objective_fd(org_msg,need_fb,new_motion)
#                    status = 0
                else:
                    if enable_debug:
                        print("motion,obj,org_msg",motion,obj,org_msg)
                    if motion in (org_msg[0].strip()).split(' '):
                        update_data = True


            elif need_fb == 2:
                #found motion but did not find objective
                obj = []
                input_obj = input(msg_feedback)
                obj.append(input_obj)
                is_obj = True
                if enable_debug:
                    print("is_motion, motion",is_motion, motion)
                    print("input_obj,org_msg",input_obj,org_msg)
                if input_obj in (org_msg[0].strip()).split(' '):
                    update_data = True
            else:
                status = 0
        elif status == 2:
            print("Thank you for use! Bye bye!")
            break
 #           print(test)
