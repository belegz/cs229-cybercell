from bs4 import BeautifulSoup
import numpy as np
import os
import numpy as np
# from sklearn.model_selection import train_test_split
from get_branch import *
import splitfolders

filepath = './Dataset/'
savepath = './Datasetnew/'
#filename = '2337'

def get_index(word,sentence):
    index = 0
    idx = []
    while index < len(sentence):
        if index != -1:
            index = sentence.find(word,index)
        if index == -1:
            break
#        print("{} found at {}".format(word,index))
        idx.append(index)
        index +=len(word)
    return idx

def get_dict(sentence,word1,word2):
#    print("sentence",sentence)

    separate = sentence.split('"')
#    print("separate",separate)

    dict = {}
    for i in range(len(separate)):
        if word1 in separate[i]:
            key = separate[i+1]
#            print("key",key)
        if word2 in separate[i]:
            value = separate[i+1]
#            print("value",value)
            dict[key] = value

#    print("dict",dict)

    return dict

def get_keyword(sentence,word,dict):
#    print("sentence",sentence)

    separate = sentence.split('"')
#    print("separate",separate)

    keyword = []
    str = ''
    for i in range(len(separate)):
        if word in separate[i]:
            key = separate[i+1]
#            print("key",key)
            keyword.append(dict[key])
#            str = str + dict[key] + ' '
         
#    print("keyword",keyword)
#    print("str",str)

    str = '; '.join(keyword)
#    print("str",str)

    str_array = np.array([str])
#    print("str_array",str_array)
    return keyword, str_array

def get_nonkeyword(sentence,keyword):
    non_keyword = []

    separate = sentence[0].split(' ')
#    print("separate",separate)

    for i in range(len(separate)):
        if separate[i] not in keyword:
            non_keyword.append(separate[i])
#    print("non_keyword",non_keyword)

    if len(non_keyword) > 0:
        str = '; '.join(non_keyword)
#        print("str",str)

        str_array = np.array([str])
#        print("str_array",str_array)
    else:
        str_array = np.array(['none'])
#        print("str_array",str_array)

    return non_keyword, str_array

def get_augkeyword(Bs_data):
    aug_keyword = []

    separate = sentence[0].split(' ')
#    print("separate",separate)
    pos_list = get_poss(Bs_data)
    det_list = get_det(Bs_data)
    colon_list = []
    # if len(pos_list) > 0 & len(det_list) > 0:

    # str = ''
    
        # for i in range(len(separate)):

        # if word in separate[i]:
            # key = separate[i+1]
#            print("key",key)
            # keyword.append(dict[key])
#            str = str + dict[key] + ' '
         
#    print("keyword",keyword)
#    print("str",str)

    # str = '; '.join(keyword)
#    print("str",str)

    # str_array = np.array([str])
#    print("str_array",str_array)
    # return keyword, str_array


def save_file(Bs_data, b_tokens_con,b_semantics_con,sentence,word1,word2,savepath):
#    keyword_dict = get_dict(str(b_tokens),'token id=','lemma=')
#    keyword,str = get_keyword(str(b_semantics),'token id=',keyword_dict)
#    non_keyword,str_nonkey = get_nonkeyword(sentence,keyword)

    keyword_dict = get_dict(str(b_tokens_con),word1,word2)
#    print("keyword_dict",keyword_dict)
    keyword,str_key = get_keyword(str(b_semantics_con),word1,keyword_dict)
#    print("keyword",keyword)
    non_keyword,str_nonkey = get_nonkeyword(sentence,keyword)
#    print("non_keyword",non_keyword)

    get_augkeyword(Bs_data)
    # aug_word,str_aug = get_augkeyword(Bs_data)
    # print("aug_word",aug_word)
    # print("aug_string", str_aug)

    #Add in files:
    savepath_sen = savepath + '.abstr'
    np.savetxt(savepath_sen,np.asarray(sentence),fmt='%s')

#    savepath_key = savepath + '.contr'
    savepath_key = savepath + '.uncontr'
    np.savetxt(savepath_key,str_key,fmt='%s')

    savepath_nonkey = savepath + '.contr'

    savepath_key = savepath + '.aug'
    # np.savetxt(savepath_key,str_aug,fmt='%s')

#    if len(non_keyword) > 0:
#    np.savetxt(savepath_nonkey,str_nonkey,fmt='%s')
 #   else:
 #       open(savepath_nonkey,"w")

if __name__ == '__main__':
    files = os.listdir(filepath)
    i=1
#    for file in files:
    for file in files:
        filename = filepath+'/'+file
        print("filename",filename)
        with open(filename, 'r') as f:
            data = f.read()

        Bs_data = BeautifulSoup(data, "xml")

        # Finding sentence
        b_sentence = Bs_data.find_all('sentence')
        print(b_sentence)

        sentence=[]
#        sentence.append("command")
        for elem in b_sentence:
            sentence.append(elem.text)

#        print("sentence",sentence)
        b_tokens = Bs_data.find_all('tokens')
#        print("b_tokens",b_tokens)

        b_semantics = Bs_data.find_all('semantics')
#        print("b_tokens",b_tokens)

        savefilepath = savepath+file.split('.')[0]
        print("savefilepath",savefilepath)

#        save_file(b_tokens,b_semantics,sentence,'token id=','lemma=',savefilepath)
        save_file(Bs_data,b_tokens,b_semantics,sentence,'token id=','surface=',savefilepath)
        i=i+1

        #for debug
#        if i ==3:
#            break
    # splitfolders.ratio('Datasetnew', output="output", seed=1337, ratio=(.8, 0.1,0.1)) 