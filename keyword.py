from bs4 import BeautifulSoup


# Reading the data inside the xml
# file to a variable under the name
# data
with open('./Dataset/xml1/2337.xml', 'r') as f:
	data = f.read()

# Passing the stored data inside
# the beautifulsoup parser, storing
# the returned object
Bs_data = BeautifulSoup(data, "xml")

# Finding all instances of tag
# `sentence`
b_sentence = Bs_data.find_all('sentence')

sentence=[]
for elem in b_sentence:
    sentence.append(elem.text)

print("sentence",sentence)

# Finding all instances of tag
# `token`
#b_token = Bs_data.find_all('token')
#print("b_token",b_token)

# Using find() to extract attributes
# of the first instance of the tag, verb
#b_vb = Bs_data.find('token', {"pos":"VB"})

b_tokens = Bs_data.find_all('tokens')
#print("b_tokens",b_tokens)

b_semantics = Bs_data.find_all('semantics')

#print("b_semantics",b_semantics)

# Using find() to extract attributes
# of the first instance of the tag, verb
#b_sem = Bs_data.find('b_semantics', {"token":"id"})

#print("b_sem",b_sem)

def get_index(word,sentence):
    index = 0
    idx = []
    while index < len(sentence):
        if index != -1:
            index = sentence.find(word,index)
        if index == -1:
            break
        print("{} found at {}".format(word,index))
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

    print("dict",dict)

    return dict

def get_keyword(sentence,word,dict):
#    print("sentence",sentence)

    separate = sentence.split('"')
#    print("separate",separate)

    keyword = []
    non_keyword = []
    for i in range(len(separate)):
        if word in separate[i]:
            key = separate[i+1]
#            print("key",key)
            keyword.append(dict[key])
         
    print("keyword",keyword)

    return keyword

#print("get_index",get_index('token',str(b_token)))

#print("get_dict",get_dict(str(b_tokens),'token id=','lemma='))

keyword_dict = get_dict(str(b_tokens),'token id=','lemma=')

keyword = get_keyword(str(b_semantics),'token id=',keyword_dict)
