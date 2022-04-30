from bs4 import BeautifulSoup


# Reading the data inside the xml
# file to a variable under the name
# data
with open('./data/testXML/3272.xml', 'r') as f:
	data = f.read()

# Passing the stored data inside
# the beautifulsoup parser, storing
# the returned object
Bs_data = BeautifulSoup(data, "xml")

# Finding all instances of tag
# `sentence`
b_sentence = Bs_data.find_all('sentence')

print(b_sentence)

# Finding all instances of tag
# `token`
b_token = Bs_data.find_all('token')

print(b_token)

# Using find() to extract attributes
# of the first instance of the tag, verb
b_vb = Bs_data.find('token', {"pos":"VB"})

print(b_vb)
