from bs4 import BeautifulSoup
import numpy as np
import os
# from textblob import TextBlob
import nltk
from lxml import etree

filepath = './Dataset/xml1/'
savepath = './Datasetnew/'

#poss
def get_poss(Bs_data):
    posses = list()
    posses_id = list()
    b_posses = list()
    b_posses += Bs_data.find_all(attrs={"type":'poss'})

    # print(b_preps)
    for poss_element in b_posses:
        poss_from_id = int(poss_element['from'])
        poss_to_id = int(poss_element['to'])
        posses_id.append((poss_from_id,poss_to_id))
        poss_from = Bs_data.find_all(attrs={"id":poss_from_id})
        poss_to = Bs_data.find_all(attrs={"id":poss_to_id})
        poss_surface = (poss_to[0]['surface'],poss_from[0]['surface'])
        posses.append(poss_surface)
    print(posses)
    return posses

#det is one word, the+sth, a+sth, need to be put together
def get_det(Bs_data):
    dets = list()
    dets_id = list()
    b_dets = list()
    b_dets += Bs_data.find_all(attrs={"type":'det'})

    # print(b_preps)
    for det_element in b_dets:
        det_from_id = int(det_element['from'])
        det_to_id = int(det_element['to'])
        dets_id.append((det_from_id,det_to_id))
        det_from = Bs_data.find_all(attrs={"id":det_from_id})
        det_to = Bs_data.find_all(attrs={"id":det_to_id})
        det_surface = (det_to[0]['surface'],det_from[0]['surface'])
        dets.append(det_surface)
    print(dets)
    return dets

#if prep then one work(go to, take from), use together as keyword
def get_prep(Bs_data):
    preps = list()
    preps_id = list()
    b_preps = list()
    b_preps += Bs_data.find_all(attrs={"type":'prep'})

    # print(b_preps)
    for prep_element in b_preps:
        prep_from_id = int(prep_element['from'])
        prep_to_id = int(prep_element['to'])
        preps_id.append((prep_from_id,prep_to_id))
        prep_from = Bs_data.find_all(attrs={"id":prep_from_id})
        prep_to = Bs_data.find_all(attrs={"id":prep_to_id})
        prep_surface = (prep_from[0]['surface'],prep_to[0]['surface'])
        preps.append(prep_surface)
    # print(preps)
    return preps

def get_nouns(Bs_data):
    nouns = list()
    nouns_id = list()
    b_nouns = list()
    b_nouns += Bs_data.find_all(attrs={"pos":'NN'})
    b_nouns += Bs_data.find_all(attrs={"pos":'NNS'})
    b_nouns += Bs_data.find_all(attrs={"pos":'NNP'})
    b_nouns += Bs_data.find_all(attrs={"pos":'NNPS'})

    # print(b_nouns)
    for noun_element in b_nouns:
        noun_id = int(noun_element['id'])
        noun_surface = noun_element['surface']
        nouns_id.append(noun_id)
        nouns.append(noun_surface)
    # print(nouns_id)
    print(nouns)
    return nouns


if __name__ == '__main__':
    files = os.listdir(filepath)
    for file in files:
        filename = filepath+'/'+file
        print("filename",filename)
        with open(filename, 'r') as f:
            data = f.read()

        Bs_data = BeautifulSoup(data, "xml")

        # nouns = get_nouns(Bs_data)
        # preps = get_prep(Bs_data)
        # posses = get_poss(Bs_data)
        # det = get_det(Bs_data)


