

import math 
import caffe
import numpy as np


DIR = "network2hidden_nostop/"
PICKLES_DIR = DIR + "pickles/"
IMAGE_DIR = DIR + "images/"


# This array of tuples represents the relevant layers in the network
LAYERS = [
    ('encoder1', 'encode1out'), #param name/ blob name
    ('encoder2', 'encode2out')
]

def loadLexicon(lexFile):
    """
    Load the lexicon and create a list of [word, input vec] pairs
    """
    lex = []
    with open(lexFile, 'r') as lf:
        for word in lf:
            lex.append([word.strip()])

    lexSize = len(lex)
    for i in range(lexSize):
        # for each word in the lexicon
        # generate an input vector
        # and append it to the list at the given index
        vec = np.zeros(lexSize)
        vec[i] = 1.0
        lex[i].append(vec)

    return lex



def loadNet(structure, model):
    """
    Load the network as a caffe.Net object given a model file and structure
    """
    return caffe.Net(structure, model, caffe.TEST)
    



def computeMaximalActivations(net, layerInd):
    print("Computing maximal activations")
    # for each hidden node find the input vector that maximally activates that node

    activations = []

    layerName = LAYERS[layerInd][0]
    weights = net.params[layerName][0].data
    for h in range(weights.shape[0]):
        inp = np.empty(weights.shape[1])
        denominator = math.sqrt((weights[h]**2).sum())
        for i in range(weights.shape[1]):
            numerator = weights[h][i]
            inp[i] = numerator / float(denominator)
        
        # if layerInd != 0, then need to trace back activations to first layer
        # http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq
        ind = layerInd
        while (ind > 0):
            # trace back activations
            # get the prev layer name
            layerName = LAYERS[ind-1][0]
            A = net.params[layerName][0].data
            B = inp
            inp = np.linalg.lstsq(A, B)[0]
            ind -= 1

        # here inp has the first layer input that produces 
        # the maxActivations at the requested layer
        # sort the list from high to low
        indices = np.argsort(inp)[::-1]
        inp = np.sort(inp)[::-1]        
        inp = (inp - inp.min()) / (inp.max() - inp.min()) 
        # zip together indices and vals
        act = [(i, v) for i,v in zip(indices, inp)]
        activations.append(act)
    
    return activations
    
    
def getActivationsWords(activations, lex):
    print ("Getting associated words for activations")
    # for each activation, make a list of words most relevant
    # should be sorted by now
    activationWords = []
    for act in activations:
        # each activation should be a list of [word index, contribution] normalized
        # from 0 to 1 in descending order
        words = []
        for e in act:
            # append the (index, word, contribution)
            words.append((e[0], lex[e[0]][0], e[1]))
        
        activationWords.append(words)

    return activationWords


def getPrevLayerNodes(net, layerName, hiddenNode, k):
    weights = net.params[layerName][0].data
    weights_ih = np.copy(weights[hiddenNode])
    weights_ih_asc = np.argsort(weights_ih)
    weights_ih_dsc = weights_ih_asc[::-1]
    weights_ih_dsc_k = weights_ih_dsc[:k]
    return weights_ih_dsc_k

