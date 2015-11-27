
import os, sys, math, cPickle, colorsys
import caffe
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from viz_common import *


def getHiddenFeatures(net, lex, layerName, blobName):
    """
    Get the hidden features for each word in the lexicon at the given hidden layer
    """
    print ("Computing the hidden features for each word")
    filters = np.empty((len(lex), net.params[layerName][0].data.shape[0]))
    for i in range(len(lex)):
        vec = lex[i][1]
        net.blobs['data'].data[...] = vec
        net.forward()
        hidden = net.blobs[blobName].data
        # save the hidden features with the word and input ved
        # NOTE: The copy is vital.
        filters[i] = np.copy(hidden)

    return filters




#############################
def getWordIndex(lex, word):
    """
    TODO: optimize this with binary search
    """
    for i in range(len(lex)):
        if (lex[i][0] == word):
            return i

    return 0


def findNearestNeighbors(lex):
    print ("Finding nearest neighbors")
    X = np.empty((len(lex), lex[0][2].size))
    for i in range(X.shape[0]):
        X[i] = lex[i][2]

    nbrs = NearestNeighbors(n_neighbors=25, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    while True:
        word = str(raw_input("Enter word for nn: "))
        ind = getWordIndex(lex, word)
        for n in range(len(indices[ind])):
            nword = lex[indices[ind][n]][0]
            ndistance = distances[ind][n]
            print ("neighbor ", nword, " at distance ", ndistance)
    
#########################################


def getPCADimenReduction(filters):
    print ("Computing PCA dimensionality reduction")
    pca = PCA(n_components=50)
    return pca.fit_transform(filters)


def getTSNEEmbedding(filters, perplex=30):
    print ("Computing t-SNE Embedding")
    for i in range(filters.shape[0]):
        for j in range(filters.shape[1]):
            if (filters[i][j] == 0):
                print (i,j, " = 0")
    model = TSNE(n_components=2, perplexity=perplex, random_state=0)
    return model.fit_transform(filters)


# from http://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
def getColors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def plotTSNE(embedding, lex, words, threshold=50, nodes=None, labels=True, colorBasedOn='hidden'):
    print ("Plotting t-SNE")
    fig, axes = plt.subplots()
    axes.clear()

    # color the first threshold words in the given list a different color
    # words should be a list of [index, word, contrib]

    listToIterate = None
    if (nodes != None):
        # iterate select nodes
        listToIterate = nodes
    else:
        # iterate all nodes
        listToIterate = range(len(words))

    colorMap = {}    
    colorChoices = None
    if (colorBasedOn == 'hidden'):
        colorChoices = getColors(len(listToIterate))
    else:
        colorChoices = getColors(min(threshold, len(words[0])))

    # count total number of colored points
    c = 0
    for i in range(len(listToIterate)):
        hidNode = listToIterate[i]
        # for required words in node
        for j in range(min(threshold, len(words[i]))):
            inNode = j
            # get the words[hidden node][input node][index]
            if (not words[hidNode][inNode][0] in colorMap):
                if (colorBasedOn == 'hidden'):
                    colorMap[words[hidNode][inNode][0]] = colorChoices[i] # % len(colorChoices)]
                else:
                    colorMap[words[hidNode][inNode][0]] = colorChoices[j] # % len(colorChoices)]
                c += 1
    

    # only add points that are colored
    print ("num points = ", c)
    xs = np.empty((c,))
    ys = np.empty((c,))

    colors = []
    p = 0
    for k,v in colorMap.iteritems():
        xs[p] = embedding[k][0]
        ys[p] = embedding[k][1]
        colors.append(v)

        # label point?
        if (labels):
             axes.annotate(lex[k][0], xy=(xs[p], ys[p]), xytext=(xs[p], ys[p])) 
          
        p += 1

    # embedding is Nx2 where N is the number of elements to place in the space
#    xs = np.empty((embedding.shape[0],))
#    ys = np.empty((embedding.shape[0],))
    
    # create a color list that maps an index to a color
#    colors = []
    # get the x,y points from the embedding
#    for i in range(embedding.shape[0]):
#        xs[i] = embedding[i][0]
#        ys[i] = embedding[i][1]
#        if (i in colorMap):
#            colors.append(colorMap[i])
#        else:
#            colors.append((1.,1.,1.)) #"white"

    axes.scatter(xs, ys, c=colors, marker='o')    

    plt.show()
    plt.draw()



if __name__ == "__main__":
    if (len(sys.argv) < 4):
        print ("runs `python tsne_viz.py structure.prototxt model.caffemodel corpus.lex [dopca]`")
        sys.exit(1)

    structure = sys.argv[1]
    model = sys.argv[2]
    lexFile = sys.argv[3]
    doPCA = True if (len(sys.argv) > 4 and sys.argv[4] == "dopca") else False
    perplexity = int(sys.argv[5]) if (len(sys.argv) > 5) else 30

    # get word, vec, pairs
    lex = loadLexicon(lexFile)
    
    # init the network
    caffe.set_mode_cpu()
    net = loadNet(structure, model)

    if (not os.path.isdir(PICKLES_DIR)):
        os.makedirs(PICKLES_DIR)
        
    filters = []
    embeddings = []
    maxActivations = []
    words = []

    for l in range(len(LAYERS)):
        # load data from file if pre computed
        SPEC_FILTERS = "_iter" + model[model.rfind("_")+1:model.rfind(".caffemodel")] + ("_pca50" if (doPCA) else "") + "_l" + str(l+1)
        SPEC_TSNE = "_iter" + model[model.rfind("_")+1:model.rfind(".caffemodel")] + "_perplexity" + str(perplexity) + "_l" + str(l+1)
        SPEC_MAX_ACTS = "_iter" + model[model.rfind("_")+1:model.rfind(".caffemodel")] + "_l" + str(l+1)
        SPEC_WORDS = "_iter" + model[model.rfind("_")+1:model.rfind(".caffemodel")] + "_l" + str(l+1)
        
        FILTERS_FILE = "filters"+SPEC_FILTERS+".pickle"
        TSNE_EMBEDDINGS_FILE = "embedding"+SPEC_TSNE+".pickle"
        MAX_ACTIVATIONS_FILE = "max_activations"+SPEC_MAX_ACTS+".pickle"
        WORDS_FILE = "words"+SPEC_WORDS+".pickle"

        layerName = LAYERS[l][0]
        blobName = LAYERS[l][1]

        if (os.path.isfile(PICKLES_DIR+FILTERS_FILE)):
            print ("Loading saved filters from "+ str(PICKLES_DIR+FILTERS_FILE))
            filtersIn = open(PICKLES_DIR+FILTERS_FILE, 'rb')
            unpickler = cPickle.Unpickler(filtersIn)
            layerFilters = unpickler.load()
            filters.append(layerFilters)
            filtersIn.close()
        else:
            filtersOut = open(PICKLES_DIR+FILTERS_FILE, 'wb')
            pickler = cPickle.Pickler(filtersOut)
            # load the features for each layer
            layerFilters = getHiddenFeatures(net, lex, layerName, blobName)
            if (doPCA):
                layerFilters = getPCADimenReduction(layerFilters)
            filters.append(layerFilters)
            print ("Saving filters to " + str(PICKLES_DIR+FILTERS_FILE))
            pickler.dump(layerFilters)
            filtersOut.close()


        if (os.path.isfile(PICKLES_DIR+TSNE_EMBEDDINGS_FILE)):
            print ("Loading saved embedding from " + str(PICKLES_DIR+TSNE_EMBEDDINGS_FILE))
            embeddingsIn = open(PICKLES_DIR+TSNE_EMBEDDINGS_FILE, 'rb')
            unpickler = cPickle.Unpickler(embeddingsIn)
            layerEmbedding = unpickler.load()
            embeddings.append(layerEmbedding)
            embeddingsIn.close()
        else:
            embeddingsOut = open(PICKLES_DIR+TSNE_EMBEDDINGS_FILE, 'wb')
            pickler = cPickle.Pickler(embeddingsOut)
            layerEmbedding = getTSNEEmbedding(filters[l], perplexity)
            embeddings.append(layerEmbedding)
            print ("Saving embedding to " + str(PICKLES_DIR+TSNE_EMBEDDINGS_FILE))
            pickler.dump(layerEmbedding)
            embeddingsOut.close()


        if (os.path.isfile(PICKLES_DIR+MAX_ACTIVATIONS_FILE)):
            print ("Loading saved activations from " + str(PICKLES_DIR+MAX_ACTIVATIONS_FILE))
            maxActivationsIn = open(PICKLES_DIR+MAX_ACTIVATIONS_FILE, 'rb')
            unpickler = cPickle.Unpickler(maxActivationsIn)
            layerMaxActivations = unpickler.load()
            maxActivations.append(layerMaxActivations)
            maxActivationsIn.close()
        else:
            maxActivationsOut = open(PICKLES_DIR+MAX_ACTIVATIONS_FILE, 'wb')
            pickler = cPickle.Pickler(maxActivationsOut)
            layerMaxActivations = computeMaximalActivations(net, l)
            maxActivations.append(layerMaxActivations)
            print ("Saving max activations to " + str(PICKLES_DIR+MAX_ACTIVATIONS_FILE))
            pickler.dump(layerMaxActivations)
            maxActivationsOut.close()

    
        # words is a list of [index, word, contrib]
        if (os.path.isfile(PICKLES_DIR+WORDS_FILE)):
            print ("Loading saved words from " + str(PICKLES_DIR+WORDS_FILE))
            wordsIn = open(PICKLES_DIR+WORDS_FILE, 'rb')
            unpickler = cPickle.Unpickler(wordsIn)
            layerWords = unpickler.load()
            words.append(layerWords)
            wordsIn.close()
        else:
            wordsOut = open(PICKLES_DIR+WORDS_FILE, 'wb')
            pickler = cPickle.Pickler(wordsOut)
            layerWords = getActivationsWords(maxActivations[l], lex)
            words.append(layerWords)
            print ("Saving activation words to " + str(PICKLES_DIR+WORDS_FILE))
            pickler.dump(layerWords)
            wordsOut.close()


    findOptimalNode(net, maxActivations)

    # let the user view the visualization
    while True:
        thresh = int(raw_input("Threshold: "))
        optimal = True if (raw_input("Compute optimal 1st layer nodes from 2nd layer node? y/n ") == "y") else False
        
        n = None
        layer = None
        if (optimal):
            n = int(raw_input("Choose 2nd layer node (0 - " + str(len(words[1])) + "): "))
            layer = 0
            nodes = list(getPrevLayerNodes(net, LAYERS[1][0], n, 5))
        else:
            layer = int(raw_input("Layer (0 - " + str(len(LAYERS)-1) + "): "))
            nodes = raw_input("Nodes (0 - " + str(len(words[layer])) + "), (e.g. 0, 4, 15, 30): ")
            if (nodes == ''):
                nodes = None
            else:
                nodes = nodes.split(",")
                nodes = [int(x.strip()) for x in nodes]

        print ("nodes = ", nodes)
        labels = True if (raw_input("Show labels y/n: ") == "y") else False

        plotTSNE(embeddings[layer], lex, words[layer], thresh, nodes, labels)

        if (optimal):
            show2ndPlot = True if (raw_input("Show 2nd layer plot of node " + str(n) + "? y/n ") == "y") else False
            layer = 1
            if (show2ndPlot):
                plotTSNE(embeddings[layer], lex, words[layer], thresh, [n], labels, 'input')

    #findNearestNeighbors(lex)
    
    
