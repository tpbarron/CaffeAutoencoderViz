

import os, sys, time, cPickle
import matplotlib.pyplot as plt

from PIL import Image # Pillow image
from scipy.misc import imread
from wordcloud import WordCloud

from viz_common import *



# words is an array of [index, word, val] elements
def generateText(words, k=None):
    # gen text for first k words, all words if k == None
    text = ""
    if (k == None):
        k = len(words)
        
    for i in range(k):
        index, word, val = words[i]
        num = int(round(abs(val) * 100))
        text += (word + ' ') * num
    
    return text
        

        
wordColorMap = {}

def pos_neg_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    #print ("word = ", word, wordColorMap[word])
    if (wordColorMap[word] == 1):
        return "rgb(0, 255, 0)"
    elif (wordColorMap[word] == -1):
        return "rgb(0, 0, 255)"

def gray_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    #print ("word = ", word, wordColorMap[word])
    gray = wordColorMap[word]
    return "hsl(0, 0%, " + str(gray) + "%)"
     

# words is an array of [index, word, contrib]
def generateWordCloud(words, maskImg=None, wordsToShow=100):
    # for each value in normalized contrib
    # assign color value
    
    for w in words:
        index, word, val = w
        # add word to color map
        #wordColorMap[word] = int(round(255*(1-val)))
        wordColorMap[word] = int(round(200*(1-val)))
    
    # generate text
    text = generateText(words, min(len(words), wordsToShow))

    wc = WordCloud(background_color="white", max_words=2000, mask=maskImg)
    wc.generate(text)
    #wc.recolor(color_func=gray_color_func)

    return wc    



def mergeWordClouds(clouds, wcSize):
    padding = 10
    margin = 20
    width = len(clouds[0]) * wcSize + (len(clouds[0]) - 1) * padding + 2 * margin
    height = len(clouds) * wcSize + (len(clouds) - 1) * padding + 2 * margin
    img = Image.new('RGB', (width, height), "white")
    for k, v in clouds.iteritems():
        #print (k, v)
        layer = k
        offsetX = 0
        if (layer == 1):
            # ((total width) - (layer width)) / 2.0
            offsetX = ( width - 2 * margin - (len(v) * wcSize + (len(v) - 1) * padding)) / 2
            
        for i in range(len(v)):
            wc = v[i]
            pasteX = margin + i * wcSize + i * padding + offsetX
            pasteY = height - margin - (layer + 1) * wcSize - layer * padding
            img.paste(wc.to_image(), (pasteX, pasteY))

    return img


if __name__ == "__main__":
    if (len(sys.argv) < 4):
        print ("runs `python tsne_viz.py structure.prototxt model.caffemodel corpus.lex`")
        sys.exit(1)

    structure = sys.argv[1]
    model = sys.argv[2]
    lexFile = sys.argv[3]

    # get word, vec, pairs
    lex = loadLexicon(lexFile)
    
    # init the network
    caffe.set_mode_cpu()
    net = loadNet(structure, model)
 
    if (not os.path.isdir(PICKLES_DIR)):
        os.makedirs(PICKLES_DIR)
    if (not os.path.isdir(IMAGE_DIR)):
        os.makedirs(IMAGE_DIR)
        
    maxActivations = []
    words = []

    for l in range(len(LAYERS)):
        # load data from file if pre computed
        SPEC_MAX_ACTS = "_iter" + model[model.rfind("_")+1:model.rfind(".caffemodel")] + "_l" + str(l+1)
        SPEC_WORDS = "_iter" + model[model.rfind("_")+1:model.rfind(".caffemodel")] + "_l" + str(l+1)
        
        MAX_ACTIVATIONS_FILE = "max_activations"+SPEC_MAX_ACTS+".pickle"
        WORDS_FILE = "words"+SPEC_WORDS+".pickle"

        layerName = LAYERS[l][0]

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
 
 
    wcSize = 300
 
    # load mask
    d = os.path.dirname(__file__)
    circle_mask = imread(os.path.join(d, "black_circle_mask_whitebg_"+str(wcSize)+"x"+str(wcSize)+".png"))

    while True:
        optimal = True if (raw_input("Compute optimal 1st layer nodes from 2nd layer node? y/n ") == "y") else False
        
        n = None
        layer = None
        clouds = {}

        if (optimal):
            n = int(raw_input("Choose 2nd layer node (0 - " + str(len(words[1])) + "): "))
            prevNum = int(raw_input("How many prev layer nodes? "))
            layer = 0
            nodes = list(getPrevLayerNodes(net, LAYERS[1][0], n, prevNum))
        else:
            layer = int(raw_input("Layer (0 - " + str(len(LAYERS)-1) + "): "))
            nodes = raw_input("Nodes (0 - " + str(len(words[layer])) + "), (e.g. 0, 4, 15, 30): ")
            if (nodes == ''):
                nodes = None
            else:
                nodes = [int(x.strip()) for x in nodes.split(',')]

        print ("nodes = ", nodes)
        for node in nodes:
            wc = generateWordCloud(words[layer][node], maskImg=circle_mask, wordsToShow=50)
            if (not layer in clouds):
                clouds[layer] = []
            clouds[layer].append(wc)
            
        if (optimal):
            show2ndWc = True if (raw_input("Show 2nd layer WC of node " + str(n) + "? y/n ") == "y") else False
            layer = 1
            if (show2ndWc):
                wc = generateWordCloud(words[layer][n], maskImg=circle_mask, wordsToShow=50)
                if (not layer in clouds):
                    clouds[layer] = []
                clouds[layer].append(wc)
            
        img = mergeWordClouds(clouds, wcSize)
        img.save(IMAGE_DIR + "merged_wordcloud_n"+str(n)+".png", "PNG")         

        

            
