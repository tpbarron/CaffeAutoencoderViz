# CaffeAutoencoderViz

These scripts can be used to view what the hidden nodes of a network trained on text data have learned. The scripts take a caffe model as parameters

`python tsne_viz.py structure.prototxt model.caffemodel corpus.lex`

`python tsne_viz.py structure.prototxt model.caffemodel corpus.lex [dopca [perplexity]]`

They load the trained caffemodel and give the user options to visualize various parts of the network.
