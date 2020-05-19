import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import numpy as np
import urllib
import csv
import math 
import random
from node2vec.edges import HadamardEmbedder
import sklearn
import os
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


# Funzioni di embedding
edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
}


'''
TODO: Assicurarsi della CASUALITA' delle scelte nella funzione create, calcolare l'accuratezza con una media su N run diverse
      e confrontare i risultati sui tre diversi dataset (Astro, facebook, ProteinProtein...)
      Per ogni dataset fare anche il confronto fra l'embedding con hadamard e quello con L1
      
      ACCURACY = (TP + TN) / (TP + TN + FP + FN)

      Il testing va fatto su positive + negative test set 
'''
# ricevendo un grafo in ingresso crea i due set
def create_test_and_training_set(positive_training_set, percentage, negative_percentage, min_degree):

    # creo il grafo del test set con gli stessi nodi del training set
    positive_test_set = nx.Graph()
    positive_test_set.add_nodes_from(positive_training_set)

    # generate negative samples for training
    all_negative_samples = list(nx.non_edges(positive_training_set))
    accetable = [True for i in range(len(all_negative_samples))] # inizializzo una lista di false per decidere se l'arco è accetabile o no

    node_number = positive_training_set.number_of_nodes()
    node_list = positive_training_set.nodes()

    for i in node_list:

        l = len(positive_training_set.edges(i))
        percent = int (round((percentage * l)))
        edges = list(positive_training_set.edges(i))
        counter = 0

        # DOMANDA: gli edges vanno mescolati per aumentare la casualità?
        for u,v in edges:
            if counter < percent:
                if(positive_training_set.degree[u] > min_degree and positive_training_set.degree[v] > min_degree):
                    positive_test_set.add_edge(u,v) # lo aggiungo nel test set
                    positive_training_set.remove_edge(u,v) # lo rimuovo dal training set sia da un lato che dall'altro
                    counter += 1
            else:
                break    


    ##########################
    # generazione di esempi negativi per il training
    negative_edges_for_training = []

    for i in range(int((2*negative_percentage) * len(positive_training_set.edges()))) :
        index = random.randint(0,len(all_negative_samples))
        negative_edges_for_training.append(all_negative_samples[index])
        accetable[index] = False

    negative_training_set = nx.Graph() # DOMANDA: Serve per forza creare un nuovo grafo o basta salvarsi la lista degli edges?
    negative_training_set.add_nodes_from(G)
    negative_training_set.add_edges_from(negative_edges_for_training)  ## c'è il rischio di nodi isolati nei samples negativi così
    #############################

    
    ############################
    # generazione di esempi negativi per il test
    negative_edges_for_test = []

    i=0
    while i < (int((2*negative_percentage) * len(positive_training_set.edges()))) :
        index = random.randint(0,len(all_negative_samples))
        if accetable[index] :
            negative_edges_for_test.append(all_negative_samples[index])
            accetable[index] = False
            i+=1

    negative_test_set = nx.Graph() # DOMANDA: Serve per forza creare un nuovo grafo o basta salvarsi la lista degli edges?
    negative_test_set.add_nodes_from(G)
    negative_test_set.add_edges_from(negative_edges_for_test)
    ###########################

    ###########################
    return positive_test_set, positive_training_set, negative_test_set, negative_training_set

# ritorna gli edges con le corrispondenti labels
def get_edges_for_training(positive, negative):
        edges = list(positive) + list(negative)
        labels = np.zeros(len(edges))
        labels[:len(positive)] = 1
        return edges, labels


# effettua l'edge embedding
def edges_to_features(model, edge_list, edge_function, dimensions):
        n_tot = len(edge_list)
        features_vec = np.empty((n_tot, dimensions), dtype = 'f')

        # Iterate over edges
        for ii in range(n_tot):
            v1, v2 = edge_list[ii]

            # Edge-node features
            emb1 = np.asarray(model[str(v1)])
            emb2 = np.asarray(model[str(v2)])

            # Calculate edge feature
            features_vec[ii] = edge_function(emb1, emb2)

        return features_vec

# SCORES
'''
TODO: Va modificata per calcolare l'accuracy con la formula giusta...
'''
def get_test_score(test_type, prediction_list):

    ones = 0
    samples = len(prediction_list)

    for i in range(samples):
        if prediction_list[i] == 1:
            ones += 1
    
    score = 0

    if test_type == 'positive':
        score = (float(ones)) / (float(samples))
    else:
        score = (float(samples - ones)) / (float(samples))

    return score


def print_sample(samples, prediction_list):
    for i in range(samples):
        print(prediction_list[i])
        print('\n')

## MAIN ##

# FILES
EMBEDDING_MODEL_FILENAME = './embeddings.model'

dims = 128


# CREATE GRAPH
FileName = "ca-AstroPh.txt"
graph = nx.Graph()

# read graph from file
G = nx.read_edgelist(FileName, create_using=graph, nodetype=int, data=(('weight',int),))

# creo test e training
min_degree = 2
positive_test_set, positive_training_set, negative_test_set, negative_training_set = create_test_and_training_set(G, 0.2, 0.5, min_degree)

# edges positivi e negativi con le loro labels
edges, labels = get_edges_for_training(positive_training_set.edges(), negative_training_set.edges())

# Precompute probabilities and generate walks
node2vec = Node2Vec(positive_training_set, dimensions = dims, walk_length = 80, num_walks = 10, workers = 4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
model.save(EMBEDDING_MODEL_FILENAME) # Save model for later use

# print("++ Modello creato\n")

X_hadamard = edges_to_features(model, edges, edge_functions['hadamard'], dims)
X_l2 = edges_to_features(model,edges,edge_functions['l2'], dims)
X_l1 = edges_to_features(model,edges,edge_functions['l1'], dims)

# print("++ Training: Edge embedding completato\n")

clf_H, clf_l1, clf_l2 = LogisticRegression()

# train the model with Logistic Regression
clf_H.fit(X_hadamard, labels)
clf_l1.fit(X_l1 , labels)
clf_l2.fit(X_l2, labels)


# TESTING - prediction

# Embedding ++ Positive test set
test_edges = list(positive_test_set.edges())

positive_emb_test_H  = edges_to_features(model, test_edges, edge_functions['hadamard'], dims)
positive_emb_test_l1 = edges_to_features(model, test_edges, edge_functions['l1'], dims)
positive_emb_test_l2 = edges_to_features(model, test_edges, edge_functions['l2'], dims)

###
# Embedding -- Negative test set
negative_emb_test_H  = edges_to_features(model, test_edges, edge_functions['hadamard'], dims)
negative_emb_test_l1 = edges_to_features(model, test_edges, edge_functions['l1'], dims)
negative_emb_test_l2 = edges_to_features(model, test_edges, edge_functions['l2'], dims)

#UNIONE dei due test set per formare il finale embeddato
y_test_H  = get_edges_for_training(positive_emb_test_H,negative_emb_test_H)
y_test_l1 = get_edges_for_training(positive_emb_test_l1,negative_emb_test_l1)
y_test_l2 = get_edges_for_training(positive_emb_test_l2,negative_emb_test_l2)

###

## PREDICTION
y_pred_test_H  = clf_H. predict(y_test_H)
y_pred_test_l1 = clf_l1.predict(y_test_l1)
y_pred_test_l2 = clf_l2.predict(y_test_l2)


score_H  = sklearn.metrics.accuracy_score(y_test_H,  y_pred_test_H)
score_l1 = sklearn.metrics.accuracy_score(y_test_l1, y_pred_test_l1)
score_l2 = sklearn.metrics.accuracy_score(y_test_l2, y_pred_test_l2)

#PRINT the score on whole test set 
print("Score of Hadamard embedding test:" + score_H  + '\n')
print("Score of L1 embedding test:"       + score_l1 + '\n')
print("Score of L2 embedding test:"       + score_l2 + '\n')

'''
score = get_test_score('positive', y_pos_test)
print("++ Score for positive test: " + score + '\n')

samples = 10

print("++ Positive test for " + samples + " samples \n")
print_sample(samples, y_pos_test)



# -- Negative test
test_edges = list(negative_test_set.edges())
emb_test = edges_to_features(model, test_edges, edge_functions['hadamard'], dims)

y_neg_test = clf.predict(emb_test)
score = get_test_score('positive', y_neg_test)
print("++ Score for negative test: " + score + '\n')


print("++ Negative test for " + samples + " samples \n")
print_sample(samples, y_neg_test)

'''


