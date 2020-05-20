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
import argparse


# Embedding functions
edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "L1": lambda a, b: np.abs(a - b),
    "L2": lambda a, b: np.abs(a - b) ** 2,
}


def create_test_and_training_set(positive_training_set, percentage, negative_percentage, min_degree):

    # test graph
    positive_test_set = nx.Graph()
    positive_test_set.add_nodes_from(positive_training_set)

    # generate negative samples for training
    all_negative_samples = list(nx.non_edges(positive_training_set))
    accetable = [True for i in range(len(all_negative_samples))]

    node_number = positive_training_set.number_of_nodes()
    node_list = positive_training_set.nodes()


    for node in node_list:

        percent = int(round((percentage * len(positive_training_set.edges(node)))))
        edges = list(positive_training_set.edges(node))
        counter = 0

        edge_count = len(edges)
        for attempt in range(edge_count):

            j = random.randint(0, len(edges))
            u, v = edges.pop(j)

            if counter < percent:

                if(positive_training_set.degree[u] > min_degree and positive_training_set.degree[v] > min_degree):
                    positive_test_set.add_edge(u,v)
                    positive_training_set.remove_edge(u,v)
                    counter += 1
            else:
                break


        ''' Versione precedente
        for u,v in edges:

            if counter < percent:

                if(positive_training_set.degree[u] > min_degree and positive_training_set.degree[v] > min_degree):
                    positive_test_set.add_edge(u,v) # lo aggiungo nel test set
                    positive_training_set.remove_edge(u,v) # lo rimuovo dal training set sia da un lato che dall'altro
                    counter += 1
            else:
                break 
        '''  


    ##########################
    # generazione di esempi negativi per il training
    negative_edges_for_training = []

    for i in range(int((2*negative_percentage) * len(positive_training_set.edges()))) :
        index = random.randint(0, len(all_negative_samples))
        negative_edges_for_training.append(all_negative_samples[index])
        accetable[index] = False

    negative_training_set = nx.Graph()
    negative_training_set.add_nodes_from(G)
    negative_training_set.add_edges_from(negative_edges_for_training)  ## c'è il rischio di nodi isolati nei samples negativi così
    #############################

    
    ############################
    # generazione di esempi negativi per il test
    negative_edges_for_test = []

    i = 0
    while i < (int((2*negative_percentage) * len(positive_training_set.edges()))) :
        
        index = random.randint(0, len(all_negative_samples))
        
        if accetable[index] :
            negative_edges_for_test.append(all_negative_samples[index])
            accetable[index] = False
            i += 1

    negative_test_set = nx.Graph()
    negative_test_set.add_nodes_from(positive_training_set)
    negative_test_set.add_edges_from(negative_edges_for_test)
    ###########################


    return positive_test_set, positive_training_set, negative_test_set, negative_training_set


def get_edges_and_labels(positive, negative):

        edges = list(positive) + list(negative)
        labels = np.zeros(len(edges))
        labels[:len(positive)] = 1
        return edges, labels


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


# training and testing, computing score for each embedding function
def training_and_testing(filename):

    # FILES
    embedding_model_filename = './embeddings.model'
    # embedding_model_filename = args.e

    # CREATE GRAPH
    graph = nx.Graph()

    # read graph from file
    G = nx.read_edgelist(filename, create_using = graph, nodetype = int, data = (('weight',int),))

    # creo test e training
    min_degree = 2
    positive_test_set, positive_training_set, negative_test_set, negative_training_set = create_test_and_training_set(G, 0.2, 0.5, min_degree)

    # edges positivi e negativi con le loro labels
    edges, training_labels = get_edges_and_labels(positive_training_set.edges(), negative_training_set.edges())

    # Precompute probabilities and generate walks
    dims = 128

    node2vec = Node2Vec(positive_training_set, dimensions = dims, walk_length = 80, num_walks = 10, workers = 4)
    model = node2vec.fit(window = 10, min_count = 1, batch_words = 4)
    # model.save(embedding_model_filename) # Save model for later use
    # model.load(embedding_model_filename) # Load model from file

    X_Hadamard = edges_to_features(model, edges, edge_functions['hadamard'], dims)
    X_L1       = edges_to_features(model, edges, edge_functions['L1'], dims)
    X_L2       = edges_to_features(model, edges, edge_functions['L2'], dims)

    # LOGISTIC REGRESSION

    # train the model with Logistic Regression
    clf_H  = LogisticRegression()
    clf_L1 = LogisticRegression()
    clf_L2 = LogisticRegression()

    clf_H.fit(X_Hadamard, training_labels)
    clf_L1.fit(X_L1 , training_labels)
    clf_L2.fit(X_L2, training_labels)


    # TESTING - prediction

    # Embedding ++ Positive test set
    test_edges = list(positive_test_set.edges())

    positive_emb_test_Hadamard = edges_to_features(model, test_edges, edge_functions['hadamard'], dims)
    positive_emb_test_L1       = edges_to_features(model, test_edges, edge_functions['L1'], dims)
    positive_emb_test_L2       = edges_to_features(model, test_edges, edge_functions['L2'], dims)
        
    # Embedding -- Negative test set
    test_edges = list(negative_test_set.edges())

    negative_emb_test_Hadamard = edges_to_features(model, test_edges, edge_functions['hadamard'], dims)
    negative_emb_test_L1       = edges_to_features(model, test_edges, edge_functions['L1'], dims)
    negative_emb_test_L2       = edges_to_features(model, test_edges, edge_functions['L2'], dims)

    # Prepare whole test set for every embedding function
    Hadamard_edges, y_test_Hadamard = get_edges_and_labels(positive_emb_test_Hadamard, negative_emb_test_Hadamard)
    L1_edges, y_test_L1             = get_edges_and_labels(positive_emb_test_L1, negative_emb_test_L1)
    L2_edges, y_test_L2             = get_edges_and_labels(positive_emb_test_L2, negative_emb_test_L2)


    ## PREDICTION
    y_pred_test_Hadamard = clf_H.predict(Hadamard_edges)
    y_pred_test_L1       = clf_L1.predict(L1_edges)
    y_pred_test_L2       = clf_L2.predict(L2_edges)

    # Accuracy
    score_Hadamard = sklearn.metrics.accuracy_score(y_test_Hadamard,  y_pred_test_Hadamard)
    score_L1 = sklearn.metrics.accuracy_score(y_test_L1, y_pred_test_L1)
    score_L2 = sklearn.metrics.accuracy_score(y_test_L2, y_pred_test_L2)

    return score_Hadamard, score_L1, score_L2



#################### MAIN ####################

# Command line arguments

parser = argparse.ArgumentParser(description='Link prediction on graph edgelist dataset')
parser.add_argument("--f", default = 'ca-AstroPh.txt', help = 'dataset filename (.txt)')
parser.add_argument("--r", default = 1, help = 'number of executions for training and testing (>= 1)')
# parser.add_argument("--e", default = './embeddings.model', help = 'node embedding filename (.model)')
args = parser.parse_args()

# DATASET
filename = args.f
if not(os.path.isfile(filename)):
    print('Error: Dataset does not exist')
    exit()


# EXECUTION
runs = args.r # iterations for training and testing
if runs <= 0:
    runs = 1

Hadamard_sum = L1_sum = L2_sum = 0

for i in range(runs):
     score_Hadamard, score_L1, score_L2 = training_and_testing(filename)
     Hadamard_sum += score_Hadamard
     L1_sum += score_L1
     L2_sum += score_L2

Hadamard_avg_score = Hadamard_sum / (float(runs))
L1_avg_score = L1_sum / (float(runs))
L2_avg_score = L2_sum / (float(runs))

# Print the score on whole test set
print('++ Scores for ', runs, 'executions:\n')
print("++ ++ Hadamard embedding test: ", Hadamard_avg_score , '\n')
print("++ ++ L1 embedding test: "      , L1_avg_score       , '\n')
print("++ ++ L2 embedding test: "      , L2_avg_score       , '\n')