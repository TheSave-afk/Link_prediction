# Link_prediction
Link prediction for network using node2vec  

# Datasets
Ca-AstroPh, Facebook, Protein-Protein Interaction, Blog Catalog, Wikipedia  

# Usage
For Node2Vec mode it's the default, no argument is needed:
'''
python link_prediction.py --r 3 --f wikipedia.txt
'''
For deepwalk mode:
'''
python link_prediction.py --m deepwalk --r 3 --f wikipedia.txt
'''
