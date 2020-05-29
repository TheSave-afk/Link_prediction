# Link_prediction
Link prediction for network using node2vec  

# Datasets
Ca-AstroPh, Facebook, Protein-Protein Interaction, Blog Catalog, Wikipedia  

# Usage
For Node2Vec mode it's the default, no argument is needed:
```
python link_prediction.py --r 3 --f dataset_name
```
For deepwalk mode:
```
python link_prediction.py --m deepwalk --r 3 --f dataset_name
```

# List of all command line arguments

- --r : number of executions (default 1)
- --m : node embedding mode (default 'Node2Vec')
- --f : input dataset file name, an edge list in .txt format (default 'ca-AstroPh.txt')
- --l : log file name, where result are stored (default 'log.txt')

Run 
```
python link_prediction.py --help
```
to see full list in prompt
