import pickle
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

filename = 'Replication notebook.ipynb'
with open(filename) as ff:
    nb_in = nbformat.read(ff, nbformat.NO_CONVERT)
    
ep = ExecutePreprocessor(timeout=50000, kernel_name='python3')

stored_results = ep.preprocess(nb_in)
with open('stored_results.pkl','wb') as f:
    pickle.dump(stored_results,f)
