# SMILES corrector
Collection of scripts to create and use SMILES corrector. The SMILES corrector is a transformer model that is trained to translate invalid SMILES sequences to valid SMILES sequences. 

## Installation (with python v 3.9)
1. RDKit (version >= 2020.03) 
$ conda install -c conda-forge rdkit

2. Numpy (version >= 1.19)
$ conda install numpy

3. Scikit-Learn (version >= 0.23)
$ conda install scikit-learn

4. Pandas (version >= 1.2.2)
$ conda install pandas

5. PyTorch (version >= 1.7)
$ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 

6. Matplotlib (version >= 2.0)
$ conda install matplotlib

7. TQMD
$ pip install tqdm

8. Torchtext
$ pip install torchtext==0.11.2 

9. Modin
$ pip install modin[ray]

10. Chembl structure pipeline
$ conda install -c conda-forge chembl_structure_pipeline

11. Seaborn
$ conda install seaborn

12. tueplots
$ pip install tueplots

13. jupyer
$ pip install jupyter

## Usage
There are two main options for using the SMILES corrector; use a previously trained model (step 4) or train a new model.

The files for reuse can be found on: https://zenodo.org/record/7157412#.Y1edr3ZBxD8

To train a new model follow example in test.py to perform the following steps

1. Prepare the molecules that will be used for training
standardize the molecules using standardize from preprocess.py (this for example removes salts & canonicalizes the SMILES)
remove molecules with SMILES longer than a set threshold (this is necessary to reduce the number of model parameters)

2. Introduce ‘artificial’ errors into training molecules, done in order to create invalid-valid pairs (for training & evaluating the model)
indicate which type of error to introduce using invalid_type = (could for example be all errors (‘all’) or only syntax, ring, parentheses, valence, aromaticity, bond exists & random permutations)
indicate how many errors to introduce per sequence using num_errors
introduce the errors using get_invalid_smiles from invalidSMILES.py
the resulting invalid-valid pairs are saved as training and validation set (& optional test set) (are later used by torchtext dataloader)

3. Train & evaluate model
indicate where to find real-world error set using error source = (apart from training & evaluating the model on artificial invalid SMILES it will also be evaluated on actual invalid SMILES created by de novo generation)
use model_training from modelling.py to train the model (this function will initialize a transformer model and train, evaluate and save it)
the transformer architecture is defined in transformer.py 
if a trained model already exists the weights from the last epoch (modelname_last) are loaded & existing model is trained further
evaluation is done using functions from metric.py
performance & model are saved in Data/performance/

4. Use model to correct invalid SMILES
done using correct_SMILES from modelling.py (best model is saved in Data/performance/[modelname].pkg (!= the modelname ending with _last))
for an example of how to use the SMILES corrector for exploration see testexplore.py 

## Support
Send a message to l.schoenmaker@lacdr.leidenuniv.nl
