
# Repurpose: A Python-based platform for reproducible similarity-based drug repurposing

Over the past years many methods for similarity-based (a.k.a. knowledge-based, 
guilt-by-association-based) drug repurposing, yet most of these studies do not 
provide the code or the model used in the study. To improve reproducibility, 
we present a Python-platform offering
- drug feature data parsing and similarity calculation 
- data balancing
- (disjoint) cross validation 
- classifier building

Using this platform we investigate the effect using unseen data in the test
set in similarity-based classification.

See Jupyter (IPython) [Repurpose Notebook](repurpose.ipynb) for reproducing the analysis 
presented in the manuscript and example runs.

See [DDI Notebook](interaction.ipynb) for the analysis of drug-drug interaction
prediction using drug similarity.

## Requirements
The Python platform has the following dependencies:

- [Numpy](http://www.numpy.org)
- [Scikit-learn](http://scikit-learn.org)


## Installing & running tests

Just download (i.e. clone) the files to your computer, no additional install is required.
Several test cases for the methods in `utilities.py` are provided in `tests.py`. 
To run these, type

```python
python tests.py
```

It should give an output similar to below
>......
>----------------------------------------------------------------------
>Ran 6 tests in 0.002s
>
>OK


## Data sets
The data sets used in the analysis are freely available 
[online](http://astro.temple.edu/~tua87106/drugreposition.html)

We have modified these data sets slightly for parsing in Python by
- converting all drug, disease and side effect terms to lowercase
- removing the quotations and making the text tab delimited
- we also added the 'Drug' text to the header 

These modified files are available under `data/` folder.

We have also retrieve pharmocokinetic drug-drug interaction (DDI) information
from [DrugBank](http://drugbank.ca/) database (v4.5.0) and mapped the drugs 
on the data set above.

## Usage

For running the code with the default parameters defined in the `default.ini` in `src/` directory, type

```python
config_file = "default.ini"
config_section = "DEFAULT"
python main.py -c config_file -s config_section 
```

Alternatively, for using the `check_ml` method that builds a machine learning classifier to predict
drug-disease associations using a cross-validation scheme, include the following in the python code

```python
import ml
ml.check_ml(data, n_run, knn, n_fold, n_proportion, n_subset, model_type, prediction_type, features, recalculate_similarity, disjoint_cv, split_both = False, output_file = None, model_fun = None, verbose = False, n_seed = None)
```

data can be loaded using the following function

```python
import utilities
data = utilities.get_data(drug_disease_file, drug_side_effect_file, drug_structure_file, drug_target_file, drug_interaction_file=None)
```

See the [Repurpose Notebook](repurpose.ipynb) for several use cases on repurposing drugs using chemical, target profile and side effect similarity. For drug-drug interaction prediction using drug similarity, see the [DDI Notebook](interaction.ipynb).


## Customizing the experimental settings
The configuration information for the experiments are in `default.ini`. The 
path of the data file has to be defined based on your local file structure.

Parameters in `default.ini`:

- drug_disease_file: File containing drug-disease associations (a binary matrix where rows are drugs, columns are diseases) 
- drug_side_effect_file = File containing drug-side effect associations (a binary matrix where rows are drugs, columns are side effects) 
- drug_structure_file = File containing drug-chemical sub structure mapping (a binary matrix where rows are drugs, columns are substructures) 
- drug_target_file: File containing drug-target mapping (a binary matrix where rows are drugs, columns are targets)
- output_file: File in which the output AUC and AUPRC values are going to be stored
- random_seed: A number to assign use as seed to random package functions (set it an integer for reproducibility, if -1 the output would vary depending on the random selection) 
- model_type: Machine learning model to be  used to build the classifier, either svm | logistic | knn | tree | rf | gbc
- prediction_type = Whether the classifier will be build to predict drug-disease ('disease') or drug-side effect ('side effect') associations
- features = Features to be used to build the classifier, a combination of chemical | target | phenotype 
- disjoint: Whether the cross-validation folds contain overlapping drugs (True) or not (False)
- pairwise_disjoint : Whether the cross-validation folds should group both of the pairs within the same group
- recalculate_similarity = Whether to recalculate k-NN based drug-disease and drug-side effect association score within training and test sets (True: recalculate, default, False: do not recalculate)
- knn = Number of most similar drugs to consider while calculating drug-disease and drug-side effect association score
- n_fold: Number of cross-validation folds
- n_proportion: Proportion of negative instances compared to positives (e.g., 2 means for each positive instance there are 2 negative instances)
- n_subset: If not -1, it uses a random subset of size n_subset of the positive instances (to reduce the computational time for large data sets)
- n_run = Number of repetitions of cross-validation analysis


## Customizing the methods
- Data balancing and cross validation (in `utilities.py`)

```python
balance_data_and_get_cv(pairs, classes, n_fold, n_proportion, n_subset=-1, disjoint=False, split_both=False, n_seed=None)
```

Input parameters:
    pairs: all possible drug-disease pairs
    classes: labels of these drug-disease associations (1: known, 0: unknown)
    n_fold: number of cross-validation folds
    n_proportion: proportion of negative instances compared to positives (e.g.,
    2 means for each positive instance there are 2 negative instances)
    n_subset: if not -1, it uses a random subset of size n_subset of the positive instances
    (to reduce the computational time for large data sets)
    disjoint: whether the cross-validation folds contain overlapping drugs (True) or not (False)
    split_both: whether the cross-validation folds should group both of the pairs within the same group
    n_seed: number to feed to the random generator to for reproducibility (of the cross-validation folds)

Output:
    This function returns (pairs, classes, cv) after balancing the data and
    creating the cross-validation folds. cv is the cross validation iterator containing 
    train and test splits defined by the indices corresponding to elements in the 
    pairs and classes lists.

- Classifier model (in `utilities.py`)

```python
get_classification_model(model_type, model_fun = None)
```

Input parameters:
    model_type: custom | svm | logistic | knn | tree | rf | gbc
    model_fun: the function implementing classifier when the model_type is custom

    The allowed values for model_type are custom, svm, logistic, knn, tree, rf, gbc
    corresponding to custom model provided in model_fun by the user or the default 
    models in Scikit-learn for support vector machine, k-nearest-neighbor, 
    decision tree, random forest and gradient boosting classifiers, respectively. 

Output:
    Returns the classifier object that provides fit and predict_proba methods.


## Citation

Guney E., REPRODUCIBLE DRUG REPURPOSING: WHEN SIMILARITY DOES NOT SUFFICE.
Pac Symp Biocomput. 2016;22:132-143. [Pubmed](https://www.ncbi.nlm.nih.gov/pubmed/27896969)

