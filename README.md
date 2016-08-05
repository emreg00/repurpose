
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

## Data sets
The data sets used in the analysis are freely available 
[online](http://astro.temple.edu/~tua87106/drugreposition.html)

The Python platform has the following dependencies:

- [Numpy](http://www.numpy.org)
- [Scikit-learn](http://scikit-learn.org)
- [Toolbox](https://github.com/emreg00/toolbox)

## Usage
> python ml.py

## Customizing the experimental settings
The configuration information for the experiments are in default.ini. The 
path of the data file has to be defined based on your local file structure.

## Customizing the methods
- data balancing and cross validation

> balance_data_and_get_cv(pairs, classes, n_fold, n_proportion, n_subset, disjoint=False)

Input parameters:
    pairs: all possible drug-disease pairs
    classes: labels of these drug-disease associations (1: known, 0: unknown)
    n_fold: number of cross-validation folds
    n_proportion: proportion of negative instances compared to positives (e.g.,
    2 means for each positive instance there are 2 negative instances)
    n_subset: if not -1, it uses a random subset of size n_subset of the positive instances
    (to reduce the computational time for large data sets)
    disjoint: whether the cross-validation folds contain overlapping drugs (True) 
    or not (False)

Output:
    This function returns (pairs, classes, cv) after balancing the data and
    creating the cross-validation folds. cv is the cross validation iterator containing 
    train and test splits defined by the indices corresponding to elements in the 
    pairs and classes lists.

> get_classification_model(model_type, model_fun = None)

Input parameters:
    model_type: custom | svm | logistic | knn | tree | rf | gbc
    model_fun: the function implementing classifier when the model_type is custom

    The allowed values for model_type are custom, svm, logistic, knn, tree, rf, gbc
    corresponding to custom model provided in model_fun by the user or the default 
    models in Scikit-learn for support vector machine, k-nearest-neighbor, 
    decision tree, random forest and gradient boosting classifiers, respectively. 

Output:
    Returns the classifier object that provides fit and predict_proba methods.

## Coming soon

- A local copy of the data sets
- Refactoring for isolating customizable functions and removing toolbox dependency
- Jupyter notebook with example runs

