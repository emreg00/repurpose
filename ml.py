
import os, cPickle, numpy, random
from toolbox import TsvReader, configuration
# ML related
from sklearn import preprocessing
from sklearn import tree, ensemble
from sklearn import svm, linear_model, neighbors
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc, average_precision_score
#from scipy import interp
import time

CONFIG = configuration.Configuration() 

def main():
    #check_ml_all()
    n_seed = int(CONFIG.get("random_seed"))
    #random.seed(n_seed) # for reproducibility
    n_run = int(CONFIG.get("n_run"))
    knn = int(CONFIG.get("knn"))
    model_type = CONFIG.get("model_type")
    #prediction_type = "side effect" #"disease"
    prediction_type = CONFIG.get("prediction_type")
    features = set(CONFIG.get("features").split("|"))
    recalculate_similarity = CONFIG.get_boolean("recalculate_similarity") 
    disjoint_cv = CONFIG.get_boolean("disjoint_cv") 
    output_file = CONFIG.get("output_file")
    n_fold = int(CONFIG.get("n_fold"))
    n_proportion = int(CONFIG.get("n_proportion"))
    n_subset = int(CONFIG.get("n_subset")) # for faster results - subsampling
    check_ml(n_run, knn, n_fold, n_proportion, n_subset, model_type, prediction_type, features, recalculate_similarity, disjoint_cv, output_file, model_fun = None)
    return


def get_zhang_data():
    """
    d=read.csv("drug_protein.csv", row.names=1)
    rownames(d) = tolower(rownames(d))
    write.table(d, "drug_protein.dat", quote=F, sep="\t")
    Add 'Drug' as the first column header
    """
    # Get disease data
    file_name = CONFIG.get("drug_disease_file")
    parser = TsvReader.TsvReader(file_name, delim="\t")
    disease_to_index, drug_to_values = parser.read(fields_to_include=None)
    drugs = set(drug_to_values.keys())
    print len(drugs)
    #print drug_to_values.items()[:5]
    #print disease_to_index["acromegaly"]
    #print drug_to_values["carnitine"]
    # Get SE data
    file_name = CONFIG.get("drug_side_effect_file")
    parser = TsvReader.TsvReader(file_name, delim="\t")
    se_to_index, drug_to_values_se = parser.read(fields_to_include=None)
    drugs &= set(drug_to_values_se.keys())
    #print drug_to_values_se.items()[:5]
    print len(drug_to_values), len(drug_to_values_se), len(disease_to_index), len(se_to_index) 
    # Consider common drugs only
    print len(drugs)
    # Get structure data
    file_name = CONFIG.get("drug_structure_file")
    parser = TsvReader.TsvReader(file_name, delim="\t")
    structure_to_index, drug_to_values_structure = parser.read(fields_to_include=None)
    print len(drug_to_values_structure), len(structure_to_index)
    drugs &= set(drug_to_values_structure.keys())
    print len(drugs)
    # Get target data
    file_name = CONFIG.get("drug_target_file")
    parser = TsvReader.TsvReader(file_name, delim="\t")
    target_to_index, drug_to_values_target = parser.read(fields_to_include=None)
    print len(drug_to_values_target), len(target_to_index)
    drugs &= set(drug_to_values_target.keys())
    print len(drugs)
    drugs = sorted(drugs)
    return drugs, disease_to_index, drug_to_values, se_to_index, drug_to_values_se, drug_to_values_structure, drug_to_values_target


def get_similarity(drugs, drug_to_values, drugs_test = None):
    # Get se based similarity
    drug_to_index = dict((drug, i) for i, drug in enumerate(drugs))
    #M = numpy.matrix([ list([ 0 for j in xrange(len(drugs))]) for i in xrange(len(drugs)) ])
    if drugs_test is not None:
	n_feature = len(drug_to_values[drugs[0]][0]) # use the first drug to get the feature vector size
	#print len(set(drugs) & drugs_test), n_feature 
	M = numpy.matrix([ map(float, drug_to_values[drug][0]) if drug not in drugs_test else [0.0] * n_feature for drug in drugs ])
    else:
	M = numpy.matrix([ map(float, drug_to_values[drug][0]) for drug in drugs ])
    M_similarity = numpy.corrcoef(M)
    # This might be overkill, assigning 0s for some training drugs as well
    #M_similarity[numpy.isnan(M_similarity)] = 0.0 # the drugs in test set has 0 variance and accordingly NaN values
    #print M_similarity[drug_to_index["carnitine"]][drug_to_index["cefditoren"]]
    return drug_to_index, M_similarity


def get_drug_disease_mapping(drugs, drug_to_values, disease_to_index):
    # Get disease drug mapping
    disease_to_drugs = {}
    pairs = []
    classes = []
    for drug, values in drug_to_values.iteritems():
	#for i, val in enumerate(values):
	if drug not in drugs:
	    continue
	for disease, idx in disease_to_index.iteritems():
	    #flag = 0
	    if values[0][idx] == "1":
		disease_to_drugs.setdefault(disease, set()).add(drug)
	    #	flag = 1
	    #if disease in disease_to_drugs:
	    #	pairs.append((drug, disease))
	    #	classes.append(flag)
    # Get all pairs (for the diseases for which there is some drug)
    drugs_all = reduce(lambda x,y: x|y, disease_to_drugs.values())
    for disease, drugs_gold in disease_to_drugs.iteritems():
	for drug in drugs_all:
	    pairs.append((drug, disease))
	    flag = 0
	    if drug in drugs_gold: 
		flag = 1
	    classes.append(flag)
    print len(drugs), len(drugs_all), len(disease_to_drugs), len(pairs), sum(classes)
    return disease_to_drugs, pairs, classes


def balance_data_and_get_cv(pairs, classes, n_fold, n_proportion, n_subset, disjoint=False):
    """
    pairs: all possible drug-disease pairs
    classes: labels of these drug-disease associations (1: known, 0: unknown)
    n_fold: number of cross-validation folds
    n_proportion: proportion of negative instances compared to positives (e.g.,
    2 means for each positive instance there are 2 negative instances)
    n_subset: if not -1, it uses a random subset of size n_subset of the positive instances
    (to reduce the computational time for large data sets)
    disjoint: whether the cross-validation folds contain overlapping drugs (True) 
    or not (False)
    This function returns (pairs, classes, cv) after balancing the data and
    creating the cross-validation folds. cv is the cross validation iterator containing 
    train and test splits defined by the indices corresponding to elements in the 
    pairs and classes lists.
    """
    classes = numpy.array(classes)
    pairs = numpy.array(pairs)
    idx_true_list = [ list() for i in xrange(n_fold) ]
    idx_false_list = [ list() for i in xrange(n_fold) ]
    if disjoint:
	# Get train & test such that no drug in train is in test
	def get_groups(idx_true_list, idx_false_list, n_subset, n_proportion=1, shuffle=False):
	    """
	    >>> a = get_groups([[13,2,1],[14,3,4],[15,5,6]], [[7,8],[9,10],[11,12]], 1, 1, True)
	    """
	    n = len(idx_true_list)
	    if n_subset != -1:
		n_subset = n_subset / n 
	    for i in xrange(n):
		if n_subset == -1: # use all data
		    indices_test = idx_true_list[i] + idx_false_list[i][:n_proportion * len(idx_true_list[i])]
		else:
		    if shuffle:
			indices_test = random.sample(idx_true_list[i], n_subset) + random.sample(idx_false_list[i], n_proportion * n_subset)
		    else:
			indices_test = idx_true_list[i][:n_subset] + idx_false_list[i][:(n_proportion * n_subset)]
		indices_train = []
		for j in xrange(n):
		    if i == j:
			continue
		    if n_subset == -1: # use all data
			indices_train += idx_true_list[j] + idx_false_list[j][:n_proportion * len(idx_true_list[j])]
		    else:
			if shuffle:
			    indices_train += random.sample(idx_true_list[j], n_subset) + random.sample(idx_false_list[j], n_proportion * n_subset)
			else:
			    indices_train += idx_true_list[j][:n_subset] + idx_false_list[j][:(n_proportion * n_subset)]
		yield indices_train, indices_test
	i_random = random.randint(0,100) # for getting the shuffled drug names in the same fold below
	for idx, (pair, class_) in enumerate(zip(pairs, classes)):
	    drug, disease = pair
	    i = sum([ord(c) + i_random for c in drug]) % n_fold
	    if class_ == 0:
		idx_false_list[i].append(idx)
	    else:
		idx_true_list[i].append(idx)
	print "+/-:", map(len, idx_true_list), map(len, idx_false_list)
	cv = get_groups(idx_true_list, idx_false_list, n_subset, n_proportion, shuffle=True)
    else:
	indices_true = numpy.where(classes == 1)[0]
	indices_false = numpy.where(classes == 0)[0]
	if n_subset == -1: # use all data
		n_subset = len(classes)
	indices_true = indices_true[:n_subset]
	numpy.random.shuffle(indices_false)
	indices = indices_false[:(n_proportion*indices_true.shape[0])]
	print "+/-:", len(indices_true), len(indices), len(indices_false)
	pairs = numpy.concatenate((pairs[indices_true], pairs[indices]), axis=0)
	classes = numpy.concatenate((classes[indices_true], classes[indices]), axis=0) 
	#print pairs[classes==1] 
	#print pairs[classes==0]
	cv = cross_validation.StratifiedKFold(classes, n_folds=n_fold, shuffle=True)
    return pairs, classes, cv


def get_scores_and_labels(pairs, classes, drug_to_disease_to_scores):
    values = []
    for drug, disease in pairs:
	scores = drug_to_disease_to_scores[drug][disease]
	values.append(scores)
    X = numpy.asmatrix(values).reshape((len(values),len(values[0])))
    y = numpy.array(classes) 
    return X, y


def get_similarity_based_scores(drugs, disease_to_drugs, drug_to_index, list_M_similarity, knn = 20, pairs_train = None, pairs_test = None, approach = "all_vs_all", file_name=None):
    if file_name is not None and os.path.exists(file_name):
	drug_to_disease_to_scores = cPickle.load(open(file_name))
    else:
	drug_to_disease_to_scores = {}
	if approach == "all_vs_all":
	#if pairs_train is None and pairs_test is None:
	    for drug1 in drugs:
		idx1 = drug_to_index[drug1]
		drug_to_disease_to_scores[drug1] = {}
		for disease, drugs_gold in disease_to_drugs.iteritems():
		    scores = []
		    for M_similarity in list_M_similarity:
			vals = []
			for drug2 in drugs:
			    if drug1 == drug2:
				continue
			    idx2 = drug_to_index[drug2]
			    vals.append([M_similarity[idx1][idx2], int(drug2 in drugs_gold)])
			vals.sort()
			score = 0
			for sim, val in vals[-knn:]:
			    score += sim*val
			scores.append(score)
		    drug_to_disease_to_scores[drug1][disease] = scores
	elif approach == "train_test_vs_train_test":
	    #elif pairs_train is not None and pairs_test is not None:
	    drugs = set(zip(*pairs_train)[0])
	    drugs |= set(zip(*pairs_test)[0]) #reduce(lambda x,y: x|y, disease_to_drugs.values())
	    pairs_train = set((k, v) for k,v in pairs_train) 
	    pairs_test = set((k, v) for k,v in pairs_test) 
	    pairs = pairs_train | pairs_test
	    disease_to_drugs_train = disease_to_drugs.copy()
	    for drug, disease in pairs_test:
		# Remove the test drug from the gold standard to unify the score calculation below
		drugs_gold = disease_to_drugs_train[disease] - set([drug]) 
		disease_to_drugs_train[disease] = drugs_gold
	    for drug1, disease in pairs:
		drugs_gold = disease_to_drugs_train[disease]
		idx1 = drug_to_index[drug1]
		scores = []
		for M_similarity in list_M_similarity:
		    vals = []
		    for drug2 in drugs:
			if drug1 == drug2:
			    continue
			idx2 = drug_to_index[drug2]
			vals.append([M_similarity[idx1][idx2], int(drug2 in drugs_gold)])
		    vals.sort()
		    score = 0
		    for sim, val in vals[-knn:]: 
			score += sim*val
		    scores.append(score)
		d = drug_to_disease_to_scores.setdefault(drug1, {})
		d[disease] = scores
	elif approach == "train_vs_train":
	#elif pairs_test is None:
	    drugs = set(zip(*pairs_train)[0])
	    pairs_train = set((k, v) for k,v in pairs_train) 
	    for drug1 in drugs:
		idx1 = drug_to_index[drug1]
		drug_to_disease_to_scores[drug1] = {}
		for disease, drugs_gold in disease_to_drugs.iteritems():
		    if (drug1, disease) not in pairs_train: 
			continue
		    drugs_gold_mod = drugs_gold & drugs
		    scores = []
		    for M_similarity in list_M_similarity:
			vals = []
			for drug2 in drugs:
			    if drug1 == drug2:
				continue
			    idx2 = drug_to_index[drug2]
			    vals.append([M_similarity[idx1][idx2], int(drug2 in drugs_gold_mod)])
			vals.sort()
			score = 0
			for sim, val in vals[-knn:]:
			    score += sim*val
			scores.append(score)
		    drug_to_disease_to_scores[drug1][disease] = scores
	if file_name is not None:
	    cPickle.dump(drug_to_disease_to_scores, open(file_name, 'w'))
    #print drug_to_disease_to_scores["carnitine"]["asthma"]
    return drug_to_disease_to_scores


def get_classification_model(model_type, model_fun = None):
    """
    model_type: custom | svm | logistic | knn | tree | rf | gbc
    model_fun: the function implementing classifier when the model_type is custom
    The allowed values for model_type are custom, svm, logistic, knn, tree, rf, gbc
    corresponding to custom model provided in model_fun by the user or the default 
    models in Scikit-learn for support vector machine, k-nearest-neighbor, 
    decision tree, random forest and gradient boosting classifiers, respectively. 
    Returns the classifier object that provides fit and predict_proba methods.
    """
    if model_type == "svm":
	clf = svm.SVC(kernel='linear', probability=True, C=1)
    elif model_type == "logistic":
	clf = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0) #, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    elif model_type == "knn":
	clf = neighbors.KNeighborsClassifier(n_neighbors=5) #weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
    elif model_type == "tree":
	clf = tree.DecisionTreeClassifier(criterion='gini') #splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, class_weight=None, presort=False)
    elif model_type == "rf":
	clf = ensemble.RandomForestClassifier(n_estimators=10, criterion='gini') #, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    elif model_type == "gbc":
	clf = ensemble.GradientBoostingClassifier(n_estimators=100, loss='deviance', learning_rate=0.1, subsample=1.0) #, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
    elif model_type == "custom":
	if fun is None:
	    raise ValueError("Custom model requires fun argument to be defined!")
	clf = fun
    else:
	raise ValueError("Uknown model type: %s!" % model_type)
    return clf


def check_ml(n_run, knn, n_fold, n_proportion, n_subset, model_type, prediction_type, features, recalculate_similarity, disjoint_cv, output_file = None, model_fun = None):
    drugs, disease_to_index, drug_to_values, se_to_index, drug_to_values_se, drug_to_values_structure, drug_to_values_target = get_zhang_data()
    #data = get_zhang_data()
    #drugs, disease_to_index, drug_to_values, se_to_index, drug_to_values_se, drug_to_values_structure, drug_to_values_target = data
    if prediction_type == "disease":
	disease_to_drugs, pairs, classes = get_drug_disease_mapping(drugs, drug_to_values, disease_to_index)
    elif prediction_type == "side effect":
	# For side effect prediction
	disease_to_drugs, pairs, classes = get_drug_disease_mapping(drugs, drug_to_values_se, se_to_index)
	drug_to_values_se = drug_to_values
	se_to_index = disease_to_index
    else:
	raise ValueError("Uknown prediction_type: " + prediction_type)
    list_M_similarity = []
    #list_M_similarity = [M_similarity_se, M_similarity_chemical, M_similarity_target]
    if "phenotype" in features:
	drug_to_index, M_similarity_se = get_similarity(drugs, drug_to_values_se)
	list_M_similarity.append(M_similarity_se)
    if "chemical" in features:
	drug_to_index, M_similarity_chemical = get_similarity(drugs, drug_to_values_structure)
	list_M_similarity.append(M_similarity_chemical)
    if "target" in features:
	drug_to_index, M_similarity_target = get_similarity(drugs, drug_to_values_target)
	list_M_similarity.append(M_similarity_target)
    if output_file is not None:
	output_f = open(output_file, 'a')
	output_f.write("n_fold\tn_proportion\tn_subset\tmodel type\tprediction type\tfeatures\trecalculate\tdisjoint\tvariable\tauc.mean\tauc.sd\tauprc.mean\tauprc.sd\n")
    else:
	output_f = None
    values = []
    values2 = []
    for i in xrange(n_run): 
	pairs_, classes_, cv = balance_data_and_get_cv(pairs, classes, n_fold, n_proportion, n_subset, disjoint = disjoint_cv)
	val, val2 = check_ml_helper(drugs, disease_to_drugs, drug_to_index, list_M_similarity, pairs_, classes_, cv, knn, n_fold, n_proportion, n_subset, model_type, prediction_type, features, recalculate_similarity, disjoint_cv, output_f, model_fun)
	values.append(val)
	values2.append(val2)
    print numpy.mean(values), numpy.std(values), values
    if output_f is not None:
	output_f.write("%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%f\t%f\t%f\t%f\n" % (n_fold, n_proportion, n_subset, model_type, prediction_type, "|".join(features), recalculate_similarity, disjoint_cv, "avg", numpy.mean(values), numpy.std(values), numpy.mean(values2), numpy.std(values2)))
	output_f.close()
    return numpy.mean(values), numpy.mean(values2)


def check_ml_helper(drugs, disease_to_drugs, drug_to_index, list_M_similarity, pairs, classes, cv, knn, n_fold, n_proportion, n_subset, model_type, prediction_type, features, recalculate_similarity, disjoint_cv, output_f, model_fun):
    #clf = svm.SVC(kernel='linear', probability=True, C=1)
    #clf = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0) #, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    clf = get_classification_model(model_type, model_fun)
    all_auc = []
    all_auprc = []
    for i, (train, test) in enumerate(cv):
	#print test
	file_name = None # for saving results
	pairs_train = pairs[train]
	classes_train = classes[train] 
	pairs_test = pairs[test]
	classes_test = classes[test] 
	print "Fold", i, len(pairs_train), len(pairs_test) #, len(train), len(test)
	#print list(pairs_train)[:5]
	prev_time = time.time() 
	if recalculate_similarity:
	    drug_to_disease_to_scores = get_similarity_based_scores(drugs, disease_to_drugs, drug_to_index, list_M_similarity = list_M_similarity, knn = knn, pairs_train = pairs_train, pairs_test = None, approach = "train_vs_train", file_name = file_name) 
	else:
	    # Using similarity scores of all drugs, not only within the subset
	    drug_to_disease_to_scores = get_similarity_based_scores(drugs, disease_to_drugs, drug_to_index, list_M_similarity = list_M_similarity, knn = knn, pairs_train = pairs_train, pairs_test = pairs_test, approach = "train_test_vs_train_test", file_name = file_name) # similar to all_vs_all above, but removes the test pair
	print "t:", time.time() - prev_time 
	#print drug_to_disease_to_scores["trovafloxacin"]
	X, y = get_scores_and_labels(pairs_train, classes_train, drug_to_disease_to_scores)
	#print pairs_train[classes_train==1]
	if recalculate_similarity:
	    drug_to_disease_to_scores = get_similarity_based_scores(drugs, disease_to_drugs, drug_to_index, list_M_similarity = list_M_similarity, knn = knn, pairs_train = pairs_test, pairs_test = None, approach = "train_vs_train", file_name = file_name) 
	X_new, y_new = get_scores_and_labels(pairs_test, classes_test, drug_to_disease_to_scores)
	#print X_new[y_new==1,:]
	probas_ = clf.fit(X, y).predict_proba(X_new)
	fpr, tpr, thresholds = roc_curve(y_new, probas_[:, 1]) 
	roc_auc = auc(fpr, tpr)
	all_auc.append(roc_auc)
	prc_auc = average_precision_score(y_new, probas_[:, 1])
	all_auprc.append(prc_auc)
    print numpy.mean(all_auc), numpy.std(all_auc), all_auc
    if output_f is not None:
	output_f.write("%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%f\t%f\t%f\t%f\n" % (n_fold, n_proportion, n_subset, model_type, prediction_type, "|".join(features), recalculate_similarity, disjoint_cv, "cv", numpy.mean(all_auc), numpy.std(all_auc), numpy.mean(all_auprc), numpy.std(all_auprc)))
    return numpy.mean(all_auc), numpy.mean(all_auprc)


### SOMEWHAT OBSOLETE ###
def check_ml_all():
    base_dir = "../data/"
    drugs, disease_to_index, drug_to_values, se_to_index, drug_to_values_se, drug_to_values_structure, drug_to_values_target = get_zhang_data()
    disease_to_drugs, pairs, classes = get_drug_disease_mapping(drugs, drug_to_values, disease_to_index)
    drug_to_index, M_similarity = get_similarity(drugs, drug_to_values_se)
    # Use all data
    file_name = base_dir + "drug_to_disease_to_scores.pcl"
    drug_to_disease_to_scores = get_similarity_based_scores(drugs, disease_to_drugs, drug_to_index, [M_similarity], pairs_train = None, pairs_test = None, approach = "all_vs_all", file_name = file_name)
    #drug_to_disease_to_scores = get_similarity_based_scores(drugs, disease_to_drugs, drug_to_index, list_M_similarity = list_M_similarity, knn = knn, pairs_train = None, pairs_test = None, approach = "all_vs_all", file_name = file_name) # use all the info
    # Create training data
    X_new, y_new = get_training_data(drug_to_disease_to_scores, disease_to_drugs)
    # Feature weights
    # Tree 
    clf = tree.DecisionTreeClassifier() #random_state=51234 #criterion='gini', max_depth=None, max_features=None, min_samples_leaf=1, min_samples_split=2)
    #clf = clf.fit(X_new, y_new)
    #print clf.feature_importances_
    print cross_validation.cross_val_score(clf, X_new, y_new, cv=5, scoring="roc_auc")
    #print values
    # SVM - CV AUC
    clf = svm.SVC(kernel='linear', probability=True, C=1)
    scores = cross_validation.cross_val_score(clf, X_new, y_new, cv=5, scoring="roc_auc")
    print scores
    cv = cross_validation.StratifiedKFold(y_new, n_folds=5)
    #mean_tpr = 0.0
    #mean_fpr = numpy.linspace(0, 1, 100)
    all_auc = []
    for i, (train, test) in enumerate(cv):
	# Compute ROC curve and area the curve
	probas_ = clf.fit(X_new[train], y_new[train]).predict_proba(X_new[test])
	fpr, tpr, thresholds = roc_curve(y_new[test], probas_[:, 1]) 
	#fpr, tpr, thresholds = roc_curve(y_new[test], X_new[test] / X_new[test].max())
	roc_auc = auc(fpr, tpr)
	all_auc.append(roc_auc)
	#mean_tpr += interp(mean_fpr, fpr, tpr)
    print numpy.mean(all_auc), all_auc
    #mean_tpr[0] = 0.0
    #mean_tpr /= len(cv)
    #mean_tpr[-1] = 1.0
    #mean_auc = auc(mean_fpr, mean_tpr)
    #print mean_auc
    return

def get_training_data(drug_to_disease_to_scores, disease_to_drugs):
    values = []
    classes = []
    for drug, disease_to_scores in drug_to_disease_to_scores.iteritems():
	for disease, scores in disease_to_score.iteritems():
	    values.append(scores)
	    flag = int(drug in disease_to_drugs[disease])
	    classes.append(flag)
    X = numpy.asmatrix(values).reshape((len(values),len(values[0])))
    y = numpy.array(classes) 
    print X.shape, y.shape
    #print X[1:10], y[1:10]
    X, y = preprocess(X, y)
    print X.shape, y.shape
    return X, y

def preprocess(X, y, balance=True, scale=True):
    X_true = X[y==1,:]
    X_false = X[y==0,:]
    # Balance True & False
    if balance:
	indices = range(X_false.shape[0])
	random.shuffle(indices)
	indices = indices[:X_true.shape[0]]
	X_false = X[indices,:]
	print X_true.shape, X_false.shape
    X_new = numpy.concatenate((X_true, X_false), axis=0)
    y_new = numpy.concatenate((numpy.ones(X_true.shape[0]), numpy.zeros(X_true.shape[0])), axis=0) 
    # Scaling
    if scale:
	X_new = preprocessing.scale(X_new)
    return X_new, y_new


if __name__ == "__main__":
    main()

