
import os, cPickle, numpy, random
from toolbox import TsvReader, configuration
# ML related
from sklearn import preprocessing
from sklearn import tree
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc
from scipy import interp

CONFIG = configuration.Configuration() 
#base_dir = "/home/eguney/data/gottlieb/zhang/"

def main():
    #check_ml_all()
    n_seed = int(CONFIG.get("random_seed"))
    #random.seed(n_seed) #!
    n_run = int(CONFIG.get("n_run"))
    values = []
    for i in xrange(n_run): 
	val = check_ml()
	values.append(val)
    print numpy.mean(values), values
    return


def get_zhang_data():
    """
    d=read.csv("drug_protein.csv", row.names=1)
    rownames(d) = tolower(rownames(d))
    write.table(d, "drug_protein.dat", quote=F, sep="\t")
    Add 'Drug' as the first column header
    """
    # Get disease data
    #file_name = base_dir + "drug_disease.dat"
    file_name = CONFIG.get("drug_disease_file")
    parser = TsvReader.TsvReader(file_name, delim="\t")
    disease_to_index, drug_to_values = parser.read(fields_to_include=None,)
    drugs = set(drug_to_values.keys())
    print len(drugs)
    #print drug_to_values.items()[:5]
    #print disease_to_index["acromegaly"]
    #print drug_to_values["carnitine"]
    # Get SE data
    #file_name = base_dir + "drug_sider.dat"
    file_name = CONFIG.get("drug_side_effect_file")
    parser = TsvReader.TsvReader(file_name, delim="\t")
    se_to_index, drug_to_values_se = parser.read(fields_to_include=None)
    drugs &= set(drug_to_values_se.keys())
    #print drug_to_values_se.items()[:5]
    print len(drug_to_values), len(drug_to_values_se), len(disease_to_index), len(se_to_index) 
    # Consider common drugs only
    print len(drugs)
    # Get structure data
    #file_name = base_dir + "drug_structure.dat"
    file_name = CONFIG.get("drug_structure_file")
    parser = TsvReader.TsvReader(file_name, delim="\t")
    structure_to_index, drug_to_values_structure = parser.read(fields_to_include=None)
    print len(drug_to_values_structure), len(structure_to_index)
    drugs &= set(drug_to_values_structure.keys())
    print len(drugs)
    # Get target data
    #file_name = base_dir + "drug_protein.dat"
    file_name = CONFIG.get("drug_target_file")
    parser = TsvReader.TsvReader(file_name, delim="\t")
    target_to_index, drug_to_values_target = parser.read(fields_to_include=None)
    print len(drug_to_values_target), len(target_to_index)
    drugs &= set(drug_to_values_target.keys())
    print len(drugs)
    drugs = sorted(drugs)
    return drugs, disease_to_index, drug_to_values, drug_to_values_se, drug_to_values_structure, drug_to_values_target


def get_similarity(drugs, drug_to_values):
    # Get se based similarity
    drug_to_index = dict((drug, i) for i, drug in enumerate(drugs))
    #M = numpy.matrix([ list([ 0 for j in xrange(len(drugs))]) for i in xrange(len(drugs)) ])
    M = numpy.matrix([ map(float, drug_to_values[drug][0]) for drug in drugs ])
    M_similarity = numpy.corrcoef(M)
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
	    flag = 0
	    if values[0][idx] == "1":
		disease_to_drugs.setdefault(disease, set()).add(drug)
		flag = 1
	    if disease in disease_to_drugs:
		pairs.append((drug, disease))
		classes.append(flag)
    print len(drugs), len(disease_to_drugs), len(pairs)
    return disease_to_drugs, pairs, classes


def balance_data(pairs, classes):
    classes = numpy.array(classes)
    indices_true = numpy.where(classes == 1)[0]#.tolist()
    #indices_true = indices_true[:100] 
    indices_false = numpy.where(classes == 0)[0]#.tolist()
    numpy.random.shuffle(indices_false)
    n_proportion = int(CONFIG.get("n_proportion"))
    indices = indices_false[:(n_proportion*indices_true.shape[0])]
    print len(indices), len(indices_true), len(indices_false)
    pairs = numpy.array(pairs)
    pairs = numpy.concatenate((pairs[indices_true], pairs[indices]), axis=0)
    classes = numpy.concatenate((classes[indices_true], classes[indices]), axis=0) 
    #print pairs[classes==1] 
    #print pairs[classes==0]
    return pairs, classes


def check_ml():
    drugs, disease_to_index, drug_to_values, drug_to_values_se, drug_to_values_structure, drug_to_values_target = get_zhang_data()
    disease_to_drugs, pairs, classes = get_drug_disease_mapping(drugs, drug_to_values, disease_to_index)
    drug_to_index, M_similarity_se = get_similarity(drugs, drug_to_values_se)
    drug_to_index, M_similarity_chemical = get_similarity(drugs, drug_to_values_structure)
    drug_to_index, M_similarity_target = get_similarity(drugs, drug_to_values_target)
    pairs, classes = balance_data(pairs, classes)
    n_fold = int(CONFIG.get("n_fold"))
    cv = cross_validation.StratifiedKFold(classes, n_folds=n_fold, shuffle=True)
    clf = svm.SVC(kernel='linear', probability=True, C=1)
    all_auc = []
    for i, (train, test) in enumerate(cv):
	#print test
	#file_name = base_dir + "drug_to_disease_to_scores.pcl.%d" % i 
	file_name = None #!
	pairs_train = pairs[train]
	classes_train = classes[train] 
	pairs_test = pairs[test]
	classes_test = classes[test] 
	print i, len(pairs_train), len(train), len(test)
	#print list(pairs_train)[:5]
	drug_to_disease_to_scores = get_similarity_based_scores(drugs, disease_to_drugs, drug_to_index, list_M_similarity = [M_similarity_se, M_similarity_chemical, M_similarity_target], pairs_train = pairs_train, pairs_test = pairs_test, file_name = file_name)
	#print drug_to_disease_to_scores["trovafloxacin"]
	X, y = get_scores_and_labels(pairs_train, classes_train, drug_to_disease_to_scores)
	#print pairs_train[classes_train==1]
	X_new, y_new = get_scores_and_labels(pairs_test, classes_test, drug_to_disease_to_scores)
	#print X_new[y_new==1,:]
	probas_ = clf.fit(X, y).predict_proba(X_new)
	fpr, tpr, thresholds = roc_curve(y_new, probas_[:, 1]) 
	roc_auc = auc(fpr, tpr)
	all_auc.append(roc_auc)
    print numpy.mean(all_auc), all_auc
    return numpy.mean(all_auc)


def check_ml_all():
    drugs, disease_to_index, drug_to_values, drug_to_values_se, drug_to_values_structure, drug_to_values_target = get_zhang_data()
    disease_to_drugs, pairs, classes = get_drug_disease_mapping(drugs, drug_to_values, disease_to_index)
    drug_to_index, M_similarity = get_similarity(drugs, drug_to_values_se)
    # Use all data
    file_name = base_dir + "drug_to_disease_to_scores.pcl"
    drug_to_disease_to_scores = get_similarity_based_scores(drugs, disease_to_drugs, drug_to_index, M_similarity, pairs = None, file_name = file_name)
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

def get_scores_and_labels(pairs, classes, drug_to_disease_to_scores):
    values = []
    for drug, disease in pairs:
	scores = drug_to_disease_to_scores[drug][disease]
	values.append(scores)
    X = numpy.asmatrix(values).reshape((len(values),len(values[0])))
    y = numpy.array(classes) 
    #X, y = preprocess(X, y, scale = True)
    return X, y

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

def get_similarity_based_scores(drugs, disease_to_drugs, drug_to_index, list_M_similarity, pairs_train = None, pairs_test = None, file_name=None):
    if file_name is not None and os.path.exists(file_name):
	drug_to_disease_to_scores = cPickle.load(open(file_name))
    else:
	drug_to_disease_to_scores = {}
	if pairs_train is None:
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
			for sim, val in vals[-20:]:
			    score += sim*val
			scores.append(score)
		    drug_to_disease_to_scores[drug1][disease] = scores
	else:
	    drugs = set(zip(*pairs_train)[0])
	    drugs |= set(zip(*pairs_test)[0]) #reduce(lambda x,y: x|y, disease_to_drugs.values())
	    pairs_train = set((k, v) for k,v in pairs_train) 
	    pairs_test = set((k, v) for k,v in pairs_test) 
	    pairs = pairs_train | pairs_test
	    disease_to_drugs_train = disease_to_drugs.copy()
	    for drug, disease in pairs_test:
		#if disease in disease_to_drugs:
		drugs_gold = disease_to_drugs_train[disease] - set([drug])
		disease_to_drugs_train[disease] = drugs_gold
		#disease_to_drugs_train.setdefault(disease, set()).add(drug) # before iterating over train
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
		    for sim, val in vals[-20:]: 
			score += sim*val
		    scores.append(score)
		d = drug_to_disease_to_scores.setdefault(drug1, {})
		d[disease] = scores
	if file_name is not None:
	    cPickle.dump(drug_to_disease_to_scores, open(file_name, 'w'))
    #print drug_to_disease_to_scores["carnitine"]["asthma"]
    return drug_to_disease_to_scores


if __name__ == "__main__":
    main()

