
import numpy
from sklearn.metrics import roc_curve, auc, average_precision_score
import utilities


def check_ml(data, n_run, knn, n_fold, n_proportion, n_subset, model_type, prediction_type, features, recalculate_similarity, disjoint_cv, output_file = None, model_fun = None, verbose=False):
    drugs, disease_to_index, drug_to_values, se_to_index, drug_to_values_se, drug_to_values_structure, drug_to_values_target = data
    if prediction_type == "disease":
	disease_to_drugs, pairs, classes = utilities.get_drug_disease_mapping(drugs, drug_to_values, disease_to_index)
    elif prediction_type == "side effect":
	# For side effect prediction
	disease_to_drugs, pairs, classes = utilities.get_drug_disease_mapping(drugs, drug_to_values_se, se_to_index)
	drug_to_values_se = drug_to_values
	se_to_index = disease_to_index
    else:
	raise ValueError("Uknown prediction_type: " + prediction_type)
    list_M_similarity = []
    if "phenotype" in features:
	drug_to_index, M_similarity_se = utilities.get_similarity(drugs, drug_to_values_se)
	list_M_similarity.append(M_similarity_se)
    if "chemical" in features:
	drug_to_index, M_similarity_chemical = utilities.get_similarity(drugs, drug_to_values_structure)
	list_M_similarity.append(M_similarity_chemical)
    if "target" in features:
	drug_to_index, M_similarity_target = utilities.get_similarity(drugs, drug_to_values_target)
	list_M_similarity.append(M_similarity_target)
    if output_file is not None:
	output_f = open(output_file, 'a')
	output_f.write("n_fold\tn_proportion\tn_subset\tmodel type\tprediction type\tfeatures\trecalculate\tdisjoint\tvariable\tauc.mean\tauc.sd\tauprc.mean\tauprc.sd\n")
    else:
	output_f = None
    values = []
    values2 = []
    for i in xrange(n_run): 
	pairs_, classes_, cv = utilities.balance_data_and_get_cv(pairs, classes, n_fold, n_proportion, n_subset, disjoint = disjoint_cv)
	val, val2 = check_ml_helper(drugs, disease_to_drugs, drug_to_index, list_M_similarity, pairs_, classes_, cv, knn, n_fold, n_proportion, n_subset, model_type, prediction_type, features, recalculate_similarity, disjoint_cv, output_f, model_fun, verbose)
	values.append(val)
	values2.append(val2)
    print "AUC over runs: %.1f (+/-%.1f):" % (numpy.mean(values), numpy.std(values)), map(lambda x: round(x, ndigits=1), values)
    if output_f is not None:
	output_f.write("%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%f\t%f\t%f\t%f\n" % (n_fold, n_proportion, n_subset, model_type, prediction_type, "|".join(features), recalculate_similarity, disjoint_cv, "avg", numpy.mean(values), numpy.std(values), numpy.mean(values2), numpy.std(values2)))
	output_f.close()
    return "AUC: %.1f" % numpy.mean(values), "AUPRC: %.1f" % numpy.mean(values2)


def check_ml_helper(drugs, disease_to_drugs, drug_to_index, list_M_similarity, pairs, classes, cv, knn, n_fold, n_proportion, n_subset, model_type, prediction_type, features, recalculate_similarity, disjoint_cv, output_f, model_fun, verbose):
    clf = utilities.get_classification_model(model_type, model_fun)
    all_auc = []
    all_auprc = []
    for i, (train, test) in enumerate(cv):
	file_name = None # for saving results
	pairs_train = pairs[train]
	classes_train = classes[train] 
	pairs_test = pairs[test]
	classes_test = classes[test] 
	if recalculate_similarity:
	    drug_to_disease_to_scores = utilities.get_similarity_based_scores(drugs, disease_to_drugs, drug_to_index, list_M_similarity = list_M_similarity, knn = knn, pairs_train = pairs_train, pairs_test = None, approach = "train_vs_train", file_name = file_name) 
	else:
	    # Using similarity scores of all drugs, not only within the subset
	    drug_to_disease_to_scores = utilities.get_similarity_based_scores(drugs, disease_to_drugs, drug_to_index, list_M_similarity = list_M_similarity, knn = knn, pairs_train = pairs_train, pairs_test = pairs_test, approach = "train_test_vs_train_test", file_name = file_name) # similar to all_vs_all above, but removes the test pair
	X, y = get_scores_and_labels(pairs_train, classes_train, drug_to_disease_to_scores)
	if recalculate_similarity:
	    drug_to_disease_to_scores = utilities.get_similarity_based_scores(drugs, disease_to_drugs, drug_to_index, list_M_similarity = list_M_similarity, knn = knn, pairs_train = pairs_test, pairs_test = None, approach = "train_vs_train", file_name = file_name) 
	X_new, y_new = get_scores_and_labels(pairs_test, classes_test, drug_to_disease_to_scores)
	probas_ = clf.fit(X, y).predict_proba(X_new)
	fpr, tpr, thresholds = roc_curve(y_new, probas_[:, 1]) 
	roc_auc = 100*auc(fpr, tpr)
	all_auc.append(roc_auc)
	prc_auc = 100*average_precision_score(y_new, probas_[:, 1])
	all_auprc.append(prc_auc)
	if verbose:
	    print "Fold:", i+1, "# train:", len(pairs_train), "# test:", len(pairs_test), "AUC: %.1f" % roc_auc, "AUPRC: %.1f" % prc_auc
    #if verbose:
    #	print "AUC: %.1f (+/-%.1f):" % (numpy.mean(all_auc), numpy.std(all_auc)), all_auc
    if output_f is not None:
	output_f.write("%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%f\t%f\t%f\t%f\n" % (n_fold, n_proportion, n_subset, model_type, prediction_type, "|".join(features), recalculate_similarity, disjoint_cv, "cv", numpy.mean(all_auc), numpy.std(all_auc), numpy.mean(all_auprc), numpy.std(all_auprc)))
    return numpy.mean(all_auc), numpy.mean(all_auprc)


def get_scores_and_labels(pairs, classes, drug_to_disease_to_scores):
    values = []
    for drug, disease in pairs:
	scores = drug_to_disease_to_scores[drug][disease]
	values.append(scores)
    X = numpy.asmatrix(values).reshape((len(values),len(values[0])))
    y = numpy.array(classes) 
    return X, y




