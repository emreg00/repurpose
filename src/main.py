
import random
from toolbox import configuration
from ml import get_data, check_ml

CONFIG = configuration.Configuration() 

def main():
    n_seed = int(CONFIG.get("random_seed"))
    if n_seed != -1:
	random.seed(n_seed) # for reproducibility
    n_run = int(CONFIG.get("n_run"))
    knn = int(CONFIG.get("knn"))
    model_type = CONFIG.get("model_type")
    prediction_type = CONFIG.get("prediction_type")
    features = set(CONFIG.get("features").split("|"))
    recalculate_similarity = CONFIG.get_boolean("recalculate_similarity") 
    disjoint_cv = CONFIG.get_boolean("disjoint_cv") 
    output_file = CONFIG.get("output_file")
    n_fold = int(CONFIG.get("n_fold"))
    n_proportion = int(CONFIG.get("n_proportion"))
    n_subset = int(CONFIG.get("n_subset")) # for faster results - subsampling
    drug_disease_file = CONFIG.get("drug_disease_file")
    drug_side_effect_file = CONFIG.get("drug_side_effect_file")
    drug_structure_file = CONFIG.get("drug_structure_file")
    drug_target_file = CONFIG.get("drug_target_file")
    # Get data
    data = get_data(drug_disease_file, drug_side_effect_file, drug_structure_file, drug_target_file)
    # Check prediction accuracy of ML classifier on the data set using the parameters above
    check_ml(data, n_run, knn, n_fold, n_proportion, n_subset, model_type, prediction_type, features, recalculate_similarity, disjoint_cv, output_file, model_fun = None)
    return


if __name__ == "__main__":
    main()

