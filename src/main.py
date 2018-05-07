
import random, argparse
import configuration
from ml import check_ml
from utilities import get_data

def main():
    """
    Usage: python main.py [ -c config_file -s config_section ]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', default="default.ini") 
    parser.add_argument('-s', '--config_section', default="DEFAULT") 
    args = parser.parse_args()
    config_file = args.config_file #"default.ini"
    config_section = args.config_section #"DEFAULT" # "DISJOINT" "TEST"
    CONFIG = configuration.Configuration(config_file, config_section) 
    n_seed = int(CONFIG.get("random_seed"))
    if n_seed != -1:
        random.seed(n_seed) # for reproducibility
    else:
        n_seed = None
    n_run = int(CONFIG.get("n_run"))
    knn = int(CONFIG.get("knn"))
    model_type = CONFIG.get("model_type")
    prediction_type = CONFIG.get("prediction_type")
    features = set(CONFIG.get("features").split("|"))
    recalculate_similarity = CONFIG.get_boolean("recalculate_similarity") 
    disjoint_cv = CONFIG.get_boolean("disjoint_cv") 
    try:
        split_both = CONFIG.get_boolean("pairwise_disjoint") 
    except:
        split_both = False
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
    check_ml(data, n_run, knn, n_fold, n_proportion, n_subset, model_type, prediction_type, features, recalculate_similarity, disjoint_cv, split_both, output_file, model_fun = None, n_seed = n_seed)
    return


if __name__ == "__main__":
    main()

