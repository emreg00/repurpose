import unittest
from utilities import balance_data_and_get_cv, get_similarity, get_similarity_based_scores

class TestUtilityMethods(unittest.TestCase):

    def setUp(self):
        # Mock data: 4 drugs defined by 3 features, 2 diseases
        self.drugs = ["c1", "c2", "c3", "c4"]
        self.diseases = ["p1", "p2"]
        self.drug_to_values = dict([("c1", [[1,0,0]]), ("c2", [[1,1,0]]), ("c3", [[1,0,1]]), ("c4", [[0,0,1]])])
        self.pairs = [ (drug, disease) for drug in self.drugs for disease in self.diseases ]
        #self.disease_to_drugs = dict([("p1", set(["c1"])), ("p2", set(["c2"]))])
        #self.classes = [1, 0, 0, 1, 0, 0, 0, 0]
        #self.n_proportion = 2
        #self.n_fold = 2
        #self.drug_to_index = dict(zip(["c1", "c2", "c3", "c4"], range(4)))


    def test_balance_data_and_get_cv_twofold(self):
        classes = [1, 0, 0, 1, 0, 0, 0, 0] # 2 known drug-disease associations
        n_fold = 2
        n_proportion = 1
        pairs_, classes_, cv = balance_data_and_get_cv(self.pairs, classes, n_fold, n_proportion, disjoint=False)
        # Each fold should contain 1 positive 1 negative instance
        for train, test in cv:
            self.assertEqual(len(train), 2)
            self.assertEqual(len(test), 2)
            self.assertEqual(sum(classes_[train]), 1)
            self.assertEqual(sum(classes_[test]), 1)
            #self.assertTrue(all(train != test))
            self.assertEqual(len(set(train) & set(test)), 0)


    def test_balance_data_and_get_cv_twofold_doublenegative(self):
        classes = [1, 0, 0, 1, 0, 0, 0, 0] # 2 known drug-disease associations
        n_fold = 2
        n_proportion = 2
        pairs_, classes_, cv = balance_data_and_get_cv(self.pairs, classes, n_fold, n_proportion, disjoint=False)
        # Each fold should contain 1 positive 2 negative instances
        for train, test in cv:
            self.assertEqual(len(train), 3)
            self.assertEqual(len(test), 3)
            self.assertEqual(sum(classes_[train]), 1)
            self.assertEqual(sum(classes_[test]), 1)
            self.assertEqual(len(set(train) & set(test)), 0)


    def test_balance_data_and_get_cv_threefold_doublenegative(self):
        classes = [1, 1, 0, 1, 0, 0, 0, 0] # 3 known drug-disease associations
        n_fold = 3
        n_proportion = 2
        pairs_, classes_, cv = balance_data_and_get_cv(self.pairs, classes, n_fold, n_proportion, disjoint=False)
        # Each fold should contain 1 positive 2 negative instances, with the exception of last 
        # fold which would containg 1 positive 1 negative since there are 8 pairs in data set
        for i, (train, test) in enumerate(cv):
            # Folds will have 3 / 3 / 2 (last group one data point less) 
            if i == (n_fold - 1): 
                self.assertEqual(len(train), 6)
                self.assertEqual(len(test), 2)
            else:            
                self.assertEqual(len(train), 5)
                self.assertEqual(len(test), 3)
            self.assertEqual(sum(classes_[train]), 2)
            self.assertEqual(sum(classes_[test]), 1)
            self.assertEqual(len(set(train) & set(test)), 0)


    def test_balance_data_and_get_cv_twofold_doublenegative_disjoint(self):
        classes = [1, 0, 0, 1, 0, 0, 0, 0] # 2 known drug-disease associations
        n_fold = 2
        n_proportion = 2
        pairs_, classes_, cv = balance_data_and_get_cv(self.pairs, classes, n_fold, n_proportion, disjoint=True)
        # Each fold should contain 1 positive 2 negative instances, none of the first elements in the pair 
        # in one fold should appear in the other (and accordingly train and test sets are disjoint w.r.t. to the first
        # elements in the pairs
        for train, test in cv:
            #print pairs_[train], pairs_[test]
            self.assertEqual(len(train), 3)
            self.assertEqual(len(test), 3)
            self.assertEqual(sum(classes_[train]), 1)
            self.assertEqual(sum(classes_[test]), 1)
            self.assertEqual(len(set(train) & set(test)), 0)
            self.assertEqual(len(set(zip(*pairs_[train])[0]) & set(zip(*pairs_[test])[0])), 0)


    def test_get_similarity(self):
        drug_to_index, M = get_similarity(self.drugs, self.drug_to_values)
        drug1 = "c1"
        # c1-c2 similarity cor([1,0,0], [1,1,0]) == 0.5
        drug2 = "c2"
        self.assertAlmostEqual(M[drug_to_index[drug1]][drug_to_index[drug2]], 0.5)
        # c1-c3 similarity cor([1,0,0], [1,0,1]) == 0.5
        drug2 = "c3"
        self.assertAlmostEqual(M[drug_to_index[drug1]][drug_to_index[drug2]], 0.5)
        # c1-c4 similarity cor([1,0,0], [0,0,1]) == -0.5
        drug2 = "c4"
        self.assertAlmostEqual(M[drug_to_index[drug1]][drug_to_index[drug2]], -0.5)


    def test_get_similarity_based_scores_knn2(self):
        # c1 is used for p1, c2 is used for p2
        disease_to_drugs = dict([("p1", set(["c1"])), ("p2", set(["c2"]))])
        # Get similarity between all pair of drugs
        drug_to_index, M = get_similarity(self.drugs, self.drug_to_values)
        list_M_similarity = [ M ]
        knn = 2
        drug_to_disease_to_scores = get_similarity_based_scores(self.drugs, disease_to_drugs, drug_to_index, list_M_similarity, knn)
        # c1's similarity to c2, c3, c4: 0.5, 0.5, -0.5
        # c1's score for p2: 0.5*label(c2) + 0.5*label(c3) = 0.5 * 1 + 0 = 0.5
        self.assertAlmostEqual(drug_to_disease_to_scores["c1"]["p2"][0], 0.5)


if __name__ == "__main__":
    #main()
    unittest.main()

