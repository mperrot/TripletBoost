import unittest

from ..algorithm import TripletBoost

from ..oracle import OraclePassive
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class TestTripletBoost(unittest.TestCase):
    """Unittest for the class TripletBoost."""
    
    def setUp(self):
        """Use the same data in all the unittest."""
        random_state = np.random.RandomState(0)
        x = random_state.rand(100,2)
        self.y = np.zeros((100,1))
        self.y[50:,:] = 1
        
        x_test = random_state.rand(100,2)

        self.oracle = OraclePassive(x,x_test,pairwise_distances,seed=0)

    def test_is_online_fit(self):
        """Check that fit can be called in an online fashion."""
        np.random.seed(0)
        
        tripletboost_initial = TripletBoost(self.oracle,self.y)
        tripletboost_initial.fit(2)
        
        np.random.seed(0)
        
        tripletboost_equal = TripletBoost(self.oracle,self.y)
        tripletboost_equal.fit(1)
        tripletboost_equal.fit(1)

        for classifier_initial,classifier_equal in zip(tripletboost_initial.classifiers,tripletboost_equal.classifiers):
            for key in classifier_initial.keys():
                self.assertTrue(key in classifier_equal.keys())
                self.assertTrue(np.all(classifier_initial[key] == classifier_equal[key]))

    def test_is_not_increasing_error_fit(self):
        """Check that fit does not increase the error."""
        tripletboost = TripletBoost(self.oracle,self.y)
        
        error_last = tripletboost.error
        for _ in range(100):
            tripletboost.fit(1)
            error_new = tripletboost.error
            self.assertTrue(error_new <= error_last)
            error_last = error_new

    def test_shape_predict_scores(self):
        """Check the shape of the array returned by predict_score."""
        tripletboost = TripletBoost(self.oracle,self.y)
        tripletboost.fit(100)

        y_test_scores = tripletboost.predict_scores(self.oracle.n_examples_test)

        self.assertEqual(y_test_scores.shape,(self.oracle.n_examples_test,tripletboost.n_labels))

    def test_predict_all(self):
        """Check that predict_all returns only labels with positive scores and only them."""
        tripletboost = TripletBoost(self.oracle,self.y)
        tripletboost.fit(100)

        y_test_scores = tripletboost.predict_scores(self.oracle.n_examples_test)
        y_test = tripletboost.predict_all(self.oracle.n_examples_test)

        for labels,scores in zip(y_test,y_test_scores):
            for label in labels:
                self.assertTrue(scores[tripletboost.unique_labels.index(label)] > 0)
            for other_label in tripletboost.unique_labels:
                if other_label not in labels:
                    self.assertTrue(scores[tripletboost.unique_labels.index(other_label)] <= 0)

    def test_predict_top(self):
        """Check that predict_top returns only the top scoring label."""
        tripletboost = TripletBoost(self.oracle,self.y)
        tripletboost.fit(100)

        y_test_scores = tripletboost.predict_scores(self.oracle.n_examples_test)
        y_test = tripletboost.predict_top(self.oracle.n_examples_test)

        for label,scores in zip(y_test,y_test_scores):
            for other_label in tripletboost.unique_labels:
                if other_label != label:
                    self.assertTrue(scores[tripletboost.unique_labels.index(label)] >= scores[tripletboost.unique_labels.index(other_label)])
            
