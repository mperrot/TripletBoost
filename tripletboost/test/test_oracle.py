import unittest

from ..oracle import OraclePassive

import numpy as np
import time
from sklearn.metrics.pairwise import pairwise_distances

class TestOraclePassive(unittest.TestCase):
    """Unittest for the class OraclePassive."""
    
    def setUp(self):
        """Use the same data in all the unittest."""
        random_state = np.random.RandomState(0)
        x = random_state.rand(100,2)
        
        x_test = random_state.rand(100,2)

        self.oracle = lambda proportion_triplets,proportion_noise,seed: OraclePassive(x,x_test,pairwise_distances,proportion_triplets=proportion_triplets,proportion_noise=proportion_noise,seed=seed)
    
    def test_is_seedable(self):
        """Check that the oracle can be seeded."""
        oracle_initial = self.oracle(1,0,0)
        oracle_equal = self.oracle(1,0,0)
        oracle_different = self.oracle(1,0,1)
        
        self.assertTrue(np.all(oracle_initial._random_state(0,1).rand(100) == oracle_equal._random_state(0,1).rand(100)))
        self.assertFalse(np.all(oracle_initial._random_state(0,1).rand(100) == oracle_different._random_state(0,1).rand(100)))

    def test_is_random(self):
        """Check that by default the oracle is random."""
        oracle_initial = self.oracle(1,0,None)
        time.sleep(3)
        oracle_different = self.oracle(1,0,None)

        self.assertFalse(np.all(oracle_initial._random_state(0,1).rand(100) == oracle_different._random_state(0,1).rand(100)))
        
    def test_is_deterministic_random_state(self):
        """Check that _random_state is a deterministic method."""
        oracle = self.oracle(1,0,0)

        random_states = {}
        for j in range(oracle.n_examples):
            for k in range(oracle.n_examples):
                random_states[(j,k)] = oracle._random_state(j,k)

        for key,random_state in random_states.items():
            deterministic_random_state = oracle._random_state(key[0],key[1])
            self.assertTrue(np.all(random_state.rand(100) == deterministic_random_state.rand(100)))
                
    def test_is_deterministic_from_pair(self):
        """Check that from_pair is a deterministic method."""
        oracle = self.oracle(0.1,0.1,0)

        triplets_all = {}
        for j in range(oracle.n_examples):
            for k in range(oracle.n_examples):
                triplets_all[(j,k)] = oracle.from_pair(j,k)

        for key,triplets in triplets_all.items():
            deterministic_triplets = oracle.from_pair(key[0],key[1])
            self.assertTrue(np.all(triplets == deterministic_triplets))
            
    def test_is_deterministic_from_pair_test(self):
        """Check that from_pair_test is a deterministic method."""
        oracle = self.oracle(0.1,0.1,0)

        triplets_all = {}
        for j in range(oracle.n_examples):
            for k in range(oracle.n_examples):
                triplets_all[(j,k)] = oracle.from_pair_test(j,k)

        for key,triplets in triplets_all.items():
            deterministic_triplets = oracle.from_pair_test(key[0],key[1])
            self.assertTrue(np.all(triplets == deterministic_triplets))
            
    def test_is_symmetric_from_pair(self):
        """Check that from_pair is a symmetric method."""
        oracle = self.oracle(1,0.1,0)

        triplets_all = {}
        for j in range(oracle.n_examples):
            for k in range(oracle.n_examples):
                triplets_all[(j,k)] = oracle.from_pair(j,k)

        for key,triplets in triplets_all.items():
            symmetric_triplets = oracle.from_pair(key[1],key[0])
            self.assertTrue(np.all(triplets == -symmetric_triplets),(key[0],key[1],triplets,-symmetric_triplets))
            
    def test_is_symmetric_from_pair_test(self):
        """Check that from_pair_test is a symmetric method."""
        oracle = self.oracle(1,0.1,0)

        triplets_all = {}
        for j in range(oracle.n_examples):
            for k in range(oracle.n_examples):
                triplets_all[(j,k)] = oracle.from_pair_test(j,k)

        for key,triplets in triplets_all.items():
            symmetric_triplets = oracle.from_pair_test(key[1],key[0])
            self.assertTrue(np.all(triplets == -symmetric_triplets))
        
    def test_proportion_triplets(self):
        """Check that proportion_triplets controls the number of triplets."""
        oracle = self.oracle(0.2,0,0)
        
        triplets = oracle.from_pair(0,1)

        n_triplets_queried = oracle.n_examples - np.sum(triplets == 0)

        self.assertTrue(np.abs(n_triplets_queried-oracle.n_examples*0.2) < oracle.n_examples*0.2*0.8)
        
        triplets = oracle.from_pair_test(0,1)

        n_triplets_queried = oracle.n_examples_test - np.sum(triplets == 0)

        self.assertTrue(np.abs(n_triplets_queried-oracle.n_examples_test*0.2) < oracle.n_examples_test*0.2*0.8)

        
    def test_proportion_noise_from_pair(self):
        """Check that proportion_noise controls the amount of noise."""
        oracle = self.oracle(1,0,0)
        oracle_noisy = self.oracle(1,0.2,0)
        
        triplets = oracle.from_pair(0,1)
        triplets_noisy = oracle_noisy.from_pair(0,1)

        n_triplets_different = np.sum(triplets != triplets_noisy)

        self.assertTrue(np.abs(n_triplets_different-oracle.n_examples*0.2) < oracle.n_examples*0.2*0.8)
        
        triplets = oracle.from_pair_test(0,1)
        triplets_noisy = oracle_noisy.from_pair_test(0,1)

        n_triplets_different = np.sum(triplets != triplets_noisy)

        self.assertTrue(np.abs(n_triplets_different-oracle.n_examples_test*0.2) < oracle.n_examples_test*0.2*0.8)
