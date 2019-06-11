__all__ = ['Oracle','OraclePassive']

import numpy as np

from abc import ABCMeta,abstractmethod
import time

class Oracle(metaclass=ABCMeta):
    """An abstract oracle that returns triplets.

    Parameters
    ----------
    n_examples : int
        The number of training examples.

    n_examples_test : int
        The number of test examples.

    n_draws : int
        The longest sequence of random numbers that can be queried
        from any random state.
    
    seed : int or None
        The seed used to initialize the random number generators. If
        None the current time is used, that is
        int(time.time()). (Default: None).

    Attributes
    ----------
    n_examples : int
        The number of training examples.

    n_examples_test : int
        The number of test examples.

    n_draws : int
        The longest sequence of random numbers that can be queried
        from any random state.

    seed : int
        The seed used to initialize the random number generators.

    """
    def __init__(self,n_examples,n_examples_test,n_draws,seed=None):
        self.n_examples = n_examples
        
        self.n_examples_test = n_examples_test
        
        self.n_draws = n_draws
        
        if seed is not None:
            self.seed = seed
        else:
            self.seed = int(time.time())
                
    @abstractmethod
    def from_pair(j,k):
        """Returns the training triplets associated to a pair.

        In this method, i, j, and k are identifiers for training
        examples. Given an oracle object, this method should be
        deterministic and should return the same triplets for
        from_pair(j,k) and from_pair(k,j). The method _random_state
        can be used for this purpose.

        Parameters
        ----------
        j : int
            The identifier of the first training reference example.
        
        k : int
            The identifier of the second training reference example.
        

        Returns
        -------
        triplets : numpy array, shape (n_examples,)
            An array of values in {1,-1,0}. In entry i, the value 1
            indicates that the triplet (i,j,k) is available, the value
            -1 indicates that the triplet (i,k,j) is available, and
            the value 0 indicates that neither of the triplets is
            available.

        """
        pass

    @abstractmethod
    def from_pair_test(j,k):
        """Returns the test triplets associated to a pair.

        In this method, i, j, and k are identifiers for test
        examples. Given an oracle object, this method should be
        deterministic and should return the same triplets for
        from_pair_test(j,k) and from_pair_test(k,j). The method
        _random_state can be used for this purpose.

        Parameters
        ----------
        j : int
            The identifier of the first training reference example.
        
        k : int
            The identifier of the second training reference example.
        

        Returns
        -------
        triplets : numpy array, shape (n_examples_test,)
            An array of values in {1,-1,0}. In entry i, the value 1
            indicates that the triplet (i,j,k) is available, the value
            -1 indicates that the triplet (i,k,j) is available, and
            the value 0 indicates that neither of the triplets is
            available.

        """
        pass

    def _random_state(self,j,k):
        """Returns the reinitialized random state associated to the ordered
        pair of examples j,k and the sequence size n_draws.

        It is assumed that the random state will be used to generate
        at most n_draws number. Otherwise it might happen that
        sequences of random numbers overlap. This method can be used
        to ensure the deterministic nature of from_pair and
        from_pair_test.
                
        Parameters
        ----------
        j : int
            The identifier of the first training reference example.
        
        k : int
            The identifier of the second training reference example.

        n_draws : int
            The maximum number of examples that can be queried before
            having a risk of sequences overlapping.
            
        Returns
        -------
        random_state : numpy RandomState
            The reinitialized random state associated with the pair of
            examples j,k and the sequence size n_draws.

        """            
        seed = self.seed + int(j*self.n_examples + k)
        div = seed // (2**32)
        seed = seed % (2**32)
        
        random_state = np.random.RandomState(seed)
        
        random_state.rand(div*self.n_draws,)
        
        return random_state

    
class OraclePassive(Oracle):
    """An oracle that returns passively queried triplets from standard
    data.
    
    It uses random states as given by the superclass method
    _random_state to ensure that from_pair and from_pair_test are
    deterministic for a given instance.

    Parameters
    ----------
    x : numpy array, shape (n_examples,n_features)
        An array containing the training examples.

    x_test : numpy array, shape (n_examples_test,n_features)
        An array containing the test examples.

    metric : function
        The metric to use to compute the distance between examples. It
        should take two numpy arrays of shapes
        (n_examples_1,n_features) and (n_examples_2,n_features) and
        return a distance matrix of shape (n_examples_1,n_examples_2).
    
    proportion_triplets : float, optional
        The overall proportion of triplets that should be
        generated. (Default: 0.1).

    proportion_noise : float, optional
        The overall proportion of noise in the triplets. 
        (Default: 0.0).

    seed : int or None
        The seed used to initialize the random states. (Default:
        None).

    Attributes
    ----------
    x : numpy array, shape (n_examples,n_features)
        An array containing the training examples.

    x_test : numpy array, shape (n_examples_test,n_features)
        An array containing the test examples.

    metric : function
        The metric to use to compute the distance between examples.
    
    proportion_triplets : float
        The overall proportion of triplets that should be generated.

    proportion_noise : float
        The overall proportion of noise in the triplets.

    n_examples : int
        The number of training examples.

    n_examples_test : int
        The number of test examples.

    n_draws : int
        The longest sequence of random numbers that can be queried
        from any random state.

    seed : int
        The seed used to initialize the random states.

    """    
    def __init__(self,x,x_test,metric,proportion_triplets=0.1,proportion_noise=0.0,seed=None):
        self.x = x

        self.x_test = x_test
        
        self.metric = metric
        
        self.proportion_triplets = proportion_triplets

        self.proportion_noise = proportion_noise

        n_examples = x.shape[0]
        n_examples_test = x_test.shape[0]
        n_draws = (n_examples + n_examples_test)*2
        super(OraclePassive,self).__init__(n_examples,n_examples_test,n_draws,seed)

    def from_pair(self,j,k):
        if j == k:
            return np.zeros(self.n_examples)
        
        if j < k:
            random_state = self._random_state(j,k)
        else:
            random_state = self._random_state(k,j)
            
        selector_triplets = (random_state.rand(self.n_examples,) < self.proportion_triplets)
        selector_noise = (random_state.rand(self.n_examples,) < self.proportion_noise)
        
        distance_to_jk = self.metric(self.x,self.x[[j,k],:])

        triplets_noisy_all = np.logical_xor(distance_to_jk[:,0] <= distance_to_jk[:,1],selector_noise)
        triplets_noisy_passive = np.where(np.logical_and(selector_triplets,triplets_noisy_all),1,
                                          np.where(np.logical_and(selector_triplets,np.logical_not(triplets_noisy_all)),-1,0))
        
        return triplets_noisy_passive

    def from_pair_test(self,j,k):
        if j == k:
            return np.zeros(self.n_examples_test)
        
        if j < k:
            random_state = self._random_state(j,k)
        else:
            random_state = self._random_state(k,j)
        
        # Draw some ghost numbers to avoid using the same sequence as for the training triplets
        random_state.rand(self.n_examples*2,)

        selector_triplets = (random_state.rand(self.n_examples_test,) < self.proportion_triplets)
        selector_noise = (random_state.rand(self.n_examples_test,) < self.proportion_noise)
        
        distance_to_jk = self.metric(self.x_test,self.x[[j,k],:])

        triplets_noisy_all = np.logical_xor(distance_to_jk[:,0] <= distance_to_jk[:,1],selector_noise)
        triplets_noisy_passive = np.where(np.logical_and(selector_triplets,triplets_noisy_all),1,
                                          np.where(np.logical_and(selector_triplets,np.logical_not(triplets_noisy_all)),-1,0))
        
        return triplets_noisy_passive
