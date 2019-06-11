__all__ = ['TripletBoost']

import numpy as np
import pickle,gzip,time

class TripletBoost:
    """TripletBoost

    Parameters
    ----------
    oracle : Oracle object
        An oracle used to query the triplets.
    y : list of (list of labels), len (n_examples)
        A list containing the (list of labels) of each example.
        
    Attributes
    ----------
    oracle : Oracle object
        The oracle used to query the triplets.
    
    y : list of (list of labels), len (n_examples)
        A list containing the (list of labels) of each example.
    
    unique_labels : list of labels, len (n_labels)
        A list containing all the unique labels.
    
    n_labels : int
        The number of unique labels.
    
    n_examples : int
        The number of training examples.
    
    y_binary : numpy array, shape (n_examples, n_labels) 
        An array where each column coresponds to a unique label (in
        the same order as in unique_labels) and contains 1 in each
        example row associated to this label and -1 in the other.
    
    y_weights : numpy array, shape (n_examples, n_labels) 
        An array containing the weight of each
        (example,label)-pair. The weights are initialized to be
        uniform.
    
    classifiers : list of classifiers
        The list of weak triplet classifiers selected by TripletBoost
        and whose weights are different from 0.

    error : float
        The training error after training for n_iterations.

    n_iterations : int
        The number of boosting iterations performed so far.
    
    time_elapsed : float
        The time taken to train the model so far. It also includes the
        time taken by the oracle to generate the triplets.

    Notes
    -----
    To use the fit method of TripletBoost, the Oracle object should
    exhibit a from_pair(j,k) method that takes the indices of two
    training examples and returns a numpy array of shape (n_examples,)
    of values in {1,-1,0}. In entry i, the value 1 indicates that the
    triplet (i,j,k) is available, the value -1 indicates that the
    triplet (i,k,j) is available, and the value 0 indicates that
    neither of the triplets is available. This method should be
    deterministic, that is repeated calls to from_pair(j,k) should
    always return the same value.

    To use the prediction methods of TripletBoost, the Oracle object
    should also exhibit a from_pair_test(j,k) method that takes the
    indices of two training examples and returns a numpy array of
    shape (n_examples_test,) of values in {1,-1,0}. In entry i, the
    value 1 indicates that the triplet (i,j,k) is available, the value
    -1 indicates that the triplet (i,k,j) is available, and the value
    0 indicates that neither of the triplets is available. This method
    should be deterministic, that is repeated calls to
    from_pair_test(j,k) should always return the same value.

    """
    def __init__(self,oracle,y):
        self.oracle = oracle
        
        self.y = y

        unique_labels = []
        for labels in self.y:
            unique_labels.extend(labels)
        self.unique_labels = list(set(unique_labels))

        self.n_labels = len(self.unique_labels)
        
        self.n_examples = len(y)
        
        self.y_binary = -np.ones((self.n_examples,self.n_labels))
        for i,labels in enumerate(y):
            for label in labels:
                self.y_binary[i,self.unique_labels.index(label)] = 1
            
        self.y_weights = np.full((self.n_examples,self.n_labels),1/(self.n_examples*self.n_labels))

        self.classifiers = []
        
        self.error = 1
        
        self.n_iterations = 0
        
        self.time_elapsed = 0
            
    def fit(self,n_new_iterations):
        """Performs n_new_iterations iterations of boosting to improve the
        current model.
                
        Parameters
        ----------
        n_new_iterations : int
            The number of rounds of boosting to perform.

        Returns
        -------
        self : object

        """     
        time_start = time.process_time()
        
        for _ in range(n_new_iterations):
            classifier,classifier_train_masks = self._get_classifier()

            if classifier is not None:
                self.classifiers.append(classifier)
                
                normalization_factor = self._update_weights(classifier,classifier_train_masks)
        
                self.error *= normalization_factor
                
        time_end = time.process_time()
        self.time_elapsed += (time_end-time_start)

        return self
        
    def predict_scores(self,n_examples_test):
        """Predict the scores of each unique label for each test example.
                
        Parameters
        ----------
        n_examples_test : int
            The number of test examples.

        Returns
        -------
        y_test_score : numpy array, shape (n_examples_test,n_labels)
            An array that contains the raw score of the current model
            for each (example,label)-pair.

        """
        y_test_scores = np.zeros((n_examples_test,self.n_labels))

        for classifier in self.classifiers:
            j,k = classifier.keys()

            triplets = self.oracle.from_pair_test(j,k)

            y_test_scores[(triplets == 1),:] += classifier[j]
            y_test_scores[(triplets == -1),:] += classifier[k]

        return y_test_scores

    def predict_all(self,n_examples_test):
        """Predict the labels with positive score for each test example.
        
        Parameters
        ----------
        n_examples_test : int
            The number of test examples.

        Returns
        -------
        y_test : list of (list of labels), len (n_examples_test)
            A list containing the (list of labels) of each test
            example as predicted by the model. The predicted labels
            are the one that have a positive score.

        """
        y_test_scores = self.predict_scores(n_examples_test)

        y_test = [[self.unique_labels[i] for i,score in enumerate(y_test_score) if score>0] for y_test_score in y_test_scores]

        return y_test
    
    def predict_top(self,n_examples_test):
        """Predict the most likely label for each test example.
        
        Parameters
        ----------
        n_examples_test : int
            The number of test examples.

        Returns
        -------
        y_test : list of label, len (n_examples_test)
            A list containing the most likely labels) for each test
            example as predicted by the model. The predicted label is
            the one with the highest score.

        """
        y_test_scores = self.predict_scores(n_examples_test)

        y_test = [self.unique_labels[np.argmax(y_test_score)] for y_test_score in y_test_scores]

        return y_test
    
    def _get_classifier(self):
        """Returns a weak triplet classifier that decreases the training error
        of the current model.
                
        Returns a weak triplet classifier, that is a random reference
        pair (j,k) with different labels and the labels that should be
        predicted to ensure that the training error decreases (can be
        different from the original labels of j and k). Returns None
        when the random weak triplet classifier has a weight of 0,
        that is the error could not be decreased with this reference
        pair.

        Returns
        -------
        classifier : None or dict
            Returns None when the weight of the classifier is
            0. Otherwise returns a dictionary where the keys are j and
            k, the reference examples, and the values are the weighted
            predictions.

        classifier_train_masks : None or dict
            Returns None when the weight of the classifier is
            0. Otherwise returns a dictionary where the keys are j and
            k, the reference examples, and the values are the masks
            corresponding to the triplets predictions on the training
            set.

        Notes
        -----
        The dictionary classifier_train_masks is returned for
        convenience and to avoid repeated calls to the oracle.

        """
        x_weights = np.sum(self.y_weights,axis=1)
        x_weights /= np.sum(x_weights)

        j, k = np.random.choice(self.n_examples,p=x_weights),np.random.choice(self.n_examples,p=x_weights)
        while np.all(self.y_binary[j,:] == self.y_binary[k,:]):
            j, k = np.random.choice(self.n_examples,p=x_weights),np.random.choice(self.n_examples,p=x_weights)

        triplets = self.oracle.from_pair(j,k)
    
        mask_j = (triplets == 1)
        mask_k = (triplets == -1)
    
        weights_positive_j = np.sum(np.multiply((self.y_binary[mask_j,:]+1)/2,self.y_weights[mask_j,:]),axis=0)
        weights_negative_j = np.sum(np.multiply((-self.y_binary[mask_j,:]+1)/2,self.y_weights[mask_j,:]),axis=0)
    
        labels_j = np.where(weights_positive_j<=weights_negative_j,-1.,1.)
    
        weight_labels_positive_j = np.sum(weights_positive_j[labels_j==1]) + np.sum(weights_negative_j[labels_j==-1])
        weight_labels_negative_j = np.sum(weights_positive_j[labels_j==-1]) + np.sum(weights_negative_j[labels_j==1])
  
        weights_positive_k = np.sum(np.multiply((self.y_binary[mask_k,:]+1)/2,self.y_weights[mask_k,:]),axis=0)
        weights_negative_k = np.sum(np.multiply((-self.y_binary[mask_k,:]+1)/2,self.y_weights[mask_k,:]),axis=0)

        labels_k = np.where(weights_positive_k<=weights_negative_k,-1.,1.)
        
        weight_labels_positive_k = np.sum(weights_positive_k[labels_k==1]) + np.sum(weights_negative_k[labels_k==-1])
        weight_labels_negative_k = np.sum(weights_positive_k[labels_k==-1]) + np.sum(weights_negative_k[labels_k==1])
    
        alpha = 0.5*np.log(np.divide(weight_labels_positive_j+weight_labels_positive_k+1/self.n_examples,weight_labels_negative_j+weight_labels_negative_k+1/self.n_examples))

        if alpha == 0:
            classifier = None
            classifier_train_masks = None
        else:
            alpha_j = alpha*labels_j
            
            alpha_k = alpha*labels_k

            classifier = {j: alpha_j, k: alpha_k}
            classifier_train_masks = {j: mask_j, k: mask_k}
        
        return classifier, classifier_train_masks
        
    def _update_weights(self,classifier,classifier_train_masks):
        """Updates the (example,label)-pair weights in y_weights given a
        classifier. Returns the normalization factor.
                
        Parameters
        ----------
        classifier : dict
            A dictionary where the keys are j and k, the reference
            examples, and the values are the weighted predictions.

        classifier_train_masks : dict
            A dictionary where the keys are j and k, the reference
            examples, and the values are the masks corresponding to
            the triplets predictions on the training set.
            
        Returns
        -------
        normalization_factor : float
            The normalization factor used to get the new weights.

        """
        partial_weights = np.zeros((self.n_examples,self.n_labels))

        for key in classifier.keys():
            partial_weights[classifier_train_masks[key],:] = -np.multiply(self.y_binary[classifier_train_masks[key],:],classifier[key])
            
        self.y_weights = np.multiply(self.y_weights,np.exp(partial_weights))+1/(self.n_labels*self.n_examples*1e10)
        
        normalization_factor = np.sum(self.y_weights)

        self.y_weights /= normalization_factor

        return normalization_factor
