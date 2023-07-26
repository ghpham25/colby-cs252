'''rbf_net.py
Radial Basis Function Neural Network
Giang Pham
CS 252: Mathematical Data Analysis Visualization, Spring 2021
'''
import numpy as np
import kmeans
import linear_regression as lr
import scipy.linalg as sl


class RBF_Net:
    def __init__(self, num_hidden_units, num_classes):
        '''RBF network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset

        TODO:
        - Define number of hidden units as an instance variable called `k` (as in k clusters)
            (You can think of each hidden unit as being positioned at a cluster center)
        - Define number of classes (number of output units in network) as an instance variable
        '''
        # k: int. Number of hidden units
        self.k = num_hidden_units

        # num_classes: int. Number of classes in the dataset
        self.num_classes = num_classes

        # prototypes: Hidden unit prototypes (i.e. center)
        #   shape=(num_hidden_units, num_features)
        self.prototypes = None

        # sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        # are similar to the unit's prototype (i.e. center).
        #   shape=(num_hidden_units,)
        #   Larger sigma -> hidden unit becomes active to dissimilar inputs
        #   Smaller sigma -> hidden unit only becomes active to similar inputs
        self.sigmas = None

        # wts: Weights connecting hidden and output layer neurons.
        #   shape=(num_hidden_units+1, num_classes)
        #   The reason for the +1 is to account for the bias (a hidden unit whose activation is always
        #   set to 1).
        self.wts = None

    def get_prototypes(self):
        '''Returns the hidden layer prototypes (centers)

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, num_features).
        '''
        return self.prototypes

    def get_num_hidden_units(self):
        '''Returns the number of hidden layer prototypes (centers/"hidden units").

        Returns:
        -----------
        int. Number of hidden units.
        '''
        return self.k

    def get_num_output_units(self):
        '''Returns the number of output layer units.

        Returns:
        -----------
        int. Number of output units
        '''
        return self.num_classes

    def get_weights(self):
        return self.wts
    
    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj):
        '''Compute the average distance between each cluster center and data points that are
        assigned to it.

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        centroids: ndarray. shape=(k, num_features). Centroids returned from K-means.
        cluster_assignments: ndarray. shape=(num_samps,). Data sample-to-cluster-number assignment from K-means.
        kmeans_obj: KMeans. Object created when performing K-means.

        Returns:
        -----------
        ndarray. shape=(k,). Average distance within each of the `k` clusters.

        Hint: A certain method in `kmeans_obj` could be very helpful here!
        '''
        avg_dists = np.zeros((centroids.shape[0])) # (4extension)changed self. k -> centroids.shape[0]
        # loop through each centroid and extract all datas that are assigned to it
        for i in range(centroids.shape[0]): #(4extension)changed self. k -> centroids.shape[0]
            sum_dists = 0
            centroid = centroids[i]
            assigned_data_pts = data[cluster_assignments == i]
            # loop through each sample in the assigned data and calculate the sum distances
            for j in range(assigned_data_pts.shape[0]):
                samp = assigned_data_pts[j]
                dist = kmeans_obj.dist_pt_to_pt(samp, centroid)
                sum_dists += dist
            avg_dists[i] = sum_dists/assigned_data_pts.shape[0]
        return avg_dists

    def initialize(self, data):
        '''Initialize hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        TODO:
        - Determine `self.prototypes` (see constructor for shape). Prototypes are the centroids
        returned by K-means. It is recommended to use the 'batch' version of K-means to reduce the
        chance of getting poor initial centroids.
            - To increase the chance that you pick good centroids, set the parameter controlling the
            number of iterations > 1 (e.g. 5)
        - Determine self.sigmas as the average distance between each cluster center and data points
        that are assigned to it. Hint: You implemented a method to do this!
        '''
        kmeans_obj = kmeans.KMeans(data)
        kmeans_obj.cluster_batch(k=self.k, n_iter=5, init_method="random")
        self.prototypes = kmeans_obj.get_centroids()

        self.sigmas = self.avg_cluster_dist(
            data, self.prototypes, kmeans_obj.get_data_centroid_labels(), kmeans_obj)

    def initialize_separate(self, data, labels):
        '''Initialize SEPARATELY hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster
        '''

        prototypes = []
        sigmas = []

        clusters_made = 0 
        for i in range(self.num_classes): 
            #run kmeans on the samples belonging the class
            class_data = data[labels == i]
            kmeans_obj = kmeans.KMeans(class_data)
            
            # determine the number of clusters for the class using the proportion of the class in the whole dataset
            class_proportion = list(labels).count(i)/labels.shape[0] #determine the proportion of each class in the whole dataset 
            class_num_cluster = np.round(self.k * class_proportion).astype(int)
            
            #when reach the last class, just use the remaining unused k 
            if i == self.num_classes - 1: 
                class_num_cluster = self.k - clusters_made
            
            kmeans_obj.cluster_batch(k=class_num_cluster, n_iter=5, init_method="random")
            clusters_made += class_num_cluster

            class_prototypes = kmeans_obj.get_centroids()
            class_sigmas = self.avg_cluster_dist(
                class_data, class_prototypes, kmeans_obj.get_data_centroid_labels(), kmeans_obj)

            prototypes.extend(class_prototypes)
            sigmas.extend(class_sigmas)

        self.prototypes = np.array(prototypes)
        self.sigmas = np.array(sigmas)

    def linear_regression(self, A, y):
        '''Performs linear regression
        CS251: Adapt your SciPy lstsq code from the linear regression project.
        CS252: Adapt your QR-based linear regression solver

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_features+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: Remember to handle the intercept ("homogenous coordinate")
        '''
        linreg_obj = lr.LinearRegression(A)
        c = linreg_obj.linear_regression_qr(A,y)
        return c

    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation
        '''
        kmean_obj = kmeans.KMeans()
        num_samps = data.shape[0]
        hidden_act = np.zeros((num_samps, self.k))
        for i in range(num_samps):
            samp = data[i]
            dist_to_each_cntrd = np.square(
                kmean_obj.dist_pt_to_centroids(samp, self.prototypes))
            i_activation = np.exp(-dist_to_each_cntrd /
                                  (2*np.square(self.sigmas) + 1e-8))
            hidden_act[i] = i_activation
        return hidden_act

    def output_act(self, hidden_acts):
        '''Compute the activation of the output layer units

        Parameters:
        -----------
        hidden_acts: ndarray. shape=(num_samps, k).
            Activation of the hidden units to each of the data samples.
            Does NOT include the bias unit activation.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_units).
            Activation of each unit in the output layer to each of the data samples.

        NOTE:
        - Assumes that learning has already taken place
        - Can be done without any for loops.
        - Don't forget about the bias unit!
        '''
        bias_col = np.ones((hidden_acts.shape[0], 1))
        biased_hidden_acts = np.hstack((hidden_acts, bias_col))
        output_act = biased_hidden_acts @ self.wts
        return output_act

    def train(self, data, y, method = "common"):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the correct classes of each training sample. To solve for the weights going FROM all of the
        hidden units TO output unit c, recode the class vector `y` to 1s and 0s:
            1 if the class of a data sample in `y` is c
            0 if the class of a data sample in `y` is not c

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.
        '''
        # initialize the network
        self.initialize(data)

        if (method == "separate"): 
            self.initialize_separate(data, y)

        # set up A (H matrix (hidden layer matrix) and the weights matrix
        hidden_act = self.hidden_act(data)

        weights_matrix = np.zeros((self.k+1, self.num_classes))

        num_sample = data.shape[0]
        # run linear regression num_class (c) times
        for i in range(self.num_classes):
            # make y vector
            binary_vec_y = np.zeros((num_sample))
            binary_vec_y[y == i] = 1

            # run linear regression and build the weights matrix
            weight_col = self.linear_regression(hidden_act, binary_vec_y)
            weights_matrix[:, i] = weight_col

        self.wts = weights_matrix
        # print(self.wts)

    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each data sample.

        TODO:
        - Pass the data thru the network (input layer -> hidden layer -> output layer).
        - For each data sample, the assigned class is the index of the output unit that produced the
        largest activation.
        '''

        hidden_acts = self.hidden_act(data)
        output_act = self.output_act(hidden_acts)

        num_samples = data.shape[0]
        y_pred = np.zeros((num_samples))

        # print(output_act)

        # loop through each sample of data
        # get the index of the output unit with largest activation for each data sample
        for i in range(num_samples):
            y_pred[i] = np.argmax(output_act[i])

        return y_pred

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        difference = y-y_pred
        acc = np.count_nonzero(difference == 0) / y.shape[0]
        return acc
        
class RBF_Reg_Net(RBF_Net):
    '''RBF Neural Network configured to perform regression
    '''

    def __init__(self, num_hidden_units, num_classes, h_sigma_gain=5):
        '''RBF regression network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset
        h_sigma_gain: float. Multiplicative gain factor applied to the hidden unit variances

        TODO:
        - Create an instance variable for the hidden unit variance gain
        '''
        super().__init__(num_hidden_units, num_classes)
        self.h_sigma_gain = h_sigma_gain

    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation

        TODO:
        - Copy-and-paste your classification network code here.
        - Modify your code to apply the hidden unit variance gain to each hidden unit variance.
        '''
        kmean_obj = kmeans.KMeans()
        num_samps = data.shape[0]
        hidden_act = np.zeros((num_samps, self.k))
        for i in range(num_samps):
            samp = data[i]
            dist_to_each_cntrd = np.square(
                kmean_obj.dist_pt_to_centroids(samp, self.prototypes))
            i_activation = np.exp(-dist_to_each_cntrd /
                                  (2* self.h_sigma_gain * np.square(self.sigmas) + 1e-8))
            hidden_act[i] = i_activation
        return hidden_act


    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the desired y output of each training sample.

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.

        TODO:
        - Copy-and-paste your classification network code here, modifying it to perform regression on
        the actual y values instead of the y values that match a particular class. Your code should be
        simpler than before.
        - You may need to squeeze the output of your linear regression method if you get shape errors.
        '''
        self.initialize(data)
        hidden_act = self.hidden_act(data)
        self.wts = self.linear_regression(hidden_act, y)

    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_neurons). Output layer neuronPredicted "y" value of
            each sample in `data`.

        TODO:
        - Copy-and-paste your classification network code here, modifying it to return the RAW
        output neuron activaion values. Your code should be simpler than before.
        '''
        hidden_acts = self.hidden_act(data)
        output_act = self.output_act(hidden_acts)
        return output_act
