'''knn.py
K-Nearest Neighbors algorithm for classification
Giang Pham
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from palettable import cartocolors


class KNN:
    '''K-Nearest Neighbors supervised learning algorithm'''

    def __init__(self, num_classes):
        '''KNN constructor

        TODO:
        - Add instance variable for `num_classes`
        '''
        # exemplars: ndarray. shape=(num_train_samps, num_features).
        #   Memorized training examples
        self.exemplars = None
        # classes: ndarray. shape=(num_train_samps,).
        #   Classes of memorized training examples
        self.classes = None
        # number of classes
        self.num_classes = num_classes

    def train(self, data, y):
        '''Train the KNN classifier on the data `data`, where training samples have corresponding
        class labels in `y`.

        Parameters:
        -----------
        data: ndarray. shape=(num_train_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_train_samps,). Corresponding class of each data sample.

        TODO:
        - Set the `exemplars` and `classes` instance variables such that the classifier memorizes
        the training data.
        '''
        self.exemplars = data
        self.classes = y

    def predict(self, data, k):
        '''Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network.
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.

        TODO:
        - Compute the distance from each test sample to all the training exemplars.
        - Among the closest `k` training exemplars to each test sample, count up how many belong
        to which class.
        - The predicted class of the test sample is the majority vote.
        '''
        # compute the distance from each sample in data [num_test_samp, num_feature] to all
        # training exemplars -> an array for each sample in test -> shape [num_test_samp, num_train samp]
        samples_to_exemplars_dist = []
        for test_samp in data:
            cursamp_to_exemplars_dist = np.sqrt(
                np.sum(np.square(self.exemplars - test_samp), axis=1))
            samples_to_exemplars_dist.append(cursamp_to_exemplars_dist)
        samples_to_exemplars_dist = np.array(samples_to_exemplars_dist)

        # pick the closest k exemplars from each list
        knn_classes = []
        for dist in samples_to_exemplars_dist:
            knn_indices = dist.argsort()[:k]
            knn_classes.append(self.classes[knn_indices])
        knn_classes = np.array(knn_classes).astype(int)

        # vote and find the assign the class to test the same as class of the voted closest exemplar
        y_test_pred = []
        for samp in knn_classes:
            voted_class = np.argmax(np.bincount(samp))
            y_test_pred.append(voted_class)

        # return predicted class of each test data
        return np.array(y_test_pred)

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
        y_minus = y-y_pred
        acc = np.count_nonzero(y_minus == 0) / y.shape[0]
        return acc

    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Recall: the rows represent the "actual" ground truth labels, the columns represent the
        predicted labels.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        confusion = np.zeros((self.num_classes, self.num_classes))
        #loop through y and ypred and fill in the confusion matrix
        for i in range(len(y)): 
            j = y[i]
            k = y_pred[i]
            confusion[int(j),int(k)] += 1
        return confusion

    def plot_predictions(self, k, n_sample_pts):
        '''Paints the data space in colors corresponding to which class the classifier would
         hypothetically assign to data samples appearing in each region.

        Parameters:
        -----------
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.
        n_sample_pts: int.
            How many points to divide up the input data space into along the x and y axes to plug
            into KNN at which we are determining the predicted class. Think of this as regularly
            spaced 2D "fake data" that we generate and plug into KNN and get predictions at.

        TODO:
        - Pick a discrete/qualitative color scheme. We suggest, like in the clustering project, to
        use a ColorBrewer color palette. List of possible ones here:
        https://github.com/CartoDB/CartoColor/wiki/CARTOColor-Scheme-Names
            - An example: cartocolors.qualitative.Safe_4.mpl_colors
            - The 4 stands for the number of colors in the palette. For simplicity, you can assume
            that we're hard coding this at 4 for 4 classes.
        - Each ColorBrewer palette is a Python list. Wrap this in a `ListedColormap` object so that
        matplotlib can parse it (already imported above).
        - Make an ndarray of length `n_sample_pts` of regularly spaced points between -40 and +40.
        - Call `np.meshgrid` on your sampling vector to get the x and y coordinates of your 2D
        "fake data" sample points in the square region from [-40, 40] to [40, 40].
            - Example: x, y = np.meshgrid(samp_vec, samp_vec)
        - Combine your `x` and `y` sample coordinates into a single ndarray and reshape it so that
        you can plug it in as your `data` in self.predict.
            - Shape of `x` should be (n_sample_pts, n_sample_pts). You want to make your input to
            self.predict of shape=(n_sample_pts*n_sample_pts, 2).
        - Reshape the predicted classes (`y_pred`) in a square grid format for plotting in 2D.
        shape=(n_sample_pts, n_sample_pts).
        - Use the `plt.pcolormesh` function to create your plot. Use the `cmap` optional parameter
        to specify your discrete ColorBrewer color palette.
        - Add a colorbar to your plot
        '''

        # initialize x and y vectors
        color = cartocolors.qualitative.Bold_4.mpl_colors
        samp_vec = np.linspace(-40, 40, n_sample_pts)
        x, y = np.meshgrid(samp_vec, samp_vec)
        data = np.column_stack((x.flatten(), y.flatten()))

        # plot data
        y_pred = self.predict(data, k).reshape(x.shape)
        plt.pcolormesh(x, y, y_pred, cmap=ListedColormap(color))
        plt.colorbar()
