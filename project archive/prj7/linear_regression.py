'''linear_regression.py
Subclass of Analysis that performs linear regression on data
Giang Pham
CS 252 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt

import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean squared error (MSE). float. Measure of quality of fit
        self.mse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var, method='scipy', p=1):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var` using the method `method`.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's linregress function.
            'normal': Use normal equations.
            'qr': Use QR factorization

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression using the appropriate method.
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''

        n = len(ind_vars)
        self.A = self.data.select_data(ind_vars)
        self.y = self.data.select_data([dep_var])

        if method == "normal":
            c = self.linear_regression_normal(A=self.A, y=self.y)
        elif method == "qr":
            c = self.linear_regression_qr(A=self.A, y=self.y)
        else:
            c = self.linear_regression_scipy(A=self.A, y=self.y)

        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.intercept = c[0][0]

        if c.shape[0] > 1: #if there are more than 1 ind var
            self.slope = c[1:, :].reshape(n, 1)
        else:
            self.slope = c.reshape(n, 1)

        y_pred = self.predict()
        self.R2 = self.r_squared(y_pred=y_pred)
        self.residuals = self.compute_residuals(y_pred=y_pred)
        self.mse = self.compute_mse()

    def linear_regression_scipy(self, A, y):
        '''Performs a linear regression using scipy's built-in least squares solver (scipy.linalg.lstsq).
        Solves the equation y = Ac for the coefficient vector c.

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var PLUS the intercept term
        '''

        A_homog = np.hstack((np.ones([A.shape[0], 1]), A))
        c, _, _, _ = sl.lstsq(a=A_homog, b=y)
        return c

    def linear_regression_normal(self, A, y):
        '''Performs a linear regression using the normal equations.
        Solves the equation y = Ac for the coefficient vector c.

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var AND the intercept term
        '''
        A_homog = np.hstack((np.ones([A.shape[0], 1]), A))
        c = (np.linalg.inv(A_homog.T@A_homog)@A_homog.T)@y
        return c

    def linear_regression_qr(self, A, y):
        '''Performs a linear regression using the QR decomposition

        (Week 2)

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: You should not compute any matrix inverses! Check out scipy.linalg.solve_triangular
        to backsubsitute to solve for the regression coefficients `c`.
        '''
        A_homog = np.hstack((A, np.ones([A.shape[0], 1])))
        Q, R = self.qr_decomposition(A_homog)
        c = sl.solve_triangular(R, Q.T@y)
        return c

    def qr_decomposition(self, A):
        '''Performs a QR decomposition on the matrix A. Make column vectors orthogonal relative
        to each other. Uses the Gram–Schmidt algorithm

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars+1).
            Data matrix for independent variables.

        Returns:
        -----------
        Q: ndarray. shape=(num_data_samps, num_ind_vars+1)
            Orthonormal matrix (columns are orthogonal unit vectors — i.e. length = 1)
        R: ndarray. shape=(num_ind_vars+1, num_ind_vars+1)
            Upper triangular matrix

        TODO:
        - Q is found by the Gram–Schmidt orthogonalizing algorithm.
        Summary: Step thru columns of A left-to-right. You are making each newly visited column
        orthogonal to all the previous ones. You do this by projecting the current column onto each
        of the previous ones and subtracting each projection from the current column.
            - NOTE: Very important: Make sure that you make a COPY of your current column before
            subtracting (otherwise you might modify data in A!).
        Normalize each current column after orthogonalizing.
        - R is found by equation summarized in notebook
        '''
        Q = np.copy(A)
        # loop through each column of A starting from col 1
        for i in range(A.shape[1]):
            for j in range(i):  # loop through each column of Q from 1 until i-1
                Q[:, i] -= np.dot(Q[:, j], A[:, i]) * \
                    Q[:, j]  # change column i of Q
            Q[:, i] = Q[:, i]/(np.linalg.norm(Q[:, i]) + 1e-10)

        R = Q.T @ A
        return Q, R

    def predict(self, X=None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''
        if X is None:
            if self.p > 1:
                y_pred = self.make_polynomial_matrix(self.A, self.p) @ self.slope + self.intercept
            else: 
                y_pred = self.A @ self.slope + self.intercept

        else: 
            if self.p > 1:
                y_pred = self.make_polynomial_matrix(X, self.p) @ self.slope + self.intercept
            else:
                y_pred = X @ self.slope + self.intercept

        return y_pred

    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        n = self.A.shape[0]
        y_mean = np.sum(self.y)/n
        model_pred_error = np.sum(np.square(self.y - y_pred))
        model_mean_error = np.sum(np.square(self.y - y_mean))
        r2 = 1 - model_pred_error/model_mean_error
        return r2

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''
        return self.y - y_pred

    def compute_mse(self):
        '''Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error

        Hint: Make use of self.compute_residuals
        '''
        n = self.A.shape[0]
        y_pred = self.predict()
        n = self.A.shape[0]
        mse = np.sum(np.square(self.compute_residuals(y_pred=y_pred)))/n
        return mse

    def scatter(self, ind_var, dep_var, title):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        x_vars, _ = super().scatter(ind_var=ind_var, dep_var=dep_var, title=title)
        x_line = np.linspace(start=np.min(
            x_vars), stop=np.max(x_vars), num=100)
        if self.p == 1: 
            y_line = (x_line * self.slope + self.intercept).flatten()
        else: 
            poly = self.make_polynomial_matrix(x_line.reshape([x_line.shape[0],1]), self.p)
            y_line = (self.intercept + np.sum(poly @ self.slope, axis = 1)).squeeze()
        plt.plot(x_line, y_line, "purple")
        plt.title(title)

    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''
        size = len(data_vars)
        fig, axes = super().pair_plot(data_vars=data_vars, fig_sz=fig_sz)
        for i in range(size):
            for j in range(size):
                x_vars = self.data.select_data(
                    headers=[data_vars[j]]).flatten()
                y_var = self.data.select_data(headers=[data_vars[i]]).flatten()
                if hists_on_diag == True and i == j:
                    numVars = len(data_vars)
                    axes[i][j].remove()
                    axes[i][j] = fig.add_subplot(
                        numVars, numVars, i*numVars+j+1)
                    if j < numVars-1:
                        axes[i][j].set_xticks([])
                    else:
                        axes[i][j].set_xlabel(data_vars[i])
                    if i > 0:
                        axes[i][j].set_yticks([])
                    else:
                        axes[i][j].set_ylabel(data_vars[i])
                    axes[i][j].hist(x_vars)
                # plotting the linear regression line
                self.linear_regression([data_vars[j]], data_vars[i])
                x_line = np.linspace(start=np.min(
                    x_vars), stop=np.max(x_vars), num=100)
                y_line = (x_line * self.slope + self.intercept).flatten()
                axes[i][j].plot(x_line, y_line, "purple")
                axes[i][j].set_title(f"R2: {self.R2:.3f}")

        fig.tight_layout()
        
    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        will take care of that.
        '''
        n = A.shape[0]
        poly = np.ones(shape = (n,p))
        for i in range(1,p+1): 
            poly[:,i-1] = np.power(A,i).squeeze()
        return poly        

    def poly_regression(self, ind_var, dep_var, p):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1s).

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        self.ind_vars = ind_var
        self.dep_var = dep_var

        self.A = self.data.select_data([self.ind_vars])
        self.y = self.data.select_data([self.dep_var])
        self.p = p
        poly_matrix = self.make_polynomial_matrix(self.A, p)
        poly_homog = np.hstack((np.ones(shape = [poly_matrix.shape[0],1]), poly_matrix ))

        c= self.linear_regression_qr(poly_matrix, self.y)
        #make y_pred 
        self.intercept = c[0][0]

        self.slope = c[1:,:]
        y_pred = self.predict()
        self.R2 = self.r_squared(y_pred=y_pred)
        self.residuals = self.compute_residuals(y_pred=y_pred)
        self.mse = self.compute_mse()

    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope
        

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return self.intercept

    def get_r_squared(self):
        return self.R2
    
    def get_mse(self):
        return self.mse

    def get_A(self):
        return self.A

    def get_y(self):
        return self.y

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)
        
        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor.
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.y = self.data.select_data([self.dep_var])
        self.A = self.data.select_data(self.ind_vars)
        self.slope = slope
        self.intercept = intercept
        self.p = p
        self.mse = self.compute_mse()
        self.R2 = self.r_squared(self.predict())

