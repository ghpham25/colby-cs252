'''analysis.py
Run statistical analyses and plot Numpy ndarray data
Giang Pham
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new
        Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''

        selected_data = self.data.select_data(headers, rows)
        mins = np.amin(selected_data, axis=0)
        return mins

    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices
            if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        selected_data = self.data.select_data(headers, rows)
        maxs = np.amax(selected_data, axis=0)
        return maxs

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        rangee = [self.min(headers, rows), self.max(headers, rows)]
        return np.array(rangee)

    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices
            if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: There should be no loops in this method!
        '''
        selected_data = self.data.select_data(headers, rows)
        sums = np.sum(selected_data, axis=0)
        if len(rows) > 0:
            mean = sums/len(rows)
        else:
            mean = sums/self.data.get_num_samples()
        return mean

    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices
            if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE: You CANNOT use np.var or np.mean here!
        NOTE: There should be no loops in this method!
        '''
        selected_data = self.data.select_data(headers, rows)
        meanlist = self.mean(headers, rows)
        sigma = np.sum((selected_data - meanlist)**2, axis=0)
        if len(rows) > 0:
            var = sigma*(1/(len(rows) - 1))
        else:
            var = sigma*(1/(self.data.get_num_samples() - 1))
        return var

    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE: You CANNOT use np.var, np.std, or np.mean here!
        NOTE: There should be no loops in this method!
        '''
        return self.var(headers, rows)**(1/2)

    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()

    def scatter(self, ind_var, dep_var, title):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        x_vars = self.data.select_data(headers=[ind_var]).flatten()
        y_vars = self.data.select_data(headers=[dep_var]).flatten()
        plt.plot(x_vars, y_vars, "o")
        plt.title(title)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        return x_vars, y_vars

    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column should be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey keyword arguments of plt.subplots.
        Because variables may have different ranges, pair plot columns usually share the same
        x axis and rows usually share the same y axis.
        '''

        size = len(data_vars)
        fig, axs = plt.subplots(
            size, size, figsize=fig_sz)
        fig.suptitle(title)
        for i in range(size):
            for j in range(size):
                if i == size - 1:
                    axs[i, j].set_xlabel(data_vars[j])
                else:
                    axs[i, j].set_xticklabels([])
                if j == 0:
                    axs[i, j].set_ylabel(data_vars[i])
                else:
                    axs[i, j].set_yticklabels([])

                x_var = self.data.select_data(
                    headers=[data_vars[j]]).flatten()
                y_var = self.data.select_data(
                    headers=[data_vars[i]]).flatten()
                axs[i, j].scatter(x_var, y_var)

        return fig, axs

    def scatter_with_linreg(self, ind_var, dep_var, title=''):
        '''
        Make a scatterplot that has a linear regression line 

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        None         
        '''
        x_vars, y_vars = self.scatter(ind_var, dep_var, title)
        c = np.polyfit(x_vars, y_vars, deg=1)
        y_vars_fit = c[0]*x_vars + c[1]
        linreg = plt.plot(x_vars, y_vars_fit, '-')

        return x_vars, y_vars_fit

    def animate(self, ind_var, dep_var, title=''):
        '''
        Animate a pair plot showing a data point at a time and generate a linear regression for each data point being plotted

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis

        Returns:
        -----------
        None 
        '''

        fig, ax = plt.subplots()
        data_stream = self.select_data(
            headers=["bill_length_mm", "bill_depth_mm"])
        graph = ax.scatter([], [])

        linreg, = ax.plot([], [])

        ax.axis([np.min(x_vars) - 1, np.max(x_vars) + 1,
                np.min(y_vars) - 1, np.max(y_vars) + 1])

        def update(frame):
            a, b = np.polyfit(
                data_stream[:frame, 0], data_stream[:frame, 1], deg=1)
            y_vars_fit = a*data_stream[:frame, 0] + b
            linreg.set_data(data_stream[:frame, 0], y_vars_fit)
            graph.set_offsets(data_stream[:frame, :])
            return graph, linreg

        ani = FuncAnimation(fig, func=update, frames=np.arange(
            2, len(x_vars)), interval=100, blit=False)
