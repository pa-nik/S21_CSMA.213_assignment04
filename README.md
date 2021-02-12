### S21_CSMA.213 assignment04

For this week's assignment, you'll work with Machine Learning libraries and functions available in `sklearn` module. As a first step, you will create some sample 2D data points and plot them.  Then, you'll train a classifier (model) on the data and observe the accuracy of your results.

** Q1. ** [1 points] Matplotlib contains useful tools for creating sample datasets such as the `make_blobs` function.  The code below creates a datasets of 100 items with 2 features each and organized into 2 clusters (or centers).  Print the resulting arrays contained in X and y variables. Then, use matplotlib scatter function to plot these points with all the features contained in X mapped to horizontal and vertical axes (first two parameters of `plt.scatter`) and y mapped to ‘c’ (third parameter for color specified with `c=y`)

`from sklearn.datasets.samples_generator import make_blobs`
`X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)`

