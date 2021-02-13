### S21_CSMA.213 assignment04

For this week's assignment, you'll work with Machine Learning libraries and functions available in `sklearn` module. As a first step, you will create some sample 2D data points and plot them.  Then, you'll train a classifier (model) on the data and observe the accuracy of your results.

**Q1.** [1 points] Matplotlib contains useful tools for creating sample datasets such as the `make_blobs` function.  The code below creates a datasets of 100 items with 2 features each and organized into 2 clusters (or centers).  Print the resulting arrays contained in X and y variables. Then, use matplotlib scatter function to plot these points with all the features contained in X mapped to horizontal and vertical axes (first two parameters of `plt.scatter`) and y mapped to ‘c’ (third parameter for color specified with `c=y`)

```python
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
```

**Q2.** [1 point] The scatter plot above should produce 2 groups of points with separate colors assigned by the default 'viridis' mapping.  Create a new list `label_color_map` containing 2 color strings as shown below, then fill the list `label_colors` with 100 colors assigned blue or red according to whether corresponding value in array `y` is 0 or 1.  Plot the graph above again, this time assigning color variable `c=label_colors`.  You can use `append()` command to add correct colors to `label_colors` list. The result should be the same scatter plot as above with all points in either red or blue.   

```python
label_color_map = ['red', 'blue']
label_colors = []  # empty list of colors
# write code to fill in label_colors with colors corresponding to values in y array
```

**Q3.** [2 points] Create a Logistic Regression model and train it on data contained in `X` and `y`.  Using the list `Xnew` containing one 2-dimensional coordinate specified below, run prediction to classify the label that the new coordinate belongs to and print out the results.

```python
from sklearn.linear_model import LogisticRegression
Xnew = [[-0.75, 2.0]]
# write code to create a logistic regression model and train it on X and y 
# then use the model to predict the label of Xnew coordinate
```
