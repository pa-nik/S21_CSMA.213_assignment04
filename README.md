### S21_CSMA.213 assignment04

For this week's assignment, you'll work with Machine Learning libraries and functions available in `sklearn` module. As a first step, you will create some sample 2D data points and plot them.  Then, you'll train a classifier (model) on the data and observe the accuracy of your results.

**Q1.** [1 points] Matplotlib contains useful tools for creating sample datasets such as the `make_blobs` function.  The code below creates a datasets of 100 items with 2 features each and organized into 2 clusters (or centers).  Print the resulting arrays contained in X and y variables. Then, use matplotlib scatter function to plot these points with all the features contained in X mapped to horizontal and vertical axes (first two parameters of `plt.scatter`) and y mapped to ‘c’ (third parameter for color specified with `c=y`)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# create a scatter plot of X and y data as specified in the instructions
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

**Q4.** [1 point] Use `make_blobs` function to create a set of 5 new points with 2 centers and containing 2 features each.  Classify the new points using the Logistic Regression model trained in Q3 and print out the prediction results.

---

The second half of the assignment deals with conditional probability and Bayes Theorem, building up to the application of Naive Bayes classifier included in `sklearn` Machine Learning library.

**Q5.** [1 point] Let's assume you rolled two dice 3 times, producing the following sequence of results that we can define as sets `A` and `B`.  What sequence of numbers would be in set `C` that is the union of `A` and `B` (`C = A ∪ B`)?  If set `D` is the intersection of `A` and `B` (`D = A ∩ B`), what is the result?

```python
A = { 1, 4, 3 }
B = { 2, 1, 5 }
C = { A ∪ B } = ?
D = { A ∩ B } = ?
```

**Q6.** [1 point] Given a roll of two fair dice, what is the probability of getting a number that is even **and** greater than 3?  

Start by definining the sets `A` and `B` for each sequence of numbers that correspond to desired outcome.  Next, define probabilities `P(A)` and `P(B)` based on the fact that each dice has 6 sides.  Finally, solve for probability of `A` given that `B` occurred `P(A|B) =  { A ∩ B }  / { B }`

```python
P(A) = ?
P(B) = ?
P(A|B) = { A ∩ B }  / { B } = ?
```

**Q7.** [1 point] Bayes Theorem for calculating conditional probability of some event `A` (`P(A)`) given probability of event `B` (`P(B)`) can be written as `P(A|B) = P(B|A) * P(A) / P(B)`.  Use this formula to calculate the conditional probability for the question below.

Let's say rainy days start off cloudy 50% of the time and about 20% of all days start cloudy.  If it tends to be rainy 10% of the time, what is the probability of rain on a cloudy day?

Start by defining probablilities for rain `P(Rain)`, clouds `P(Cloudy)`, and probability of it being cloudy on rainy day `P(Cloudy|Rain)`.  Then solve for `P(Rain|Cloudy)`.

```python
P(Rain) = ?
P(Cloudy) = ?
P(Cloudy|Rain) = ?
P(Rain|Cloudy) = ?
```

**Q8.** [2 points] Finish the Python program template below to evaluate the accuracy of the Gaussian Naive Bayes model trained on the Iris flower dataset.  The template provides the code to load the Iris data and split it into training and testing arrays.  

You will need to insert the code for creating and training the model, then use it to predict new `y_pred` values based on `y_test` data.  The last line of code will calculate and print out the accuracy of the model's predictions as a percentage.

```python
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn import metrics

# load iris dataset:
iris = load_iris() 
# assign iris flower data to X and flower target (class) to y arrays:
X = iris.data 
y = iris.target 
# split X and y into training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# insert code to create GaussianNB model, fit training data to it and use it to 
# generate predictions (y_pred array) based on y_test values

# compare actual response values (y_test) with predicted response values (y_pred):
print("Gaussian Naive Bayes model accuracy:", metrics.accuracy_score(y_test, y_pred)*100 , "%")
```