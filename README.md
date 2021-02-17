### S21_CSMA.213 assignment04

For this week's assignment, you'll work with Machine Learning libraries and functions available in `sklearn` module. As a first step, you will create some sample 2D data points and plot them.  Then, you'll train a classifier (model) on the data and observe the accuracy of your results.

**Q1.** [1 points] Matplotlib contains useful tools for creating sample datasets such as the `make_blobs` function.  The code below creates a datasets of 100 items with 2 features each and organized into 2 clusters (or centers).  Print the resulting arrays contained in X and y variables. Then, use matplotlib scatter function to plot these points with all the features contained in X mapped to horizontal and vertical axes (first two parameters of `plt.scatter`) and y mapped to ‘c’ (third parameter for color specified with `c=y`)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
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

**Q6.** [1 point] Given a roll of two fair dice, what is the probability of getting a number that is even **and** greater than 3?  Start by definining the sets `A` and `B` for each sequence of numbers that correspond to desired outcome.  Next, define probabilities `P(A)` and `P(B)` based on the fact that each dice has 6 sides.  Finally, solve for probability of `A` given that `B` occurred `P(A|B) =  { A ∩ B }  / { B }`

```python
P(A) = ?
P(B) = ?
P(A|B) = { A ∩ B }  / { B } = ?
```

**Q7.** [1 point] Bayes Theorem for calculating conditional probability of some event `A` (`P(A)`) given probability of event `B` (`P(B)`) can be written as `P(A|B) = P(B|A) * P(A) / P(B)`.  Use this formula to calculate the conditional probability for the question below.

Let's say rainy days start off cloudy 50% of the time and about 20% of all days start cloudy.  If this month tends to be rainy 10% of the time, what is the chance of rain during any day this month?

Start by defining probablilities for rain `P(Rain)`, clouds `P(Cloudy)`, and probability of it being cloudy given that rain happens `P(Cloudy|Rain)`.  Then solve for `P(Rain|Cloudy)`.