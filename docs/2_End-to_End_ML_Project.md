# Chapter 2. End-to-End Machine Learning Project

Just a collection of notes while following the book example in chapter 2.

- About performance measures:
  - $l_1$ norm or $\|\cdot\|_1$ or **Mean absolute error** $MAE(\bm{X},h) = \frac{1}{m}\sum_{i=1}^m |h(\bm{x}^{(i)})-y^{(i)}|$
  - $l_2$ norm or $\|\cdot\|_2$ or **Root mean square error**: $RMSE(\bm{X},h) = \sqrt{\frac{1}{m}\sum_{i=1}^m (h(\bm{x}^{(i)})-y^{(i)})^2}$
  - $l_k$ norm $\|\bm{v}\|_k = (|v_1|^k+|v_2|^k+...+|v_n|^k)^{1/k}$
- As $k$ increases impact of outliers increases. For bell-shaped data, RMSE is just fine!
- Put away the test data without looking at it, otherwise you might select a model given the test data shape on accident (**data scooping**)
- When running the code multiple times, your model will eventually see more and more of the test data when randomly splitting data into train and test, i.e. use a seed for the random split. This won't work if you change the dataset as a whole. Even better compute a hast of each instance's identifier (see book)
- Different ways to split data:
    1. Randomly for large datasets: `from sklearn.model_selection import train_test_split`
    2. For small datasets a random split might produce significant **sampling bias**
    Similarly, when people are asked in a phone survey, the sample should correspond to the population which is called **stratified sampling**. Population is divided into homogeneous subgroups called **strata**, and the right number of instances are sampled from each stratum to guarantee that the test set is representative. Similarly, one could divide numerical features into categories (strata) to have a sufficient number of instances in the dataset and use the histogram across those strata for train test split:
    Use `from sklearn.model_selection import StratifiedShuffleSplit` or `train_test_split` with the `stratify` argument. 
