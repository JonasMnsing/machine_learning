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
- Before exploration copy train data as we often run manipulations on them. Take a subset if train set is to large
- For missing data, either drop or set NaN to median, mean, etc (**inputation**). For last option use `sklearn.impute.SimpleImputer` or `sklearn.impute.KNNImputer` or `sklearn.impute.IterativeImputer`
- For categorical data, use `sklearn.preprocessing.OrginalEncoder` class to transform into numberical values. Most ML algorithms assume those values to be more similar as they are not as distant, which might me wrong if categories are not ordered. 
- Solution, use *one-hot encoding*: For each category set up a column with 0 and 1 for true or false (`sklearn.preprocessing.OneHotEncoder`).
- Most import preprocessing step: Scaling as features are on different scales. Use `MinMaxScaler` or `StandardScaler` from `sklearn.preprocessing. The latter is less affected by outliers!
- Both scalers don't like heavy tail / non-exponential tails! Use log-transforms or sqrt / power between 0 and 1 of those features. **Bucketizing** features: Chopping each feature into equal-sized ranges and replacing each feature value with the index of the bucket (e.g. bucket based on percentile)
- If features are multi-model (many peaks) one could make a new feature for each values similarity to a specific mode using radial basis functions e.g. `sklearn.metrics.pairwise.rbf_kernel`
- **Remark:** Most Scikit-Learn tranformers haven an `inverse_transform()`
- There are also *Custom Transformers*
- We can use `pipelines` to summarize and sequentially execute transformers, scaling etc or just use your own functions
- Use **k fold cross validation** for train test splitting
- Use Grid Search for small combinations of hyper paramters and **Randomized Search** for large combinations
- Use the `joblib' library to save the best model and transfer the file to your prodcution environment e.g. running on a server where users can use an app to pass some data and effectively use the predict method. You can also upload the model to Google Cloud Storage
- Automate e.g. weekly training, checking of input data quality etc