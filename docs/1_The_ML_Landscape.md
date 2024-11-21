# Chapter 1. The Machine Learning Landscape

- **Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed**
- **A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E**
  
### Some Definitions

- Examples that the system uses to learn: **training set** with each example being a **training instance** or **sample**
- The part that learns is called **model**
- Performance measure needs to be defined, often called **accuracy**
- ML shines if we cannot define a set of rules / need a long list of rules in a classical code e.g. spam filter or when there isn't a known algorithm, e.g. speech recognition
- ML also helps the human to find unknown pattern in data (**data mining**) e.g. which words are actually related to spam
- ML can be retrained on fluctuating environments

### Types of Machine Learning

1. Training Supervision
    - In **supervised learning**, the training set you feed to the algorithm includes the desired solution (**labels**), e.g. classification (spam) or regression (price)
    - In **unsupervised learning** data is unlabeled, e.g. clustering to detect groups of similar humans / visualization models / dimensionality reduction (feature extraction) / anomaly detection (credit card fraud) / association rule learning (when people buy bbq sauce, and chips they also often buy steak)
    - In **semi-supervised learning** we have a combination of e.g. clustering first and then regression when a couple of labels are present
    - In **self-supervised learning** a fully labeled dataset is generated from a fully unlabeled one, e.g. algorithm that recovers mask images
    - In **Reinforcement Learning** an agent observes the environment, select and performs an action and gets a reward or penalty. It then learns by itself what is the best strategy, called policy

2. Batch vs Online Learning
    - In **batch learning**, the system is incapable of learning incrementally - must be trained using all the available data. Typically, this is performed **offline**, i.e. First the system is trained, and then it is launched / runs without learning. This leads to **data drift** as the world continues to evolve while the model has finished training. If you want to train a trained batch learning system you need to train it all over again (on new and old data as well) and then replace the old model with the new one
    - In **online learning** you train incrementally by feeding data instances sequentially (mini-batches) which helps if either the environment changes quickly (stock market) or the amount of data does not fit on one machine's memory (**out-of-core learning**). Major feature is called the **learning rate** which defines how fast the system adapts to changing data
  
3. Instance-Based vs Model-Based Learning
    - **Instance-based learning** firstly learns by heart (mails identical to spam) and then generalizes to new cases by using a similarity measure (mails similar to mails labeled as spam)
    - **Model-based learning** defines a clear model from a set of examples and uses this model to make predictions, e.g. linear functions 
  
### Main Challenges of Machine Learning

As we want to fit a model based on data. There are two things that can go wrong: "Bad model" or "bad data". 

1. Bad Data:
   - **Insufficient Quantity of Training Data**: The amount of training data is considered to be more import than the actual selected model (ref. *THE UNREASONABLE EFFECTIVENESS OF DATA*). For large datasets different models perform about evenly well. However, since most datasets are still in midrange we still need proper models
   - **Sampling noise**/**Sampling bias**: As data is too small, the dataset becomes nonrepresentative of chance of cases you want to generalize to
   - **Poor-data Quality**: Most Data Scientists spend a significant part of their time cleaning data
     - Outliers: Discard those or fix the error
     - Missing data: Ignore this feature, fill missing values e.g. by median or train one model with and one model without
   - **Irrelevant Features**: Garbage in, garbage out! Model will only perform well as data contains enough relevant features. Do **feature engineering**:
     - Select most important feature to train on among existing features
     - Extract new features by combining existing features (dimensionality reduction)
     - Create new feature by gathering new data
2. Bad Model:
   - **Overfitting the training data**: Fitting the noise instead of the underlying relation by overgeneralization
     -  Either simplify the model (less parameter), gather more training data or reduce the noise
     -  Constraining a model to make it simpler and reduce the risk of overfitting is called **regularization** e.g. in linear models we can tune the modification of let's say the slope using a **hyperparameter** which is not a model parameter but a parameter of the learning algorithm (constant during training)
   - **Underfitting the training data**: Model is too simple for underlying data
     - Select better model with more parameters or do feature engineering
     - Reduce the constraints on the model (reducing regularization parameter)

### Testing and Validating

We need to train the model on training data while testing and estimating the **generalization error** or **out-of-sample error** by evaluating the performance on unseen test data. High generalization error at low train errors means overfitting!
If you need to decide between two models you can train both and compare how well they generalize but what if you also want to tune hyperparameters. When running 100 instances of this model using 100 different hyperparameters you might get the best performing sets of hyperparameters for best generalization. However, now you basically fitted your hyperparameters to produce the best model for that particular set of test data.

Solution: **Holdout validation**:
1. Select a portion of the training data (**train set**) and train various models with different hyperparameters
2. Select the remaining portion of the training data (**validation set**) to evaluate the best generalization
3. Then you train the best generalization model on the whole training data, i.e. train + validation set
4. The final model is then evaluated on unseen test data (**test set**)

- If validation set is too small you may end up selecting a suboptimal model by mistake.
- If validation set is too large, then the remaining train set will be much smaller than the full training data which is not ideal as candidate models for the final model have been trained on a much smaller dataset

We can solve this by **cross-validation** using many small validation sets. Each model is evaluated once per validation set allowing us to average over many evaluations (training time is obviously multiplied)

- **Data Mismatch** becomes an issue when dividing the data in train validation and test set as we need all of those sets to be representative for the overall task.
  Rule to remember: *Validation and Test set have to be as representative as possible!* e.g. we want an app to classify flowers in self-made pictures and train the algorithm using flowers from the web. If we only have 1000 flowers actually taken with the app while the rest is from the web, we should put half of the representative flowers into validation and half into test.
  However, if you train the model on the web pictures and get poor validation scores you cannot tell if this is due to the web flowers being nonrepresentative or due to overfitting.
  **One Solution**: You can hold out some web training pictures in yet another set (*train-dev set*). After the model is trained on the train set, you can evaluate it on the train-dev set. If the model preforms poorly, then it must have overfit. If it performs well, then you can evaluate it on the validation set. If it performs poorly here, the problem comes from data mismatch.

### No Free Lunch Theorem

You always assume something when selecting a model, e.g. liner model means you assume the data to be fundamentally linear and that the distance between the instances and the straight line is just noise, safely to be ignored.
Which means if one doesn't make any assumptions, there is no reason to prefer one model over any other. There is no model that is *a priori* guaranteed to work better (hence the name of the theorem)

