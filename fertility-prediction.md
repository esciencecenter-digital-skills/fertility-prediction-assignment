---
title: "Fertility prediction using LISS data"
teaching: 0
exercises: 300
---

:::::::::::::::::::::::::::::::::::::: questions 

- How do you start a real-world machine learning project?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Be able to successfully start a real-world machine learning project.

::::::::::::::::::::::::::::::::::::::::::::::::

## Introducing the assignment
In this assignment you will work individually on a machine learning problem.
The end-goal is that you are able to pick up a real-world machine learning project.

This assignment will guide you through such a real-world project. 

### A note on how to learn the most from this assignment
Some participants will learn the most from just loading in the data, explore and build a machine learning pipeline and solve all problems you encounter on the way yourself.
Most of you will likely need the guidance of the exercises and solutions in this assignment to bring you up to speed. 
Try to take some time to solve the challenges yourself before looking at the solutions.
But if you get stuck for too long do not hesitate to let the solutions guide you, or ask one of the trainers.

You will have to find a balance between guidance and self-exploration yourself.

## 1. Introducing the problem
The goal is to predict who will have a child in the next 3 years, 
based on all the data from previous years. 

To train your models, you will have all the background data (features) up to 2019 and the data about the outcome - having a child in 2020-2022 (for about half of cases). So, the outcome is binary (0 - no new children in 2020-2022, 1- at least one new child in 2020-2022).
The performance of the models will then be tested on the remaining cases.

You are encouraged to use different strategies to predict the outcome – e.g. theory-driven (to select a relatively small set of variables that were found significant in previous studies) and data-driven (e.g. features selected based on regularisation).

Features that can be used for the first simple models: gender, age, education level, current partnership status, health, religion, fertility intentions

::: challenge 

## Challenge: Formulating the problem

1. What kind of problem is this? Classification or regression?
2. What are the inputs (features) and what are the outcomes (target)?
2. How many classes are there in the target variable?


:::: solution 

## Solution
1. This is a classification problem
2. Inputs are the background data of the persons (for example gender, age, partnership status), outcome is who will have a child in the next 3 years.
3. There are 2 classes: not having a child in the next 3 years (0) and having a child in the next 3 years (1). This is thus a binary classification problem.

::::
:::


## 2. Reading in and exploring the data

::: challenge 

## Challenge: Reading in the data

1. Try to read in the data using pandas. The features are in 'LISS 2019.csv', the outcome variable is stored in 'outcome.csv'/
Hint: You might run into an encoding error. See if you can fix it by googling for a solution.

:::: solution 

## Solution
### Reading in the features:
```python
# This might result in UnicodeDecodeError: 'utf-8' codec can't decode byte 0x92 in position 50416: invalid start byte
data = pd.read_csv('LISS 2019.csv')

# To tackle the encoding error:
data = pd.read_csv('LISS 2019.csv', encoding='cp1252')
```
### Reading the outcomes:
```python
outcome = pd.read_csv('outcome.csv')
```

::::
:::



::: challenge

## Challenge: Exploring the data

Explore the data.

1. How many features do we have?
2. How many samples do we have?
3. Are the outcome and features datasets ordered in the same way?
4. What type of data do we have as features?
5. Is the target variable well balanced?
6. Do we have any missing data?


:::: solution 

## Solution

Quickly explore the data:
```python
data.shape
```
```output
(9970, 2355)
```

```python
data.head(10)
```

```outcome
	Unnamed: 0	nomem_encr	gebjaar	geslacht	nohouse_encr2019	wave2019	positie2019	leeftijd2019	lftdcat2019	lftdhhh2019	...	cw19l600	cw19l601	cw19l602	cw19l603	cw19l604	cw19l605	cw19l606	cw19l607	cw19l608	cw19l609
0	1	800000.0	1980	Female	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
1	2	800018.0	1985	Male	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
2	3	800021.0	1979	Female	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
3	4	800033.0	1991	Male	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
4	5	800042.0	1975	Female	500277.0	201912.0	Wedded partner	44.0	35 - 44 years	47.0	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
5	6	800057.0	1975	Male	580532.0	201912.0	Unwedded partner	44.0	35 - 44 years	43.0	...	80.0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
6	7	800076.0	1985	Female	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
7	8	800085.0	1977	Male	545773.0	201912.0	Household head	42.0	35 - 44 years	42.0	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
8	9	800091.0	1983	Male	515359.0	201912.0	Household head	36.0	35 - 44 years	36.0	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
9	10	800100.0	1990	Female	518099.0	201912.0	Wedded partner	29.0	25 - 34 years	34.0	...	NaN	NaN
```

```python
outcome.head(10)
```
```outcome
	nomem_encr	new_child
0	800000.0	NaN
1	800018.0	NaN
2	800021.0	NaN
3	800033.0	NaN
4	800042.0	NaN
5	800057.0	0.0
6	800076.0	NaN
7	800085.0	NaN
8	800091.0	NaN
9	800100.0	1.0
```

1. There are 2353 features (excluding `nomem_encr` which is just an identifier)
2. There are 9970 samples
3. Yes, looking at the first 10 rows they are ondered in the same way.
4. The features are both categorical and numerical

```python
outcome['new_child'].describe()
```

```output
count    1292.000000
mean        0.164087
std         0.370498
min         0.000000
25%         0.000000
50%         0.000000
75%         0.000000
max         1.000000
Name: new_child, dtype: float64
```
5. The target variable is pretty unbalanced, only 16.4 % of the samples are in the `1` class
6. There are a lot of missing data in both the features and the outcome.

::::
:::

## 3. Taking a step back
Before you start enthusiastically typing all kinds of pandas and sklearn commands that you just learned.
Now that you have explored the data, it is good to take a step back and think about how you want to approach the problem.

::: challenge

## Challenge: What steps to take next?

Now that you know the data a little bit. What are the steps that you need to take before we can train our first modeling pipeline.
Think about the essential steps that will get you a working pipeline. 
Take shortcuts where possible.
It does not need to be a very accurate predictor (yet ;) ).
Also think about the order in which to do things.

:::: solution

## Solution

These are the minimum steps that need to happen, in this order:

1. Select features
2. Deal with missing data. The quickest way is to just drop all rows that have any missing value.
3. Preprocess the features: scaling for numerical values, one-hot encoding for categorical values.
4. Split the data in a train and test set.
5. Train the model and evaluate!

We will not worry about the unbalanced target yet. Let's first see how a model performs on the unbalanced dataset.
::::
:::

## 4. Prepare the data

::: challenge

## Challenge: Selecting data for a quick and dirty first model

1. Take max 5 minutes to have a look at the codebook. Quickly pick 5 variables that you think will have some explanatory value for predicting fertility.
2. Use `pandas` to keep only these columns in your dataset

:::: solution
There is no right or wrong here.

A good set to start with would be to pick 5 variables from 2019. The most recent year has probably the most explanatory value.
Based on gut-feeling we selected these 5 variables:

* burgstat2019
* leeftijd2019
* woonvorm2019
* oplmet2019
* aantalki2019

What matters is that you for the first cycle, just quickly pick the most promising variables.

You can easily spend hours deciding which variables to pick, 
but it is important to have your first pipeline as soon as possible and start iterating from there.
This is maybe something you are not used to in research!

#### 2. Select the data

```python
selected_columns = ['burgstat2019', 'leeftijd2019', 'woonvorm2019', 'oplmet2019', 'aantalki2019']
features = data[selected_columns]
```
::::
:::

### Dealing with missing values
There are many ways to deal with missing data. 
The quick and dirty way is to just get rid of all rows that contain any missing value.

::: challenge

## Challenge: Remove missing values
Remove all samples in both the features and target that have any missing value.
NB: so if the target value is null or any of the features is null we drop the entire sample.

Hint: You need to find a clever way to delete the samples in both the target and feature datasets.

How many samples do we have left?

:::: solution

## Solution

```python
# The null values in the target variable
y_isna = outcome['new_child'].isnull()

# The rows where any of the features is NaN
X_isna = features.isnull().any(axis=1)

# For both datasets drop rows where any of the features is NaN OR the outcome is NaN
features = features.drop(features[y_isna | X_isna].index)
outcome = outcome.drop(outcome[y_isna | X_isna].index)
```
You could also first `join` the two datasets, drop any row with missing value,
then split the datasets again.

```python
features.shape
```
```outcome
(1292, 5)
```

There are 1292 samples left.

::::
:::

::: callout
Remember that the benchmark dataset that is used to test your submissions will also have missing values.
The quick and dirty solution that we are starting with will thus not be able to make predictions for the samples in the benchmark dataset that have missing values in the features that you picked!
:::

### Preprocess the data

::: challenge

## Challenge: Preprocess the data

Create a preprocessor to process the features. (hint: use the `ColumnTransformer` class that we introduced before) to preprocess the data
Use a standard scaler for numerical values, one-hot encoding for categorical values.

:::: solution

## Solution

Select the numerical and categorical columns automatically:
```python
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(features)
categorical_columns = categorical_columns_selector(features)
```

Define a preprocessor:
```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)])
```
::::
:::

### Separate data in train and test

::: challenge

## Separate the data in a train and test set

Separate the data in a train and test set, the test set should have 30% of the samples.

:::: solution

## Solution

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    features, outcome['new_child'], test_size=0.30, random_state=42)
```

::::
:::

## 5. Train a model

::: challenge

## Challenge: Train the model
Create a pipeline that pipes our preprocessor and a `LogisticRegression` model.
Fit the pipeline to the data.

:::: solution
## Solution
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
model
```
```output
Pipeline(steps=[('columntransformer',
                 ColumnTransformer(transformers=[('one-hot-encoder',
                                                  OneHotEncoder(handle_unknown='ignore'),
                                                  ['burgstat2019',
                                                   'woonvorm2019', 'oplmet2019',
                                                   'aantalki2019']),
                                                 ('standard_scaler',
                                                  StandardScaler(),
                                                  ['leeftijd2019'])])),
                ('logisticregression', LogisticRegression(max_iter=500))])
```

```python
_ = model.fit(X_train, y_train)
```
::::
:::

## 6. Evaluate the model

::: challenge

## Challenge: Evaluate the model
Let's evaluate our first model.

1. Visualize the performance of the model.
2. What do you think of the results? Why do you think the results are like this?

:::: solution

## Solution
You can best visualize the results using a confusion matrix:
```python
from sklearn.metrics import ConfusionMatrixDisplay

_ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
```
![](../fig/confusion-matrix-1.png)

As you can see the model almost always predicts the majority class (0 - no kids).
Because 84% of the data falls within this class, this is actually not a bad strategy. 
It leads to an accuracy of 84%!

Of course this is not what we want, we also want to model to make correct predictions for the minority class.

We thus have to deal with imbalanced data!

::::
:::

## 7. Dealing with imbalanced data

::: challenge
## Challenge: How do deal with unbalanced ata?
What do you think is a good way to deal with the unbalance in the target variable?

:::: solution

## Solution
There are multiple ways to deal with this:

1. Find a model that can handle unbalanced data better
2. Upsample the data (duplicate a proportion of the minority class samples)
3. Downsample the data (take a subset of the majority class samples)
::::
:::

::: challenge

## Challenge: Balance the dataset

Go ahead and balance the training dataset using the upsampling strategy.
Hint: Try googling first, but you can have a look at: https://vitalflux.com/handling-class-imbalance-sklearn-resample-python/

:::: solution

## Solution
```python
from sklearn.utils import resample
#
# Create oversampled training data set for minority class
#
X_oversampled, y_oversampled = resample(X_train[y_train == 1],
                                        y_train[y_train == 1],
                                        replace=True,
                                        n_samples=X_train[y_train == 0].shape[0],
                                        random_state=123)
#
# Append the oversampled minority class to training data and related labels
#
X_balanced = pd.concat((X_train[y_train == 0], X_oversampled))
y_balanced = pd.concat((y_train[y_train == 0], y_oversampled))
```
Check that the data is indeed balanced:
```python
y_balanced.mean()
```
```outcome
0.5
```

Check that there is now indeed more in the training set:
```python
X_balanced.shape
```
```output
(1506, 5)
```
::::
:::

## 8. Train and evaluate the model trained on a balanced dataset
Now that we have balanced our dataset, we can see whether this improves the performance of our model!

::: challenge

## Challenge: Train the model on the balanced dataset

1. Train the model on the balanced dataset
2. Evaluate the model by visualizing the results. What do you see?

:::: solution

```python
_ = model.fit(X_balanced, y_balanced)
_ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
```
![](../fig/confusion-matrix-2.png)

This already looks much better! 
The predictions are more balanced over the two classes, and are also quite often correct.
So balancing the data helped a lot!
::::
:::

## 9. Use evaluation metrics to evaluate the model
So far we have used a confusion matrix to visually evaluate the model.
When proceeding it would be better to use evaluation metrics for this.

::: challenge
## Challenge: Evaluation metrics
Evaluate the model using the appropriate evaluation metrics.
Hint: the dataset is unbalanced.

:::: solution

## Solution
Good evaluation metrics would be macro precision, recall, and F1-score, 
because we want to get a feeling for how the model performs in both classes of the target variable.
In other words, we value a model that can predict both true positives as well as true negatives.

```python
y_pred = model.predict(X_test)
p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
print(f'Precision: {p}, recall: {r}, F1-score: {f}')
```
```outcome
Precision: 0.6297419895408973, recall: 0.7251215721662405, F1-score: 0.6295138888888889
```

::::
:::


## 10. Adapt, train, evaluate. Adapt, train, evaluate.
Good job! You have now set up a simple, yet effective machine learning pipeline on a real-world problem.
Notice that you already went through the machine learning cycle twice.
From this point onwards it is a matter of adapting your approach, train the model, evaluate the results. Again, and again, and again.

Of course there is still a lot of room for improvement. 
Every time you evaluate the results, try to come up with a shortlist of things that seem most promising to try out in the next cycle.

::: challenge
## Challenge: Improving the model further
What ideas do you have to improve the model further?

:::: solution

## Solution
There is no right or wrong, but here are some pointers:

- Include more features. You can think of automated feature selection to select features or go through the codebook by hand and select the most promising.
- Engineer more features. You could try and create new features yourself, based on the variables in the dataset, or maybe combine a different dataset.
- Try out different models.
- Come up with a good baseline, to get a feeling for how good the model is performing.
- Use hyperparameter tuning to find the best hyperparameters for a model.
- Try out different preprocessing approaches
- Deal with missing data differently. Try out imputation.

::::
:::

