# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

Attribute Information:

Input variables:
# bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

The best performing model found out using Azure's AutoML was VotingEnsemble classifier. With Azure's AutoML, we received an accuracy score of 0.9166 within just 5 iterations.

## Scikit-learn Pipeline
Steps in entry train.py:

1.TabularDataset creation using TabularDatasetFactory.
2.Data cleaning 
3.Splitting the data into train and test sets.
4.Training the logistic regression model using arguments from the HyperDrive runs.
5.Calculating the accuracy score.

Steps involved in udacity-project.ipynb:

1.Assigning a compute cluster to be used as the target.
2.Specifying the parameter sampler(RandomParameterSampling).
3.Specifying an early termination policy(BanditPolicy).
4.Creating a SKLearn estimator for use with train.py.
5.Creating a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
6.Submitting the hyperdrive run to the experiment and showing run details.
7.Getting the best run id and saving the model.
8.Saving the model under the workspace for deployment.

Logistic regression is a classification algorithm, used when the value of the target variable is categorical in nature. Logistic regression is most commonly used when the data in question has binary output, so when it belongs to one class or another.For our project , the marketing campaigns were based on phone calls to find out whether bank term deposit would be subscribed or not subscribed.

**What are the benefits of the parameter sampler you chose?**
Parameter sampling means to search the hyperparameter space defined for your model and select the best values of a particular hyperparameter.Azure supports three types of parameter sampling - Random sampling,Grid sampling and Bayesian sampling. In random sampling, hyperparameter values are randomly selected from the defined search space. Random sampling allows the search space to include both discrete and continuous hyperparameters. It supports early termination of low-performance runs. Some users do an initial search with random sampling and then refine the search space to improve results. Bayesian sampling does not support early termination.

**What are the benefits of the early stopping policy you chose?**
An early termination policy ensures that we don't keep running the experiment for too long and end up wasting resources and time, in order to find the optimal parameter. A run is cancelled when the criteria of a specified policy are met. In our project we have used BanditPolicy as the early termination policy.This early termination policy is based on the slack factor and delay evaluation. Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated. For more aggressive savings on resourses, we used Bandit Policy with a smaller allowable slack over other policies.

## AutoML
**Description of the model and hyperparameters generated by AutoML:**
Since it was a classifiaction task, the primary metric that was to be maximized was set to 'accuracy'. We provided the cleaned version of the data, and set no. of cross-validations folds to 7 to prevent overfitting, if in case. The model was trained remotely on the compute cluster created initially. We have used the task as "clasification" and primary metric as "accuracy" to compare it with the HyperDrive option.the AutoML model was iterated only 5 times as the experiment was bound to time out after 30 minutes . The model that gave the best results turned out to be VotingEnsembleClassifier that takes the average of the predictions of the base models. It gave as an accuracy score of 0.9166 which was slightly better than the score achieved using HyperDrive. 

## Pipeline comparison
**Comparison between the two models and their performance:**
Model trained using AutoML gave slightly better results. The AutoML model gave best accuracy of 0.9166(with VotingEnsembleClassifier), while the model built using SKLearn and HyperDrive gave a slightly lower score of 0.9072. The SKlearn model was iterated 20 times, while the AutoML model was iterated only 5 times(due to time limits). If the number of entries in the dataset is large, this could be a differentiating factor. Apart from that, an entry script was not a part of the architecture of the AutoML config.

## Future work
Implementation of feature engineering can be a potential field for future work. We can also tune some other hyperparameters used in the model and use the pipelines suggested by the AutoML models in order to achieve better results in the future. Moreover , intead of the traditional logistic regression used in this project , better results can be obtained using deep learning models.
