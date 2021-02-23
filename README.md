# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.

In this project i had the the opportunity to create and optimize an ML pipeline. The idea was to create a custom-coded model —a standard Scikit-learn Logistic Regression — them make hyperparameters tunning optimized using HyperDrive. I also had to use AutoML to build and optimize a model on the same dataset. After that the intend was to compare the results of the two methods, as a plus i excecuted a standard scikit-learn Logistic Regression to see if there's a huge change.

Project phases:
> - Creating a `train.py` with a custom model Logistic Regression from the dataset provided as Tabular Dataset  and evaluate the performance of the model.
>- Creating of a jupyter notebook using HyperDrive to find the best parameters, using the `train.py` as input model 
>- In the same notebook, i was load the dataset as Tabular Dataset and perform AutoML to find the best model, evaluate and keep the best model.
>- As Final task, was compared the results of two methods and write a research report in this readme format.

Below you can see a flow design of all tasks:

![Project Flow](https://video.udacity-data.com/topher/2020/September/5f639574_creating-and-optimizing-an-ml-pipeline/creating-and-optimizing-an-ml-pipeline.png) 

## Summary

The Dataset contains data about Banking-Marketing of a Portuguese banking institution, and the main goal was to predict whenever a customer will respond or not (column y) an marketing campaigns (phone calls). For this case the best performing model was  **VotingEnsemble** provided from **AutoML method** with Accuracy of : 0.9156 that was higher than Hyperdrive (0.9146) and simple logistic Regression(0.9150).

# Index

1. [Scikit-learn Pipeline](#ScikitlearnPipeline)
2. [AutoML](#AutoML)
3. [Pipeline comparison](#PipelineComparison)
4. [Future work](#FutureWork)
5. [Proof of cluster clean up](#Proof)
6. [Licensing and Acknowledgements](#Licensing)
7. [References](#References)

## Scikit-learn Pipeline <a name="ScikitlearnPipeline"></a>

The data as mentioned was cleaned (making categorical variables numerics and removing null values) before creating the model. Thus  the data was splited into training and test using ratio of 70/30.

The Logistic-Regression model was created with `C` of `1.0` and `max_iter` of `100` as default. In this case using a compute instance to run the model had 0.9150 of accuracy  that is the metric chosen for compare all approachs. See the imagem below.

![Project Flow](https://i.ibb.co/Wk5SwxZ/simplelogisticregression.jpg)

The hyperparameters search method chosen for hyperparameter tuning with HyperDrive was the RandomParameterSampling method, because it's fast simple search method and supports early stopping. The GridParameterSamplig is probably better to find parameters but it is uses exhaustively search and may need more time and cumputational resource to be executed. In this case i chose to combine `C` and `max_iter` to find the best params, like the code below:

    ps = RandomParameterSampling(
        {
            '--C': choice(0.001,0.1,1,10,50,100,500,1000),
            '--max_iter': choice(30,50,100,200,500)
        }
    )

Those parameters in Logistic-Regression docummentation means:
	`C` - Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
	`max_iter` - Maximum number of iterations taken for the solvers to converge.
	
For the early stopping policy i chose the BanditPolicy method, with optional parameters like **evaluation_interval** and **slack_factor**. Following the docummentation it explicit says that `evaluation_interval` is the frequency for applying the policy, it reffers to amount of runs that logs the primary metric. Otherwise the `slack_factor` means the ratio used to calculate the allowed distance from the best performing experiment run. So as i chose the time to be applied the policy is 2 by 2 and the max distance allowed is 0.1, thus any run that doesn't meet the requiriments will be finished and if the result doesn't keep increasing the accuracy 2 times in sequence the HyperDrive will stop with the best model, that's why i chose this reasonable parameter, as you can see bellow:

	policy = BanditPolicy(evaluation_interval = 2, slack_factor = 0.1)


## AutoML <a name="AutoML"></a>

For AutoML the parameters chosen follow below

	automl_config = AutoMLConfig(compute_target = compute_target,
   		experiment_timeout_minutes=30,
   		task='classification',
    		primary_metric='accuracy',
    		training_data=ds,
    		label_column_name='y',
    		enable_onnx_compatible_models=True,
    		n_cross_validations=2

AutoML is a simple way to do machine learning in cloud, the parameters are easy to understand, from the chosen params only `enable_onnx_compatible_models`, `experiment_timeout_minutes` and `compute_target` probably needs to be explained. Thus follow the explanation from the docummentation bellow:

>- `enable_onnx_compatible_models`- Whether to enable or disable enforcing the ONNX-compatible models. The default is False. For more information about Open Neural Network Exchange (ONNX) and Azure Machine Learning, see this  [article](https://docs.microsoft.com/en-us/azure/machine-learning/concept-onnx)

>- `experiment_timeout_minutes`- Maximum amount of time in hours that all iterations combined can take before the experiment terminates.

>- `compute_target`- The Azure Machine Learning compute target to run the Automated Machine Learning experiment on.

For the **primary_metric** i have used the accuracy since is the metric chosen to compare all approachs, and **task** was classification beacause of the nature of the problem that we want to solve.

## Pipeline comparison <a name="PipelineComparison"></a>

For the two pipilines presented above with HyperDrive and AutoML. The second one performed slightly better but the gain is almost inrelevant between two result metrics, HyperDrive got  accuracy of 0.9146 and AutoML 0.9156. Thus The HyperDrive model  would be a better Choice since it took almost half of time comparing with AutoML to run. Otherwise AutoML reached a very accurated result with low effort of explicit programing. The table below presents both model details.

| METHOD  | ALGORITHIM | EXPERIMENT ID  | ACCURACY|
| ------------- | ------------- | ------------- | ------------- |
| Sklearn | Logistic Regression  |  | 0.9150 |
| HyperDrive | Logistic Regression  | HD_6e67214b-d7cb-4f3a-afc0-11a9445d5d17 | 0.9146 |
| AutoML  | VotingEnsemble | AutoML_ac058689-383f-4349-a9a8-d1ad303ce880_25 | 0.9156 |

The difference of accuracy between  those models can be caused by the parameters space chosen for hyperparemeter tunning doens't contemplate the best result, thus using Gridsearch would be a good try to improve HyperDrive. In AutoML the Ensemble model is more powerful than  simple Logistic-Regression, this is another factor that could cause this difference.

For all Approachs that we look at this project, the improvment wasn't realy a huge gap between tham, simple SKlearn Logistic-Regression had a good result and low time of execution, In contrast AutoML had the best performance but took 30 min to execute, as conclusion those models are equaly good but you need to know when to use each and how much time you can spend by training and tunning a model.

## Future work <a name="FutureWork"></a>
This project can be improved by using some feature engineering, feature selection or even unsupervised technics (e.g clustering by age). It also can be perform a balance of calss beacause the data set has a huge disparite of peaople who responded positively and negatively, as the image below. 

![barplot](https://i.ibb.co/3kfCwh0/barplot.jpg)

This could result a better generalist model, since that imbalance occour where the distribution of examples across the known classes is biased or skewed. This results in models that have poor predictive performance, specifically for the minority class.

## Proof of cluster clean up <a name="Proof"></a>
![delete cluster](https://i.ibb.co/R9q6dMd/deletecluster.jpg)

## Licensing and Acknowledgements<a name="Licensing"></a>

[MIT License](https://github.com/git/git-scm.com/blob/master/MIT-LICENSE.txt).

Thanks for Udacity for give the oportunity to work with this data.

## References <a name="References"></a>

1. https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-first-experiment-automated-ml

2. https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/automated-machine-learning/classification-credit-card-fraud/auto-ml-classification-credit-card-fraud.ipynb

2. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-machine-learning-pipelines

4. https://github.com/microsoft/MLHyperparameterTuning/blob/master/01_Training_Script.ipynb

5. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

6. https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py

7. https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py

9. https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriverun?view=azure-ml-py

10. https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py

11. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-remote

13. https://machinelearningmastery.com/what-is-imbalanced-classification/