# Machine Learning Production Pipeline (MLPP)

### Motivation
Machine Learning projects when started from scratch need to undergo the following steps:
1. Data Cleaning
2. Data Visualization
3. Decisions based on Insights
4. Feature Engineering based on visualizations and heuristics

### Problems 
These same steps are being retraced again and again with every new dataset.
There are a few problems that we will try to address:
1. Repetitive - Could be automated with minimal intervention. Rewriting the same code is boring and not a good use of time.
2. Error prone - Looking for problems in data is tedious and tiring and we might miss out on details. 
3. Sifting through data manually is not easy - But once converted with visuals with auto-outlier detection and preliminary analysis could be good.
4. Every time we start with some model, we try the hacky methods and get the POC done but getting it into production is a headache. - Why not begin with production level design.

We're trying to eliminate the time consuming process to help developers with more of their time for creative decision making.

### MLPP Features
1. Multiple datasets
2. Decoupled feature engineering with support of multiple and hybrid feature sharing amongst models.
3. Hierarchical models
4. Automated initial data analysis and plotting
5. Autoclustering / PCA
6. Auto apply a few models in case of regression or classification tasks
7. Auto feature engineering for a few numerical, categorical and textual data
8. Apply deep learning models
9. Support for model predictions - single/batch/mass predictions
10. Support multiple input outputs