# Support Vector Machines (SVM)

## Part A: SVM Implementation
Implement a basic SVM classifier using a linear kernel.  The workflow for the implementation canbe as follows:

### Data Loading
Download the dataset from [here](https://archive.ics.uci.edu/dataset/94/spambase) and split it into training and testing sets (80% for training, 20% for testing).

### Training the SVM
Vectorize the data into X and y vectors for the features and the label respectively. Fit the SVM model on the training data and report the accuracy.

### Prediction and Evaluation
Implement functions to generate predictions on the test set and calculate accuracy, precision, recall, and F1-score for the SVM model.

### Regularization
Vary the regularization parameter of the SVM and tabularize and plot the accuracy. You can use the following values: [0.001, 0.1, 1, 10, 100]. Observe the changes in the parameter values (by observing the mean, maximum, etc.) as the value of the regularization parameter is changed.


## Part B: Kernel Tricks 
The ”kernel trick” in Support Vector Machines (SVMs) is a clever mathematical technique thatallows SVMs to operate in a higher-dimensional space without explicitly calculating the transfor-mations.  This not only saves computational resources but also opens up a variety of usage scenariosby enabling SVMs to handle different types of data distributions and complexities.

### Impact of Various Kernels
Observe the impact of various kernels on the performance of the model. Report the accuracy, precision, recall, and F1 score on the test set using the following kernels:
1. Polynomial with degree 2
2. Polynomial with degree 3
3. Sigmoid
4. Radial Basis Function (RBF)

##Part C: Overfitting & Underfitting Analysis 
In this part, we will explore the impact of kernel complexity and regularization in SVMs on modelperformance, and the presence of underfitting or overfitting.

### Dataset Reuse
Reuse training and testing datasets from Part A.

### Train SVM Models
Train SVM models with different degrees of the polynomial kernel and different 'C' values. Plot and tabulate the test and train accuracy to showcase overfitting and underfitting.


# Naive Bayes

#### Part A: Probability

1. **Simulation for k = 4 and 4 rolls:**
   - Simulated rolling a biased 4-faced die 1000 times, calculating the sum of the upward face value, and plotting a frequency distribution histogram.
   - Calculated the theoretical expected sum using the formula and compared it with the simulated sum.

2. **Simulation for k = 4 and 8 rolls:**
   - Repeated the above process for 8 rolls instead of 4.

3. **Simulation for k = 16:**
   - Repeated the process for a biased 16-faced die.

#### Part B: Implementation of Naive Bayes (From Scratch)

1. **Dataset:**
   - Used the Email Spam Classification dataset, containing 4601 instances with 57 attributes and a spam label.

2. **Loading Dataset:**
   - Loaded the data with a 70:15:15 split for training, validation, and testing using sklearn.

3. **Plot Distribution:**
   - Choose 5 columns from the dataset and plotted their probability distribution.

4. **Priors:**
   - Calculated and printed the priors of classes.

5. **Train Model:**
   - Implemented Naive Bayes algorithm from scratch, including fit and predict functions. Mentioned the total number of parameters needed to be stored for the model.

6. **Prediction and Evaluation:**
   - Implemented functions to generate predictions on the test set and calculated accuracy, precision, recall, and F1-score for the Naive Bayes model.

7. **Log Transformation:**
   - Applied log transformation to all columns of the dataset, retrained the Naive Bayes Classifier, and evaluated the model as before.

Notice in the results before and after modifying the dataset.

#### Part C: Implementation of Naive Bayes (sklearn)

1. **Train the model:**
   - Imported GaussianNB from sklearn.naivebayes and trained the model with the loaded dataset and after log transformation.

2. **High precision:**
   - Drew a ROC curve for both models and chose the best model based on the curve, considering the importance of minimizing false positives in email spam classification.

### Note:
- Detailed explanations and code implementation for each part are provided in the respective sections of the codebase.
- Code is organized and documented for easy understanding and reproducibility.
