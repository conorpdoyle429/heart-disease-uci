Heart Disease UCI

**Introduction**

This project uses data (from a Cleveland database) that represents attributes about healthcare patients. It is used to determine the presence of heart disease. Data is from [https://www.kaggle.com/ronitf/heart-disease-uci](https://www.kaggle.com/ronitf/heart-disease-uci).

The attributes are as follows:

1. age (in years)
2. sex (0=female, 1=male)
3. cp. Chest pain type (0=asymptomatic, 1=atypical angina, 2=pain without relation to angina, 3=typical angina)
4. trestbps. Resting blood pressure (mm Hg)
5. chol. Serum cholestoral (mg/dl)
6. restecg. Electrocardiogram at rest (0=probable left ventricular hypertrophy, 1=normal, 2=abnormalities in the T wave or ST segment)
7. fbs. Blood sugar level \&gt; 120mg/dl (0=no, 1=yes)
8. thalach. Maximum heart rate achieved
9. exang. Exercise induced angina(0=no, 1=yes)
10. oldpeak = ST depression induced by exercise relative to rest
11. the slope of the peak exercise ST segment (0=descending, 1=flat, 2=ascending)
12. number of major vessels coloured by dye (0-4)
13. thal.Blood flow observed by dye. (0=null, 1=normal, 2=fixed defect, 3=reversable defect)

This dataset includes data on 303 patients.

I am using this project to experiment with applying ML techniques on a real-world database. I will do some data visualisation with pandas. I will run logistical regression using scikit-learn and create a neural network using Tensorflow. This dataset is likely too small to be suitable for deep learning techniques, however the aim of this project is to practice the techniques so this isn&#39;t a major issue.

**Data Visualisation**

Pandas is Python data analysis library. I am using this to create bar charts and pie charts to represent the data.

**One Hot Encoding**

One hot encoding is a way of representing categorical data. This technique is used in this project for the following attributes:

- cp
- slope
- restecg
- ca
- thal

I will explain this technique using cp as an example. The raw data for cp includes one column with values ranging from 0 for asymptomatic chest pain to 3 for typical angina. This is an issue as an instance of the data with typical angina with get a larger weighting in the model than data with asymptomatic chest pain; ideally these should be equally weighted.

Instead of one column, there will be a column for each value. These columns will have binary input.

In the raw data, cp would be represented as:

| **cp** |
| --- |
| 3 |
| 2 |
| 1 |
| 0 |
| 2 |

Using one hot encoding, this data would be represented as:

| **cp\_0** | **cp\_1** | **cp\_2** | **cp\_3** |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 1 |
| 0 | 0 | 1 | 0 |
| 0 | 1 | 0 | 0 |
| 1 | 0 | 0 | 0 |
| 0 | 0 | 1 | 0 |

**Normalising Data**

I have normalised all non-binary data. The aim of this is to get values for all attributes to be on a similar scale. This avoids giving too much significance to attributes with higher values. For example, without normalising data values, resting heart rate has much larger values than any binary inputs. This will give them unjustified high weighting in the model.

**Logistical Regression**

I used scikit-learn for logistical regression.

I experimented with adding k-fold cross validation, a common technique when using limited data sets, which will hopefully give less biased results. This method splits the data into k subsets. One subset is used each time for validation, while the rest go to training. This averages out the error across the subsets so it matters less how the data is divided.

I also experimented with using F-score (the harmonic mean of precision and recall) to test accuracy.

**Neural Network**

I used Tensorflow to create the neural network.

I used four layers, using the following setup:

LINEAR -\&gt; RELU -\&gt; LINEAR -\&gt; RELU -\&gt; LINEAR -\&gt; RELU -\&gt; LINEAR -\&gt; SIGMOID

Weights were initialised using Xavier initialisation, which I hope with converge faster than weights that are initialised randomly and help avoid vanishing/exploding gradients.

The depth of the layers is 27, 13, 6 and 3. The number of epochs was 100 and the training rate was 0.001. Accuracy was used as a metric. I tried some tuning of these parameters, however it wasn&#39;t a large focus. Due to how small the dataset, I felt it would be hard to avoid bias here. I was more focussed on learning the frameworks than hyper-tuning parameters.

Adam optimisation is used. This is used in place of stochastic gradient descent and combines the Adaptive Gradient Algorithm (improves performance on sparse gradients) and Root Mean Square Propogation (performs well on noisy data).

I experimented with adding training rate decay. This involves reducing the training rate as the number of epochs increases, to try to help converge to a minimum during optimisation.
