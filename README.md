# Intro to Machine Learning with TensorFlow Nanodegree Program

## Supervise Learning

### Lesson 1

* Learning from labelled data

* Supervised learning can be divided in two models
    * Classification
        * Categorical Outcomes
        * Any number of categories 
        * Type of Dog

    * Regression
        * Numeric Outcomes
        * A persons height for example
    
* Unsupervised Learning
    * No labelled data
    * Grouping similar items 

* Reinforcement Learning
    * Training algorithms that take rewards based on actions
    * Game playing agents 

* ![intro](./images/intro.png)

### Lesson 2

### Linear Regression

* Linear Regression
    * Draw the best fitting line to it

    * ![moving_line](./images/moving_line.png) 

    * Absolute Trick 
        * `p`: Horizontal
        * `q`: Vertical
        * `w1`: Slop
        * `w2`: Y intercept 

        * ![linear_regression](./images/linear_regression.png)

        * ![linear_regression_1](./images/linear_regression_1.png)

        * We want to take tiny steps, so lets multiple this by a small number (learning rate)

        * ![linear_regression_2](./images/linear_regression_2.png)

        * If the point is not on top of the line, but actually underneath we will subtract instead of sum

        * ![linear_regression_3](./images/linear_regression_3.png)


        * The fact we `p` is connected with the position of our point. If the point is on the right `w1` will move the slop up, but if `p` is negative `w1` will be moved down

        * ![linear_regression_4](./images/linear_regression_4.png)

        * Also, if the point distance is small (meaning `p` is small), we will add a small value to our slop. However, if `p` is large we will add a large number to the slop 

        * ![linear_regression_5](./images/linear_regression_5.png)

        * ![linear_regression_6](./images/linear_regression_6.png)


        * Great job! Since the point is below the line, the intercept decreases; since the point has a negative x-value, the slope increases.

            * ![exercise_absolute](./images/exercise_absolute.png) 

            * ![exercise_absolute](./images/exercise_absolute1.png) 


    * Square Trick 

        * ![square_trick](./images/square_trick.png)

        * ![square_trick1](./images/square_trick_1.png)

        * Besides moving the slop `w1` horizontally (`p` value) we will also take `(q-q')` in order to account to the vertical axis

        * ![exercise_square](./images/exercise_square.png) 
    
    * Gradient Descent

        * Minimizing the error

        * ![gradient_descent](./images/gradient_descent.png)
    
    * Error Functions

        * Mean Absolute Error

            * ![mean_absolute_error](./images/mean_absolute_error.png)  

            * ![mean_absolute_error_total](./images/mean_absolute_error_total.png)  

                * Sum of all errors divided by `m` which is the number of data points
            
            * ![mean_absolute_error_total2](./images/mean_absolute_error_total_2.png)  

                * We take the absolute value so we don't cancel positive errors with negative errors in our data set
        
        * Mean Square Error

            * ![mean_square_error](./images/mean_square_error.png) 

            * ![mean_square_error](./images/mean_square_error_2.png) 

                * This is always gonna be a positive error since we are taking the square of it
            
            * ![mean_square_error](./images/mean_square_error_3.png) 
        
        * Minimizing Error Functions

            * ![minimizing_error_functions](./images/minimizing_error_functions.png) 

                * The tricks we saw before are really similar to our current derivative. Check the tricks again and you will see that `(y - y')x` is part of the trick.
            
            * ![minimizing_error_functions_2](./images/minimizing_error_functions_2.png) 

            * ![minimizing_error_functions_3](./images/minimizing_error_functions_3.png)

                * This is exactly what this gradient descent step is doing. Multiplying `(y - y')x` by a learning rate 
            
            * ![minimizing_error_functions_4](./images/minimizing_error_functions_4.png)

            * ![minimizing_error_functions_5](./images/minimizing_error_functions_5.png)
    
    * Batch vs Stochastic Gradient Descent

        * At this point, it seems that we've seen two ways of doing linear regression.

            * By applying the squared (or absolute) trick at every point in our data one by one, and repeating this process many times.

            * By applying the squared (or absolute) trick at every point in our data all at the same time, and repeating this process many times.
        
        * More specifically, the squared (or absolute) trick, when applied to a point, gives us some values to add to the weights of the model. We can add these values, update our weights, and then apply the squared (or absolute) trick on the next point. Or we can calculate these values for all the points, add them, and then update the weights with the sum of these values.

        * The latter is called **batch gradient descent. The former is called stochastic gradient descent.**

        * ![batch_vs_stochastic](./images/batch_vs_stochastic.png)

        * The question is, which one is used in practice?

        * Actually, in most cases, neither. Think about this: If your data is huge, both are a bit slow, computationally. The best way to do linear regression, is to split your data into many small batches. Each batch, with roughly the same number of points. Then, use each batch to update your weights. This is still called **mini-batch gradient descent**.

        * ![mini_batch](./images/mini_batch.png)
    
    * Mini-Batch Gradient Descent Quiz

        * In this quiz, you'll be given the following sample dataset (as in data.csv), and your goal is to write a function that executes mini-batch gradient descent to find a best-fitting regression line. You might consider looking into numpy's `matmul` function for this!

        * ```python
            def MSEStep(X, y, W, b, learn_rate = 0.001):
            """
            This function implements the gradient descent step for squared error as a
            performance metric.
            
            Parameters
            X : array of predictor features
            y : array of outcome values
            W : predictor feature coefficients
            b : regression function intercept
            learn_rate : learning rate

            Returns
            W_new : predictor feature coefficients following gradient descent step
            b_new : intercept following gradient descent step
            """
            
            # compute errors
            y_pred = np.matmul(X, W) + b
            error = y - y_pred
            
            # compute steps
            W_new = W + learn_rate * np.matmul(error, X)
            b_new = b + learn_rate * error.sum()
            return W_new, b_new
            ```

    * Linear Regression with scikit-learn

        * In this section, you'll use linear regression to predict life expectancy from body mass index (BMI). Before you do that, let's go over the tools required to build this model.

        * For your linear regression model, you'll be using scikit-learn's LinearRegression class. This class provides the function fit() to fit the model to your data.

        * ```python
            >>> from sklearn.linear_model import LinearRegression
            >>> model = LinearRegression()
            >>> model.fit(x_values, y_values)
            ```

        * Fitting the model means finding the best line that fits the training data. Let's make two predictions using the model's predict() function.

        * ```python
            >>> print(model.predict([ [127], [248] ]))
            [[ 438.94308857, 127.14839521]]
            ```

        * The model returned an array of predictions, one prediction for each input array. The first input, [127], got a prediction of 438.94308857. The second input, [248], got a prediction of 127.14839521. The reason for predicting on an array like [127] and not just 127, is because you can have a model that makes a prediction using multiple features. We'll go over using multiple variables in linear regression later in this lesson. For now, let's stick to a single value.

        * ```python
            # TODO: Add import statements
            import pandas as pd
            from sklearn.linear_model import LinearRegression

            # Assign the dataframe to this variable.
            # TODO: Load the data
            bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")

            # Make and fit the linear regression model
            #TODO: Fit the model and Assign it to bmi_life_model
            bmi_life_model = LinearRegression()
            bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])

            # Mak a prediction using the model
            # TODO: Predict life expectancy for a BMI value of 21.07931
            laos_life_exp = bmi_life_model.predict(21.07931)
            ```
        
    * Multiple Linear Regression

        * In the last section, you saw how we can predict life expectancy using BMI. Here, BMI was the predictor, also known as an independent variable. A predictor is a variable you're looking at in order to make predictions about other variables, while the values you are trying to predict are known as dependent variables. In this case, life expectancy was the dependent variable.

        * Now, let’s say we get new data on each person’s heart rate as well. Can we create a prediction of life expectancy using both BMI and heart rate?

        * Absolutely! As we saw in the previous video, we can do that using multiple linear regression.

        * If the outcome you want to predict depends on more than one variable, you can make a more complicated model that takes this into account. As long as they're relevant to the situation, using more independent/predictor variables can help you get a better prediction.

        * When there's just one predictor, the linear regression model is a line, but as you add more predictor variables, you're adding more dimensions to the picture.

        * When you have one predictor variable, the equation of the line is

        * y = mx + b

        * and the plot might look something like this:

        * ![multi_linear](./images/multi_linear.png) 

        * Adding a predictor variable to go to two predictor variables means that the predicting equation is:

        * y = m_1 x_1 + m_2 x_2 + b

        * To represent this graphically, we'll need a three-dimensional plot, with the linear regression model represented as a plane:

        * ![multi_linear_1](./images/multi_linear_1.png) 

        * You can use more than two predictor variables - in fact, you should use as many as is useful! If you use nn predictor variables, then the model can be represented by the equation

        * y=m1x1+ m2x2 +m3x3+ ... + mnxn +b

        * As you make a model with more predictor variables, it becomes harder to visualise, but luckily, everything else about linear regression stays the same. We can still fit models and make predictions in exactly the same way - time to try it!

        * ```python
            from sklearn.linear_model import LinearRegression
            from sklearn.datasets import load_boston

            # Load the data from the boston house-prices dataset 
            boston_data = load_boston()
            x = boston_data['data']
            y = boston_data['target']

            # Make and fit the linear regression model
            # TODO: Fit the model and Assign it to the model variable
            model = LinearRegression()
            model.fit(x, y)

            # Make a prediction using the model
            sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                            6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                            1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
            # TODO: Predict housing price for the sample_house
            prediction = model.predict(sample_house)
            ```

    * Linear Regression Warnings

        * Linear regression comes with a set of implicit assumptions and is not the best model for every situation. Here are a couple of issues that you should watch out for.

        * **Linear Regression Works Best When the Data is Linear**

        * Linear regression produces a straight line model from the training data. If the relationship in the training data is not really linear, you'll need to either make adjustments (transform your training data), add features (we'll come to this next), or use another kind of model.

        * ![linear_regression_pitfalls](./images/linear_regression_pitfalls.png) 


        * Linear Regression is Sensitive to Outliers

        * Linear regression tries to find a 'best fit' line among the training data. If your dataset has some outlying extreme values that don't fit a general pattern, they can have a surprisingly large effect.

        * In this first plot, the model fits the data pretty well.

        * ![linear_regression_pitfalls1](./images/linear_regression_pitfalls1.png) 

        * However, adding a few points that are outliers and don't fit the pattern really changes the way the model predicts.

        * ![linear_regression_pitfalls2](./images/linear_regression_pitfalls2.png)

* Polynomial Regression

    * ![polynomial_regression](./images/polynomial_regression.png)

        * In order to represent this line we will need a polynomial equation

        * This will work the same way as the Linear Regression, but with more weights 

        * ![polynomial_regression_1](./images/polynomial_regression_1.png)

        * The algorithm is the same thing, we take the mean absolute error or mean squared error, and take the derivative to the respect of the 4 variables and use gradient descent to modified this 4 weights in order to minimize the error.
    
    * Get some practice implementing polynomial regression in this exercise. In data.csv, you can see data generated for one predictor feature ('Var_X') and one outcome feature ('Var_Y'), following a non-linear trend. Use sklearn's PolynomialFeatures class to extend the predictor feature column into multiple columns with polynomial features. Play around with different degrees of polynomial and the Test Run button to see what fits best: when you think you have the best-fitting degree, press the Submit button to check your work!

    * ```python
        # TODO: Add import statements
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        # Assign the data to predictor and outcome variables
        # TODO: Load the data
        train_data = pd.read_csv('data.csv')
        X = train_data['Var_X'].values.reshape(-1, 1)
        y = train_data['Var_Y'].values

        # Create polynomial features
        # TODO: Create a PolynomialFeatures object, then fit and transform the
        # predictor feature
        poly_feat = PolynomialFeatures(degree = 4)
        X_poly = poly_feat.fit_transform(X)

        # Make and fit the polynomial regression model
        # TODO: Create a LinearRegression object and fit it to the polynomial predictor
        # features
        poly_model = LinearRegression(fit_intercept = False).fit(X_poly, y)
        ```
    
    * ![polynomial_regression_2](./images/polynomial_regression_2.png)

* Regularization

    * This concept works for both regression and classification

    * ![regression_vs_classification](./images/regression_vs_classification.png)

    * ![regression_vs_classification](./images/regression_vs_classification_1.png)

    * If we train a model to minimize error we will elect the one in the right. However, this model is probably over fitting 

    * ![regression_vs_classification](./images/regression_vs_classification_2.png) 

    * The right solution has many more coefficients compared to the left one

    * If we could add those coefficients as the "model complexity" chunk of our error that would be beneficial.

    * ![regression_vs_classification](./images/regression_vs_classification_3.png) 

    * Simple models has a tendency to generalize better, and that is what we want.

    * ![l1_regularization](./images/l1_regularization.png)  

        * L1 takes the coefficients and adds their absolute value to the error.
    
    * ![l1_regularization](./images/l1_regularization_1.png)   

    * L2 regularization is similar, but here we add the squares of the coefficients.

    * ![l2_regularization](./images/l2_regularization.png)
    
    * In order to determine how impactful model complexity is over the error, we introduce a new parameter lambda

    * ![l1_l2_calibration](./images/l1_l2_calibration.png) 

    * ![lambda_regularization](./images/lambda_regularization.png) 

        * Small lambda = ok with more complex models
        * Big lambda = sensitive to complex models
    
    * ![l1_vs_l2](./images/l1_vs_l2.png)

    * ```python
        # TODO: Add import statements
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import Lasso

        # Assign the data to predictor and outcome variables
        # TODO: Load the data
        train_data = pd.read_csv('data.csv', header = None)
        X = train_data.iloc[:,:-1]
        y = train_data.iloc[:,-1]

        # TODO: Create the linear regression model with lasso regularization.
        lasso_reg = Lasso()

        # TODO: Fit the model.
        lasso_reg.fit(X, y)

        # TODO: Retrieve and print out the coefficients from the regression model.
        reg_coef = lasso_reg.coef_
        print(reg_coef)
        ```

* Feature Scaling

    * What is feature scaling? Feature scaling is a way of transforming your data into a common range of values. There are two common scalings:

        * Standardizing
        * Normalizing
    
    * Standardizing

        * Standardizing is completed by taking each value of your column, subtracting the mean of the column, and then dividing by the standard deviation of the column. In Python, let's say you have a column in df called height. You could create a standardized height as:
            * In statistics, the standard deviation is a measure of the amount of variation or dispersion of a set of values.[1] A low standard deviation indicates that the values tend to be close to the mean (also called the expected value) of the set, while a high standard deviation indicates that the values are spread out over a wider range.

        * `df["height_standard"] = (df["height"] - df["height"].mean()) / df["height"].std()`

        * This will create a new "standardized" column where each value is a comparison to the mean of the column, and a new, standardized value can be interpreted as the number of standard deviations the original height was from the mean. This type of feature scaling is by far the most common of all techniques (for the reasons discussed here, but also likely because of precedent).
    

    * Normalizing

        * A second type of feature scaling that is very popular is known as normalizing. With normalizing, data are scaled between 0 and 1. Using the same example as above, we could perform normalizing in Python in the following way:

        * ```python
            df["height_normal"] = (df["height"] - df["height"].min()) / (df["height"].max() - df['height'].min())
            ```
    
    * When Should I Use Feature Scaling?

        * In many machine learning algorithms, the result will change depending on the units of your data. This is especially true in two specific cases:

            * When your algorithm uses a distance-based metric to predict.

            * When you incorporate regularization.

        * Distance Based Metrics

            * In future lessons, you will see one common supervised learning technique that is based on the distance points are from one another called Support Vector Machines (or SVMs). Another technique that involves distance based methods to determine a prediction is k-nearest neighbors (or k-nn). With either of these techniques, choosing not to scale your data may lead to drastically different (and likely misleading) ending predictions.

            * For this reason, choosing some sort of feature scaling is necessary with these distance based techniques.  
        
        * Regularization

            * When you start introducing regularization, you will again want to scale the features of your model. The penalty on particular coefficients in regularized linear regression techniques depends largely on the scale associated with the features. When one feature is on a small range, say from 0 to 10, and another is on a large range, say from 0 to 1 000 000, applying regularization is going to unfairly punish the feature with the small range. Features with small ranges need to have larger coefficients compared to features with large ranges in order to have the same effect on the outcome of the data. (Think about how ab = baab=ba for two numbers aa and bb.) Therefore, if regularization could remove one of those two features with the same net increase in error, it would rather remove the small-ranged feature with the large coefficient, since that would reduce the regularization term the most.

            * Again, this means you will want to scale features any time you are applying regularization.

            * https://www.quora.com/Why-do-we-normalize-the-data

            * A point raised in the article above is that feature scaling can speed up convergence of your machine learning algorithms, which is an important consideration when you scale machine learning applications.

            * ```python
                # TODO: Add import statements
                import numpy as np
                import pandas as pd
                from sklearn.linear_model import Lasso
                from sklearn.preprocessing import StandardScaler

                # Assign the data to predictor and outcome variables
                # TODO: Load the data
                train_data = pd.read_csv('data.csv', header = None)
                X = train_data.iloc[:,:-1]
                y = train_data.iloc[:,-1]

                # TODO: Create the standardization scaling object.
                scaler = StandardScaler()

                # TODO: Fit the standardization parameters and scale the data.
                X_scaled = scaler.fit_transform(X)

                # TODO: Create the linear regression model with lasso regularization.
                lasso_reg = Lasso()

                # TODO: Fit the model.
                lasso_reg.fit(X_scaled, y)

                # TODO: Retrieve and print out the coefficients from the regression model.
                reg_coef = lasso_reg.coef_
                print(reg_coef)
                ```

* Recap

    * In this lesson, you were introduced to linear models. Specifically, you saw:

    * **Gradient descent** as a method to optimize your linear models.
    * **Multiple Linear Regression** as a technique for when you are comparing more than two variables.
    * **Polynomial Regression** for relationships between variables that aren't linear.
    * **Regularization** as a technique to assure that your models will not only fit to the data available, but also extend to new situations.

### Perceptron Algorithm 

* Perceptron Algorithm 

    * For classification (matching values to labels), check the (AI Programming README)[https://github.com/danielmapar/AIProgramming]
        * Neural Networks section

### Decision Tress

* Decision Trees

    * Entropy

        * This is a concept that comes from physics and it measures how much a particle can move.
            * Example: water, gas and solid
            * Solid has low entropy, gas has high entropy
        
        * Entropy also works in probability 

            * ![entropy](./images/entropy.png)
        
        * The more homogeneous (less variaty), the less entropy 

        * ![entropy1](./images/entropy1.png)
            * The more knowledge one has, the less entropy it is
        
        * ![entropy2](./images/entropy2.png)

        * ![entropy3](./images/entropy3.png)

        * ![entropy4](./images/entropy4.png)

        * ![entropy5](./images/entropy5.png)

        * ![entropy6](./images/entropy6.png)

        * ![entropy7](./images/entropy7.png)

        * Instead of multiplying the probabilities, we will take their `log` (it is computationaly cheaper)

        * `log(ab) = log(a) + log(b)`

        * Since the `log` values are negative, we will make sure we take the negative `log` of those probabilities

        * ![entropy8](./images/entropy8.png)

        * 5 times the probability of red balls, plus 3 times the probability of blue balls

        * ![entropy9](./images/entropy9.png)

            * This is a general formula for entropy when we have 2 colors (2 types)
        
        * Last time, you saw this equation for entropy for a bucket with mm red balls and nn blue balls:

            * ![entropy10](./images/entropy10.png) 
        
        * We can state this in terms of probabilities instead for the number of red balls as p_1 and the number of blue balls as p_2:

            * ![entropy11](./images/entropy11.png)

            * ![entropy12](./images/entropy12.png)
        
        * This entropy equation can be extended to the multi-class case, where we have three or more possible values:

            * ![entropy13](./images/entropy13.png)
        
        * The minimum value is still 0, when all elements are of the same value. The maximum value is still achieved when the outcome probabilities are the same, but the upper limit increases with the number of different outcomes. (For example, you can verify the maximum entropy is 2 if there are four different possibilities, each with probability 0.25.)
    
    * Information Gain

        * ![information_gain](./images/information_gain.png) 

        * Information gain = Change in entropy

        * ![information_gain1](./images/information_gain1.png)

        * To calculate Information Gain we first check the entropy of the parent, then we subtracts by the average entropy of the children

        * ![information_gain2](./images/information_gain2.png)

        * ![information_gain3](./images/information_gain3.png)

        * ![information_gain4](./images/information_gain4.png)
            * Calculating entropy for the "app" property 

        * ![information_gain6](./images/information_gain6.png)

        * ![information_gain5](./images/information_gain5.png)

            * The algorithm selects the column with highest information gain, in this case "Question"
        
        * ![information_gain7](./images/information_gain7.png)
            * We can also use decision trees with continuos data
    
    * Hyperparameters for Decision Trees

        * In order to create decision trees that will generalize to new problems well, we can tune a number of different aspects about the trees. We call the different aspects of a decision tree "hyperparameters". These are some of the most important hyperparameters used in decision trees:

        * Maximum Depth

            * The maximum depth of a decision tree is simply the largest possible length between the root to a leaf. A tree of maximum length kk can have at most 2^k2 k leaves.

            * ![decision_tree_hp](./images/decision_tree_hp.png)
        
        * Minimum number of samples to split

            * A node must have at least `min_samples_split` samples in order to be large enough to split. If a node has fewer samples than `min_samples_split` samples, it will not be split, and the splitting process stops.

            * ![decision_tree_split](./images/decision_tree_split.png) 

            * However, `min_samples_split` doesn't control the minimum size of leaves. As you can see in the example on the right, above, the parent node had 20 samples, greater than `min_samples_split = 11`, so the node was split. But when the node was split, a child node was created with that had 5 samples, less than `min_samples_split = 11`.
        
        * Minimum number of samples per leaf

            * When splitting a node, one could run into the problem of having 99 samples in one of them, and 1 on the other. This will not take us too far in our process, and would be a waste of resources and time. If we want to avoid this, we can set a minimum for the number of samples we allow on each leaf.

            * ![decision_tree_split](./images/decision_tree_split2.png) 

            * This number can be specified as an integer or as a float. If it's an integer, it's the minimum number of samples allowed in a leaf. If it's a float, it's the minimum percentage of samples allowed in a leaf. For example, 0.1, or 10%, implies that a particular split will not be allowed if one of the leaves that results contains less than 10% of the samples in the dataset.

            * If a threshold on a feature results in a leaf that has fewer samples than `min_samples_leaf`, the algorithm will not allow that split, but it may perform a split on the same feature at a different threshold, that does satisfy `min_samples_leaf`.

        * Large depth very often causes overfitting, since a tree that is too deep, can memorize the data. Small depth can result in a very simple model, which may cause underfitting.
        
        * Small minimum samples per split may result in a complicated, highly branched tree, which can mean the model has memorized the data, or in other words, overfit. Large minimum samples may result in the tree not having enough flexibility to get built, and may result in underfitting.
    
    * Decision Trees in sklearn

        * ```python
            sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier()
            model.fit(x_values, y_values)
            ```
        
        * In the example above, the `model` variable is a decision tree model that has been fitted to the data `x_values` and `y_values`. Fitting the model means finding the best tree that fits the training data. Let's make two predictions using the model's predict() function.

        * ```python
            print(model.predict([ [0.2, 0.8], [0.5, 0.4] ]))
            [[ 0., 1.]]
            ```
        
        * The model returned an array of predictions, one prediction for each input array. The first input, `[0.2, 0.8]`, got a prediction of `0.`. The second input, `[0.5, 0.4]`, got a prediction of `1.`.
    
    * Hyperparameters

        * When we define the model, we can specify the hyperparameters. In practice, the most common ones are

            * `max_depth`: The maximum number of levels in the tree.
            * `min_samples_leaf`: The minimum number of samples allowed in a leaf.
            * `min_samples_split`: The minimum number of samples required to split an internal node.

        * For example, here we define a model where the maximum depth of the trees `max_depth` is 7, and the minimum number of elements in each leaf `min_samples_leaf` is 10.

            * `model = DecisionTreeClassifier(max_depth = 7, min_samples_leaf = 10)`
    
    * Decision Tree Quiz

        * In this quiz, you'll be given the following sample dataset, and your goal is to define a model that gives 100% accuracy on it.

        * ```python
            # Import statements 
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import accuracy_score
            import pandas as pd
            import numpy as np

            # Read the data.
            data = np.asarray(pd.read_csv('data.csv', header=None))
            # Assign the features to the variable X, and the labels to the variable y. 
            X = data[:,0:2]
            y = data[:,2]

            # TODO: Create the decision tree model and assign it to the variable model.
            model = DecisionTreeClassifier()

            # TODO: Fit the model.
            model.fit(X,y)

            # TODO: Make predictions. Store them in the variable y_pred.
            y_pred = model.predict(X)

            # TODO: Calculate the accuracy and assign it to the variable acc.
            acc = accuracy_score(y, y_pred)
            ```

        * Note: This quiz requires you to find an accuracy of 100% on the training set. This is like memorizing the training data! A model designed to have 100% accuracy on training data is unlikely to generalize well to new data. If you pick very large values for your parameters, the model will fit the training set very well, but may not generalize well. Try to find the smallest possible parameters that do the job—then the model will be more likely to generalize well. (This aspect of the exercise won't be graded.)