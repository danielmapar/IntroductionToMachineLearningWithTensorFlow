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


