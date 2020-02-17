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

    * Square Trick 

        * 


