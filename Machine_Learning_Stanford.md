## Notes frome the Stanford Machine Learning Course – Led by Andrew Ng ##

### Week 1: Lesson 1 ###

**Introduction:** What is machine learning?

Grew out of work with AI

Examples of when it's used:

1. Database Mining: Because of the growth of automation (and the internet) we now have massive datasets, rich in information. A few examples: web-click data fron the internet, scientific data: astronomy and genomic datasets, medical record. Basically the data all around us.

2. Applications that can't be programmed: e.g. autonomous helicopters taht need to train themselves how to fly, handwriting recognition, nearly all Natural Language Processing (NLP) applications, computer vision - identification.

3. Self-customising programs: eg. Amazon, Netflix or supermarket product recommendations.

Two definitions of Machine Learning are offered. Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.

Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

Example: playing checkers.
E = the experience of playing many games of checkers
T = the task of playing checkers.
P = the probability that the program will win the next game.
In general, any machine learning problem can be assigned to one of two broad classifications: Supervised learning and Unsupervised learning.

## Supervised Learning ##

E.g. Housing Prices. Given a set of past data, a basic algorithm night just extrapolate two parameters of data using linear fitting (regression fitting). If there are external factors that influence the data, then the function to fit may be quite complicated. A different algorithm may model the function with a higher order polynomial. Basically this is a **regression problem**.

Another example might be breast cancer tumour data. Assume you have data of both malignant and benign tumours, where both classes have a range of tumour sizes. Can you determine the probability of malignant or benign based on size? This is ** classification problem**. Binomial-looking datasets may actually be based on N > 2 classes. This becomes more difficult. More parameters (e.g include age) is likely to help an algorithm separate the classes. Many or an infinite number of attributes (parameters) is often what you need to develop a good algorithm. In astronomy, much of our data analysis can be separated into regression or classification.


Question: Scenario 1: You have a large inventory of identical items, you want to know how many items will sell over the next three months.
Scenario 2: You'd like software to ezamine individual customer accounts,  and decide if each has been hacked/compromised.
S! is a regression problem, S2 is a classification problem (.e bimary states)

**Summary:**
Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.

Example 1: Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem. We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories.

Example 2: (a) Regression - Given a picture of a person, we have to predict their age on the basis of the given picture (b) Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.

For labelled data you would normally implemet supervised learning.

## Unsupervised Learning ##

Is when you have data and you need to classify it without prior knowledge of the parameters. In this case an appropriate ML algorithm would be a clustering algorithm. THis is essentially what we do in astronomy with color-magnitude or color-color diagrams. We look for structure or clustering. This type of algorithm is used for genetic data, social network analysis, market segmentation/demographics, and of course astronomy.

Andrew tehn talks about the cocktail party problem: two people at a party with two microphones placed in a different parts of the room. Essentially you look at all combinations in the data. This is kind of like prinipal component analysis. There is a cocktail party problem algorithm: **[W,s,v]=svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x; **  *https://en.wikipedia.org/wiki/Source_separation This essentially a source separation problem. 






