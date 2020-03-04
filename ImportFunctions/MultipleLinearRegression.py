#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:33:18 2020

@author: Matt
"""

import numpy as np
import pandas as pd


def addConstantFunc(X):
    
    x0 = np.ones(len(X))
    
    #add a constant
    X['constant'] = x0
    
    return X


def calculate_cost_function(X,y, coefficients):
    
    
    
    #add a constant
    X =  addConstantFunc(X)
        
    cost = np.sum((X.dot(coefficients) - y)**2)/ (2*len(y)) 
    
    return cost
    

def LinearRegression(X,y, alpha, n_iterations, step_loss = True):
    
    """
    if step loss is true, the function returns the gradient descent output
    
    
    """
    X = addConstantFunc(X)
        
    gradient_preds = []
    
    #create base intercept
    coefficients = np.array(np.zeros(X.shape[1]))
    
    cost_history = [0] * n_iterations
    
    for i in range(n_iterations):
        
        h = X.dot(coefficients)
        
        loss = h - y
        
        gradient = X.T.dot(loss)/ len(y)
        
        coefficients = coefficients - alpha  * gradient
        
        cost = calculate_cost_function(X,y, coefficients)
        
        gradient_preds.append(tuple([i, cost]))
        
        cost_history[i] = cost
        
        steploss = pd.DataFrame(gradient_preds,columns=['Steps','Loss'])
    
    
        
    if step_loss==True:
        return (coefficients, steploss)

    else:
        return coefficients
    
    
def predict(X, coefficients):
    
    X = addConstantFunc(X)
    prediction = X.dot(coefficients)
    return prediction
    
    
    
        