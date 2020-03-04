#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




def check_purity(data):

    '''
    looks if there is only one class
    '''

    label = data[:,-1]
    #returns array of our unique labels
    unique_classes = np.unique(label)

    if len(unique_classes)==1:
        return True
    else:
        return False


# In[86]:


def determine_type_of_feature(data):
    feature_types = []
    n_unique_values_threshold = 15
    for column in data.columns:
        unique_vals = data[column].unique()
        example_value = unique_vals[0]

        if type(example_value) == str or (len(unique_vals)<= n_unique_values_threshold):
            feature_types.append("categorical")
        else:
            feature_types.append("continuous")
    return feature_types


# ## Create Leaf

# In[87]:


def calculate_mse(data):
    actual_values = data[:, -1]
    if len(actual_values) == 0:
        mse=0
    else:
        prediction = np.mean(actual_values)
        mse = np.mean((actual_values - prediction) **2)

    return mse


def classify_data(data):

    #outputs the majority class of the dataset

    label = data[:,-1]
    unique_classes, counts_of_unique_classes = np.unique(label,return_counts=True)

    index = counts_of_unique_classes.argmax()

    classification = unique_classes[index]





    return classification


# In[88]:


def get_potential_splits(data, random_subspace):

    potential_splits = {}

    _, n_columns = data.shape

    column_indices = list(range(n_columns - 1))

    if random_subspace and random_subspace <= len(column_indices):

        column_indices = random.sample(population=column_indices, k = random_subspace)


    for column_index in column_indices:

        values = data[:, column_index]

        univalues = np.unique(values)

        potential_splits[column_index] = univalues


    return potential_splits



# In[89]:


def split_data(data, split_column, split_value):



    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == 'continuous':

        data_below = data[split_column_values <=split_value]
        data_above = data[split_column_values >split_value]

    else:
        data_below = data[split_column_values ==split_value]
        data_above = data[split_column_values !=split_value]


    return data_below, data_above


# In[90]:


def create_leaf(data,ml_task):

    #outputs the majority class of the dataset

    label = data[:,-1]

    if ml_task == 'regression':

        leaf = np.mean(label)

    #classification
    elif ml_task == 'classification':


        unique_classes, counts_of_unique_classes = np.unique(label,return_counts=True)

        index = counts_of_unique_classes.argmax()

        leaf = unique_classes[index]





    return leaf


# ## Determine best split

# In[91]:


def calculate_mse(data):
    actual_values = data[:, -1]
    if len(actual_values) == 0:
        mse=0
    else:
        prediction = np.mean(actual_values)
        mse = np.mean((actual_values - prediction) **2)

    return mse


# In[92]:


def calculate_entropy(data):
    label_column = data[:,-1]
    ##determine probabilities of the classes

    #count up the number of samples in each label
    _,counts = np.unique(label_column,return_counts=True)

    #convert the count to the probability of a value falling into a certain class label
    probabilities = counts/counts.sum()


    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


# In[93]:


def calculate_overall_error_metric(data_below,data_above, metric_function):


    n_data_points = len(data_below) + len(data_above)


    #samples below the line
    p_data_below = len(data_below)/n_data_points

    #samples above the line
    p_data_above = len(data_above)/n_data_points




    overall_metric = (p_data_below * metric_function(data_below)

                      +p_data_above * metric_function(data_above))




    return overall_metric



# In[94]:


def determine_best_split(data, potential_splits, ml_task):

    first_iteration = True

    for column_index in potential_splits:

        #print(COLUMN_HEADERS[column_index], "-", len(np.unique(data[:, column_index])))

        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data,split_column=column_index,split_value=value)

            if ml_task == 'regression':

                current_overall_metric = calculate_overall_error_metric(data_below,data_above, metric_function=calculate_mse)


            #classification
            else:
                current_overall_metric = calculate_overall_error_metric(data_below,data_above, metric_function=calculate_entropy)


            if first_iteration or current_overall_metric <= best_overall_metric:

                first_iteration = False


                best_overall_metric=current_overall_metric
                best_split_column = column_index
                best_split_value = value


    return best_split_column,best_split_value


# In[97]:


def DecisionTreeAlgo(df,ml_task, counter=0, min_samples=2, max_depth=5, random_subspace=None):



    """
    minimum sample size: the minimum number of sampels a node must contain in
    order to consider splitting.
    """


    # data preparations

    if counter == 0:

        global COLUMN_HEADERS, FEATURE_TYPES


        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)

        data = df.values
    else:

        data = df


    # base case for recusive function
    if (check_purity(data)) or (len(data)< min_samples) or (counter==max_depth):
        #return our prediction
        leaf = create_leaf(data,ml_task)
        return leaf


    #recursive section

    else:

        counter += 1


        # run helper functions

        potential_splits = get_potential_splits(data)


        #find lowest overall entropy

        split_column, split_value = determine_best_split(data, potential_splits,ml_task)


        data_below, data_above = split_data(data, split_column, split_value)


        #check for empty data
        if len(data_below) == 0 or len(data_above)==0:
            leaf = create_leaf(data,ml_task)
            return leaf


        #instantiate sub-tree
        feature_name = COLUMN_HEADERS[split_column]

        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == 'continuous':
            question  = "{} <= {}".format(feature_name,split_value)

        else:

            question  = "{} == {}".format(feature_name,split_value)


        sub_tree = {question: []}


        #find answers

        yes_answer = DecisionTreeAlgo(data_below, ml_task, counter, min_samples,max_depth,random_subspace)


        no_answer = DecisionTreeAlgo(data_above, ml_task, counter, min_samples,max_depth,random_subspace)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:


            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)


        return sub_tree




# In[100]:





# In[102]:


def predict(example, tree):
    question = list(tree.keys())[0]

    feature_name, comparison, value = question.split()

    # ask question

    if comparison == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]



    #base case
    if not isinstance(answer, dict):
        return(answer)
    else:

        residual_tree = answer
        return predict(example,residual_tree)


# ## HYPYERPARAM TUNER

# In[108]:


def calculate_adjr_squared(data,tree):

    '''
    adjusted r square penalizes for adding independent variables that do not fit the model

    '''


    labels = data.iloc[:,-1]
    mean = labels.mean()

    predictions = data.apply(predict, args=(tree,), axis=1)

    ss_res = sum((labels - predictions) **2)
    ss_tot = sum((labels - mean) ** 2)
    r_squared = 1 - (ss_res /ss_tot)

    n = data.shape[0]
    k = data.iloc[:-1].shape[1]

    adjusted_r_squared = 1 - ((1-r_squared) *(n-1) / (n - k - 1))
    return adjusted_r_squared


# In[107]:




# In[110]:
'''
def gridsearch(train_df, val_df):
	grid_search = {'max_depth':[], 'min_samples':[], 'adjusted_r_square_train':[], 'adjusted_r_sqaure_val':[]}

	for max_depth in range(2,11):
    	for min_samples in range(5, 30, 5):
      		tree  = DecisionTreeAlgo(train_df,ml_task='regression',max_depth=max_depth, min_samples=min_samples)

        	adjr_train = calculate_adjr_squared(train_df,tree)
        	adjr_val = calculate_adjr_squared(val_df,tree)

        	grid_search['max_depth'].append(max_depth)
        	grid_search['min_samples'].append(min_samples)


        	grid_search['adjusted_r_square_train'].append(adjr_train)
        	grid_search['adjusted_r_sqaure_val'].append(adjr_val)


	grid_search = pd.DataFrame(grid_search)
	return grid_search.sort_values('adjusted_r_sqaure_val',ascending=False)
'''
