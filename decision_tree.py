__author__ = "Uddesh Karda"
"FIS project 2 Decision tree problem"


import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from scipy.stats import norm
import random
import pandas as pd
import math
import sys

# 0 - Iris-setosa, 1 - Iris-versicolor ,2 - Iris-virginica

attributes = ["sepal_length","sepal_width","petal_length","petal_width"]
Classes = ["Iris-setosa","Iris-versicolor","Iris-virginica"]
data = []
count = {}


def main():
    """
    Main function initiates other functions and prints accuracy
    :return: -
    """
    random_baseline_count = 0
    k = int(input("Enter number of rows in test data : " + "\n"))
    loadData()
    for i in range(len(data)):
        for j in range(len(data[i]) - 1):
            data[i][j] = float(data[i][j])
    train,test = split_into_train_test(data)
    tree = tree_builder(train,3,1)
    correct = 0
    rand_class = random.randrange(0, len(Classes))
    for i in range(len(test)):
        answer = predictor(tree, test[i])
        actual = test[i][-1]
        if actual == answer:
            correct += 1
        if rand_class == actual:
            random_baseline_count += 1
    print("random baseline = " + str((random_baseline_count / len(test)) * 100) + '%' + "\n")
    print("accuracy = " + str((correct/len(test))*100) + "%" + "\n")
    max_count = 0
    max_key = []
    for key, value in count.items():
        if value >= max_count:
            max_key.append(key)
    print("Classes with majority frequency are as follows : ")
    for i in max_key:
        print(i, str(((count.get(i)/sum(list(count.values())))*100)) + '%')


def predictor(node, row):
    """
    Takes input a node(root) and a row and then predicts class of row using the tree node
    :param node: node of decision tree
    :param row: row of data from test set
    :return: Recursive call or decision
    """
    if row[node['index']] < node['value']:
        if type(node['left']) == dict:
            return predictor(node['left'], row)
        else:
            return node['left']
    else:
        if type(node['right']) == dict:
            return predictor(node['right'], row)
        else:
            return node['right']


def split_into_train_test(data):
    """
    Splits the whole data set into training and testing sets
    :param data: All rows of iris data set
    :return: training and testing dataset lists
    """
    randlist = []
    train = []
    test = []
    for i in range(0,15):
        randlist.append(random.randint(0,50))
    for i in range(0,15):
        randlist.append(random.randint(50,100))
    for i in range(0,15):
        randlist.append(random.randint(100,150))
    for i in range(len(data)):
        if data[i] == '\n':
            continue
        if i in randlist:
            train.append(data[i])
        else:
            test.append(data[i])
    return train,test


def leaf_nodes_helper(group):
    """
    Helps in tree building
    :param group: list of rows
    :return: value
    """
    answers = set()
    for row in group:
        answers.add(row[-1])
    return max(answers)


def tree_builder(train, max_depth, min_size):
    """
    INitialises the tree building process
    :param train: list of training data rows
    :param max_depth: The maximum depth of tree
    :param min_size: The minimum size of the tree
    :return: Root node of the tree
    """
    root_node = get_split(train)
    split_data(root_node, max_depth, min_size, 1)
    return root_node


def split_data(node, max_depth, min_size, depth):
    """
    Checks tree constraints and splits the data into seperate nodes
    :param node: The current node of the tree being split
    :param max_depth: The max depth of the tree
    :param min_size: The minimum size allowed
    :param depth: The current depth of the tree
    :return:
    """
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] =  leaf_nodes_helper(left + right)
        node['right'] = leaf_nodes_helper(left + right)
        return
    if depth >= max_depth:
        node['left'] = leaf_nodes_helper(left)
        node['right'] =  leaf_nodes_helper(right)
        return
    if len(left) <= min_size:
        node['left'] = leaf_nodes_helper(left)
    else:
        node['left'] = get_split(left)
        split_data(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node['right'] = leaf_nodes_helper(right)
    else:
        node['right'] = get_split(right)
        split_data(node['right'], max_depth, min_size, depth + 1)


def get_split(dataList):
    """
    Tests the created splits on their gini index score
    :param dataList: list of data to be split
    :return: A dictionary with split details
    """
    group_dict = {}
    class_vals = set()
    for row in dataList:
        class_vals.add(row[-1])
    class_vals = list(class_vals)
    g_index,g_value,g_score,g_groups = sys.maxsize,sys.maxsize,sys.maxsize,None
    for index in range(len(dataList[0]) - 1):
        for row in dataList:
            groups = test_split(index, row[index], dataList)
            gini = gini_index(groups, class_vals)
            if gini < g_score:
                g_index, g_value, g_score, g_groups = index, row[index], gini, groups
    group_dict['index'] = g_index
    group_dict['value'] = g_value
    group_dict['groups'] = g_groups
    return group_dict


def gini_index(groups, classes):
    """
    Calculates the gini index of a split
    :param groups: The split groups for a set of data
    :param classes: List of all the classes in the problem
    :return: gini index score
    """
    count_n = float(0)
    for group in groups:
        count_n += len(group)
    gini = float(0)
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        count = 0
        score = float(0)
        for class_value in classes:
            for row in group:
                if class_value == row[-1]:
                    count += 1
            proportion = count/size
            score += proportion * proportion
        gini += (float(1) - score) * (size/count_n)
    return gini


def test_split(index, value, dataList):
    """
    Splits a given dataset into two grpups, left and right
    :param index: The attribute index
    :param value: The splitter value
    :param dataList: The data set to be split
    :return: Two lists(groups) of data
    """
    left, right = [],[]
    for row in dataList:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def loadData():
    """
    Load the data from the given file
    :return: -
    """
    f = open('iris.data')
    line = f.readline()
    for i in range(len(Classes)):
        count[Classes[i]] = 0
    while line != '\n':
        line_vals = line.split(',')
        line_vals[4] = line_vals[4].rstrip()
        if 'Iris-setosa' in line:
            del line_vals[4]
            line_vals.append(0)
            data.append(line_vals)
            count['Iris-setosa'] = count.get('Iris-setosa') + 1
        elif 'Iris-versicolor' in line:
            del line_vals[4]
            line_vals.append(1)
            data.append(line_vals)
            count['Iris-versicolor'] = count.get('Iris-versicolor') + 1
        else:
            del line_vals[4]
            line_vals.append(2)
            data.append(line_vals)
            count['Iris-virginica'] = count.get('Iris-virginica') + 1
        line = f.readline()


if __name__ == '__main__':
    main()