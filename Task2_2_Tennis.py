import pandas as pd
import matplotlib.pyplot as plt
import math
import networkx as nx
from graphviz import Digraph


# the class to store the information of a node in a decision tree
class node:
    # if the node is an attribute node, we should set the value for all the fields except label
    # if the node is the leaf node, we should only set the value for label
    ig = None
    attribute = ''
    possible_values = {}
    child_nodes = {}
    label = None

    def __init__(self, attr, pv, cn):
        self.attribute = attr
        self.possible_values = pv
        self.child_nodes = cn


# 1 input the data set S
def compute_entropy(data):
    # 2 count the N_pos and N_neg by count_observation()
    # 3 compute entropy using that N_pos and N_neg
    return entropy(count_observation(data))


# input data, find the each number of positive and negative observations
def count_observation(data):
    yes = 0
    no = 0

    for i in range(0, len(data)):
        if data[i] == 'Yes':
            yes = yes + 1
        else:
            no = no + 1

    return {'n_pos': yes, 'n_neg': no}


# input example: {'n_pos':10,'n_neg'=10}
def entropy(list_count):
    yes = list_count['n_pos']
    no = list_count['n_neg']

    if yes == 0 or no == 0:
        return 0
    else:

        sum = yes + no
        p_y = yes / sum
        p_n = no / sum
        e_s = -1 * p_y * math.log2(p_y) - p_n * math.log2(p_n)

        return e_s


# retrieve data that are needed according to requirements
def get_data_by_attribute(data_set, attribute_index):
    attribute_target = []
    for l in data_set:
        attribute_target.append([l[attribute_index], l[-1]])
    return attribute_target


# retrieve data that are needed according to requirements
def get_data_by_target(data_set):
    target = []
    for l in data_set:
        target.append(l[-1])
    return target


# retrieve data that are needed according to requirements
def get_possible_values(list_data):
    possible_values = []
    for value in list_data:
        possible_values.append(value[0])
    return set(possible_values)


# retrieve data that are needed according to requirements
def get_dataset_by_value(data_set, attribute_index, value):
    new_data_set = []
    for l in data_set:
        if l[attribute_index] == value:
            new_data_set.append(l)

    return new_data_set


# input the data set S, and attribute index number for attribute A in that data set
# for example, the attribute list for data set S is ['Outlook','Temperature', 'Humidity','Wind]
# if we want to compute the IG for 'Outlook', just input attribute_index = 0
def information_gain(data_set, attribute_index):
    # compute |S|
    no_sum = len(data_set)

    attribute_target = get_data_by_attribute(data_set, attribute_index)
    possible_values = []
    dict = {}

    # find the subset S_u_i of S where A = u_i
    for l in attribute_target:
        value = l[0]

        possible_values.append(value)
        if value not in dict.keys():
            dict[value] = [l[-1]]
        else:
            dict[value].append(l[-1])

    # find all the possible value u_i for an attribute A
    possible_values = set(possible_values)

    e_s_a = 0

    # compute the sum of |S_u_i| / |S| * Entropy(S_u_i)
    # where A = u_i
    for value in possible_values:
        # compute |S_u_i|
        no_key = len(dict[value])
        # e_s_a = e_s_a + (no_key / no_sum) * entropy(count_observation(dict[value]))
        e_s_a = e_s_a + (no_key / no_sum) * compute_entropy(dict[value])

    # compute the entropy for set S
    # e_s = entropy(count_observation(get_data_by_target(data_set)))
    e_s = compute_entropy(get_data_by_target(data_set))
    gain_s_a = e_s - e_s_a

    return gain_s_a


# extract the attributes from a dataframe, extract the data set from a datagram
# the attributes are put into a list, for example ['Outlook','Temperature', 'Humidity','Wind]
# the data set is a dict, one row in the dataframe is in a dict,
# for example [['Sunny','Hot','High','Weak','No'], ['Sunny','Hot','High','Strong','No',......]
def rearrange_data(data, target):
    dataSet = []
    attributes = []

    for key in data.keys():
        row = []
        for k in data[key].keys():
            if k not in attributes:
                attributes.append(k)
            row.append(data[key][k])
        dataSet.append(row)
    attributes.remove(target)

    return dataSet, attributes


# input the data set S, and the list of the index for attributes that you want to find the best attribute
# for each attribute, compute its IG, and find the maximum one
def choose_best_attribute(data_set, list_attribute_index):
    dict_ig = {}
    max_ig = 0
    for i in list_attribute_index:
        ig = information_gain(data_set, i)
        dict_ig[ig] = i
        if ig > max_ig:
            max_ig = ig

    return dict_ig[max_ig], max_ig


# ID3 algorithm
# input data set, its format is dict
# for example [['Sunny','Hot','High','Weak','No'], ['Sunny','Hot','High','Strong','No',......]
# input the list of the index for the attributes,
# for example [0, 1, 2, 3] indicate the index in  ['Outlook','Temperature', 'Humidity','Wind]
# The inputs are obtained from the function rearrange_data()
def id3(data_set, list_attribute_index):
    value_target = get_data_by_target(data_set)
    dict_y_n = count_observation(value_target)

    # if all Observations are class +1
    if dict_y_n['n_neg'] == 0:
        n = node('', [], {})
        n.label = 'Yes'

        return n

    # if all Observations are class -1
    if dict_y_n['n_pos'] == 0:
        n = node('', [], {})
        n.label = 'NO'

        return n

    # if Attributes is empty
    if len(list_attribute_index) == 0:

        # return the single node tree Root with the label the most common value in Targets
        if dict_y_n['n_pos'] >= dict_y_n['n_neg']:
            n = node('', [], {})
            n.label = 'Yes'

            return n
        else:
            n = node('', [], {})
            n.label = 'NO'

            return n
    else:

        # find the best attribute A from Attributes with highest Information Gain
        [best_attribute_index, best_ig] = choose_best_attribute(data_set, list_attribute_index)

        # find the possible values u_i for the best attribute A
        possible_values = get_possible_values(get_data_by_attribute(data_set, best_attribute_index))

        # set A as the Root
        root = node(attributes[best_attribute_index], possible_values, {})
        root.ig = best_ig

        for value in possible_values:

            # find the subset S_u_i where A = u_i
            new_data_set = get_dataset_by_value(data_set, best_attribute_index, value)  # ['sunny','sunny']

            # if the subset S_u_i where A = u_i is empty
            if len(new_data_set) == 0:

                child_node = node('', [], {})
                dict_y_n = count_observation(get_data_by_target(new_data_set))

                # find the most common value in Targets, set it as the label of the leaf node
                if dict_y_n['n_pos'] >= dict_y_n['n_neg']:

                    child_node.label = 'Yes'
                    root.child_nodes[value] = child_node
                else:
                    child_node.label = 'NO'
                    root.child_nodes[value] = child_node

            else:
                # compute Attributes - {A}
                new_list_attribute_index = []

                for a in list_attribute_index:
                    if a != best_attribute_index:
                        new_list_attribute_index.append(a)

                root.child_nodes[value] = id3(new_data_set, new_list_attribute_index)

        return root


# recursively traverse the tree to draw decision tree diagram
def nodes_to_edge(root, root_name):
    global index

    l_root_name = root_name

    if root.label is not None:

        return
    else:

        child_nodes = root.child_nodes
        for key in child_nodes.keys():
            node_kid = child_nodes[key]

            l_key = str(key) + str(index)
            index = index + 1

            dot.node(l_key, str(key), shape='square')
            dot.edge(l_root_name, l_key)

            if node_kid.label is not None:

                l_kid = node_kid.label + str(index)
                index = index + 1

                dot.node(l_kid, node_kid.label, shape='plaintext')

                dot.edge(l_key, l_kid)



            else:
                kid_name = node_kid.attribute + " I.G = " + str(round(node_kid.ig, 4))

                l_kid = kid_name + str(index)
                index = index + 1

                dot.node(l_kid, kid_name)

                dot.edge(l_key, l_kid)

                nodes_to_edge(node_kid, l_kid)


# read csv data for Tennis data set
csv_data = pd.read_csv('Tennis.csv')
data = csv_data.drop(columns=['Day'])
data = data.to_dict('index')
# set the target name
target = 'PlayTennis'

# rearrange the data set for later processing
(data_set, attributes) = rearrange_data(data, 'PlayTennis')

# Invoke the function for Task 2.2.2, print the result, find the best attribute to become the root
for i in range(0, 4):
    print("Gain(S, ", attributes[i], "): ", information_gain(data_set, i))

(root_attribute, root_ig) = choose_best_attribute(data_set, [0, 1, 2, 3])

print('The root is ', attributes[root_attribute], '. The I.G is ', root_ig)
print('==============================')

print('Generating Decision Tree...')
t = id3(data_set, [0, 1, 2, 3])

index = 0
root_name = t.attribute + ' I.G= ' + str(round(t.ig, 4))
dot = Digraph(format='png')
nodes_to_edge(t, root_name)
print(dot.source)

dot.render('tennis', view=True)
