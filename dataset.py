import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
import sys, os
from collections import deque
from .database_util import formatFilter, formatJoin, TreeNode, filterDict2Hist
from .database_util import *

class PlanTreeDataset(Dataset):
    def __init__(self, json_df: pd.DataFrame, train: pd.DataFrame, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample):
        self.table_sample = table_sample
        self.encoding = encoding
        self.hist_file = hist_file
        self.length = len(json_df)

        # Parse the query plans
        nodes = [plan['Plan'] for plan in json_df['json']]
        self.cards = [node.get('Actual Rows', 0) for node in nodes]
        self.costs = [plan.get('Execution Time', 0) for plan in json_df['json']]
        self.knob_settings = [self.extract_knob_setting(node) for node in nodes]
        
        knob_tensors = [torch.tensor(k, dtype=torch.float32) for k in self.knob_settings]
        self.knob_labels = torch.nn.utils.rnn.pad_sequence(knob_tensors, batch_first=True, padding_value=0)
        # Normalize labels (cost and knobs)
        # self.knob_labels=torch.from_numpy(.normalize_labels(self.knob_settings))
        self.card_labels = torch.from_numpy(card_norm.normalize_labels(self.cards))
        self.cost_labels = torch.from_numpy(cost_norm.normalize_labels(self.costs))
       
        print(f'total execution time is : {self.costs}')
        # Set target variable
        self.to_predict = to_predict
        if to_predict == 'cost':
            self.gts = self.costs
            self.labels = self.cost_labels
        elif to_predict == 'knobs':
            self.gts = self.knob_settings
            self.labels = self.knob_labels
        elif to_predict == 'both':
            self.gts = {'cost': self.costs, 'knobs': self.knob_settings}
            self.labels = {'cost': self.cost_labels, 'knobs': self.knob_labels}
        else:
            raise Exception('Unknown to_predict type')
       
        idxs = list(json_df['id'])
        self.treeNodes = []
        self.collated_dicts = [self.js_node2dict(i, node) for i, node in zip(idxs, nodes)]


    def extract_knob_setting(self, plan):
        """
        Extracts knob settings (e.g., 'Seq Scan', 'Hash Join') from the execution plan and sub-plans.
        """
        knobs = []

        # Collect node types
        if 'Node Type' in plan:
            knobs.append(plan['Node Type'])

        # Recursively collect from subplans
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                knobs.extend(self.extract_knob_setting(subplan))

        # Ensure encoding supports lists
        encoded_knobs = [self.encoding.encode_type(node) for node in knobs]

        # Pad sequences to a fixed length
        max_length = 10  # Adjust based on dataset analysis
        padded_knobs = encoded_knobs + [0] * (max_length - len(encoded_knobs))

        return padded_knobs  # Now returns a fixed-length encoded list

    
    
    # def extract_knob_setting(self, plan):
    #     """
    #     Extracts the knob settings (e.g., 'Seq Scan', 'Hash Join') from the plan.
    #     """
    #     node_type = plan['Node Type']
    #     return self.encoding.encode_type(node_type)  # Encode knob settings as numerical values
# ----------------------------------------------------------------
#  Converts a query plan node to a 
# dictionary of features and collates it into the final format.
# 1: Use traversePlan to create a TreeNode structure.
# 2:  Use node2dict to get the features, adjacency list, and other properties of the node.
# 3:Use pre_collate to apply any necessary padding and prepare the node data for use in the model.
#4:  Clear the tree nodes to free memory.
    def js_node2dict(self, idx, node):
        treeNode = self.traversePlan(node, idx, self.encoding)
        _dict = self.node2dict(treeNode)
        collated_dict = self.pre_collate(_dict)
        self.treeNodes.clear()
        del self.treeNodes[:]
        return collated_dict

# Returns the length of the dataset (i.e., how many query plans are present in the dataset).
    def __len__(self):
        return self.length

# Retrieves the features and corresponding labels for a particular query plan in the dataset.
# Depending on the to_predict variable (which could be 'cost', 'knobs', or 'both'), 
# this function returns either:
# Only the cost label.
# Only the knob settings.
# Both cost and knob labels.
    def __getitem__(self, idx):
        # Return query features and the target (cost and knob settings)
        if self.to_predict == 'both':
            return self.collated_dicts[idx], (self.cost_labels[idx], self.knob_labels[idx])
        elif self.to_predict == 'cost':
            return self.collated_dicts[idx], self.cost_labels[idx]
        elif self.to_predict == 'knobs':
            return self.collated_dicts[idx], self.knob_labels[idx]
        else:
            raise Exception('Unknown to_predict type')


# Prepares the query features and converts them into a format suitable for input into the model.
# Padding: The function ensures that the feature vector has a consistent size (max_node).
# Attention Bias: This applies bias for attention mechanisms in models, which helps in reducing unnecessary computation.
# Adjacency List and Relational Position: Create the adjacency matrix and calculate the relative positions of nodes in the query plan’s tree structure.
# Return Formatted Data: Return a dictionary containing the processed features (x), attention bias, relative positions, and node heights.
   
    def pre_collate(self, the_dict, max_node=30, rel_pos_max=20):
        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)

        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N, N], dtype=torch.bool)
            adj[edge_index[0, :], edge_index[1, :]] = True
            shortest_path_result = floyd_warshall_rewrite(adj.numpy())

        rel_pos = torch.from_numpy(shortest_path_result).long()
        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')

        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)
        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)

        return {
            'x': x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }
    
# Converts a TreeNode object (a node in the query plan tree) into a dictionary of features.
# Topological Sorting: Perform a topological sort of the nodes to capture the parent-child relationships between the nodes.
# Feature and Heights: Store the features of the node, and compute the height (depth) of the node in the tree structure.
# Return Dictionary: Return a dictionary with features, heights, and adjacency_list.
    def node2dict(self, treeNode):
        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))
        return {
            'features': torch.FloatTensor(np.array(features)),
            'heights': torch.LongTensor(heights),
            'adjacency_list': torch.LongTensor(np.array(adj_list)),
        }
# Performs a topological sort of the nodes in the query plan tree.
# Traverse the query plan in a breadth-first manner, appending each node and its children to the adjacency list.
# Features: Collect the features of each node in the query tree.
# Return the adjacency list, the number of children for each node, and the features of the nodes.
    def topo_sort(self, root_node):
        adj_list = []
        num_child = []
        features = []
        toVisit = deque()
        toVisit.append((0, root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id, child))
                adj_list.append((idx, next_id))
                next_id += 1
        return adj_list, num_child, features

# Recursively traverses through the nodes of a query plan and creates a tree structure of TreeNode objects.
# Node Type: Encode the node type (Node Type) using the provided encoding.
# Filters and Joins: Process the filters and join types, using helper functions (formatFilter, formatJoin) to extract relevant information.
# TreeNode Creation: Create TreeNode objects for each node in the plan and connect them to form a tree.
# Recursion: For each subplan (if present), recurse through the plan and add children to the parent node.
    def traversePlan(self, plan, idx, encoding):
        nodeType = plan['Node Type']
        typeId = encoding.encode_type(nodeType)
        filters, alias = formatFilter(plan)
        join = formatJoin(plan)
        joinId = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias)

        root = TreeNode(nodeType, typeId, filters, None, joinId, join, filters_encoded)
        self.treeNodes.append(root)

        if 'Relation Name' in plan:
            root.table = plan['Relation Name']
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx

        root.feature = node2feature(root, encoding, self.hist_file, self.table_sample)
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding)
                node.parent = root
                root.addChild(node)
        return root

# Calculates the height (or depth) of the nodes in the tree structure based on the adjacency list.
# Breadth-First Search: Perform a BFS to evaluate the height of each node in the query plan tree.
# Node Order: Assign a height to each node based on its position in the tree (starting from the root).
# Return Heights: Return the node heights as an array.
    def calculate_height(self, adj_list, tree_size):
        if tree_size == 1:
            return np.array([0])
        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)
        parent_nodes = adj_list[:, 0]
        child_nodes = adj_list[:, 1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]
            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order

# Extracts the features for a specific query plan node and converts it into a feature vector.
# Filters: Extract filter conditions (columns, operations, values) and pad them to ensure consistency.
# Join Type: Include information about the type of join used by the node.
# Histograms: Retrieve any relevant histogram data using filterDict2Hist.
# Table Sample: If available, include a sample of the table related to the node.
# Return Feature Vector: Concatenate all the extracted features into a single vector that can be used as input for a machine learning model.
def node2feature(node, encoding, hist_file, table_sample):
    num_filter = len(node.filterDict['colId'])
    pad = np.zeros((3, 3 - num_filter))
    filts = np.array(list(node.filterDict.values()))  # cols, ops, vals
    filts = np.concatenate((filts, pad), axis=1).flatten()
    mask = np.zeros(3)
    mask[:num_filter] = 1
    type_join = np.array([node.typeId, node.join])
    hists = filterDict2Hist(hist_file, node.filterDict, encoding)

    table = np.array([node.table_id])
    if node.table_id == 0:
        sample = np.zeros(1000)
    else:
        sample = table_sample[node.query_id][node.table]

    return np.concatenate((type_join, filts, mask, hists, table, sample))

