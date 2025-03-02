import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
import sys, os
from collections import deque
from .utils import formatFilter, formatJoin, TreeNode, filterDict2Hist
from .utils import *
from torch.nn.utils.rnn import pad_sequence

class EncodeDataset(Dataset):
    def __init__(self, json_df: pd.DataFrame, train: pd.DataFrame, encoding, hist_file, card_norm, cost_norm, knob_norm, to_predict, table_sample):
        self.table_sample = table_sample
        self.encoding = encoding
        self.hist_file = hist_file
        self.length = len(json_df)

        nodes = [json.loads(plan)['Plan'] if isinstance(plan, str) else plan['Plan'] for plan in json_df['json']]

        self.cards = [node.get('Actual Rows', 0) for node in nodes]
        self.costs = [json.loads(plan).get('Execution Time', 0) if isinstance(plan, str) else plan.get('Execution Time', 0) for plan in json_df['json']]
        self.knob_settings = [self.extract_knob_setting(node) for node in nodes]
            
        knob_tensors = [torch.tensor(k, dtype=torch.float32) for k in self.knob_settings]
        padded_knobs = pad_sequence(knob_tensors, batch_first=True, padding_value=0).numpy()

        self.card_labels = torch.from_numpy(card_norm.normalize_labels(self.cards))
        self.cost_labels = torch.from_numpy(cost_norm.normalize_labels(self.costs))
        self.knob_labels = torch.from_numpy(padded_knobs)  # Keep binary

        print(f'Total execution time is : {self.costs}')

        self.to_predict = to_predict
        if to_predict == 'cost':
            self.gts = self.costs
            self.labels = self.cost_labels
        elif to_predict == 'knobs':
            self.gts = self.knob_settings
            self.labels = self.knob_labels
        elif to_predict == 'card':
            self.gts = self.cards
            self.labels = self.card_labels
        elif to_predict == 'both':
            self.gts = {'card': self.cards, 'cost': self.costs, 'knobs': self.knob_settings}
            self.labels = {'card': self.card_labels, 'cost': self.cost_labels, 'knobs': self.knob_labels}
        else:
            raise Exception('Unknown to_predict type')

        idxs = list(json_df['id'])
        self.treeNodes = []
        self.collated_dicts = [self.js_node2dict(i, node) for i, node in zip(idxs, nodes)]

    def extract_knob_setting(self, plan):
        knob_map = {
            'Bitmap Heap Scan': 'enable_bitmapscan',
            'Gather Merge': 'enable_gathermerge',
            'Hash': 'enable_hashagg',
            'Hash Join': 'enable_hashjoin',
            'Index Only Scan': 'enable_indexonlyscan',
            'Index Scan': 'enable_indexscan',
            'Materialize': 'enable_material',
            'Merge Join': 'enable_mergejoin',
            'Nested Loop': 'enable_nestloop',
            'Parallel Append': 'enable_parallel_append',
            'Seq Scan': 'enable_seqscan',
            'Sort': 'enable_sort',
            'Tid Scan': 'enable_tidscan',
            'Gather': 'parallel_leader_participation',
        }

        knobs = {k: 0 for k in [
            'enable_bitmapscan', 'enable_gathermerge', 'enable_hashagg', 'enable_hashjoin',
            'enable_indexonlyscan', 'enable_indexscan', 'enable_material', 'enable_mergejoin',
            'enable_nestloop', 'enable_parallel_append', 'enable_parallel_hash',
            'enable_partition_pruning', 'enable_seqscan', 'enable_sort', 'enable_tidscan',
            'geqo', 'jit', 'jit_expressions', 'jit_tuple_deforming', 'parallel_leader_participation'
        ]}

        def collect_node_info(plan, node_types, parallel_aware):
            if 'Node Type' in plan:
                node_types.add(plan['Node Type'])
                if plan.get('Parallel Aware', False):
                    parallel_aware.add(plan['Node Type'])
            if 'Plans' in plan:
                for subplan in plan['Plans']:
                    collect_node_info(subplan, node_types, parallel_aware)

        node_types = set()
        parallel_aware_nodes = set()
        collect_node_info(plan, node_types, parallel_aware_nodes)

        for node_type in node_types:
            if node_type in knob_map:
                knobs[knob_map[node_type]] = 1
        if 'Gather' in node_types or any('Parallel' in nt for nt in node_types):
            knobs['parallel_leader_participation'] = 1

        return [knobs[k] for k in sorted(knobs.keys())]

    def js_node2dict(self, idx, node):
        treeNode = self.traversePlan(node, idx, self.encoding)
        _dict = self.node2dict(treeNode)
        collated_dict = self.pre_collate(_dict)
        self.treeNodes.clear()
        del self.treeNodes[:]
        return collated_dict

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        weight = 1.0 / (self.costs[idx] + 1e-6)
        if self.to_predict == 'both':
            return self.collated_dicts[idx], (self.cost_labels[idx], self.knob_labels[idx]), weight
        elif self.to_predict == 'cost':
            return self.collated_dicts[idx], self.cost_labels[idx], weight
        elif self.to_predict == 'knobs':
            return self.collated_dicts[idx], self.knob_labels[idx], weight
        else:
            raise Exception('Unknown to_predict type')

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
            'heights': heights,
            'adjacency_list': edge_index
        }

    def node2dict(self, treeNode):
        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))
        return {
            'features': torch.FloatTensor(np.array(features)),
            'heights': torch.LongTensor(heights),
            'adjacency_list': torch.LongTensor(np.array(adj_list)),
        }

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
        root.plan = plan
        root.feature = node2feature(root, encoding, self.hist_file, self.table_sample, self.knob_settings)
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding)
                node.parent = root
                root.addChild(node)
        return root

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

def node2feature(node, encoding, hist_file, table_sample, knob_settings):
    num_filter = len(node.filterDict['colId'])
    pad = np.zeros((3, 3 - num_filter))
    filts = np.array(list(node.filterDict.values()))
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

    costs = np.array([
        node.plan.get('Startup Cost', 0),
        node.plan.get('Total Cost', 0),
        node.plan.get('Actual Startup Time', 0),
        node.plan.get('Actual Total Time', 0)
    ])

    # Adjusted to handle IndexError
    if node.query_id < len(knob_settings):
        knobs = np.array(knob_settings[node.query_id])
    else:
        knobs = np.zeros(20)  # Default to 20 zeros if out of range
        # print(f"Warning: node.query_id {node.query_id} exceeds knob_settings length {len(knob_settings)}. Using default zeros.")

    return np.concatenate((type_join, filts, mask, hists, table, sample, costs, knobs))