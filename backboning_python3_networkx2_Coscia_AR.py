'''
Functions written and made available by Coscia and Neffke to reproduce the results of the paper "Network Backboning with Noisy Data" (2017).
'''

import sys, warnings
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from scipy.stats import binom


def thresholding(table, threshold):
   """Reads a preprocessed edge table and returns only the edges supassing a significance threshold.

   Args:
   table (pandas.DataFrame): The edge table.
   threshold (float): The minimum significance to include the edge in the backbone.

   Returns:
   The network backbone.
   """
   table = table.copy()
   if "sdev" in table:
       return table[(table["score"] - (threshold * table["sdev"])) > 0][["node1", "node2", "count", "score"]]
   else:
       return table[table["score"] > threshold][["node1", "node2", "count", "score"]]

def noise_corrected(table, undirected = False, return_self_loops = False, calculate_p_value = False):
   sys.stderr.write("Calculating NC score...\n")
   table = table.copy()
   node1_sum = table.groupby(by = "node1").sum()[["count"]]
   table = table.merge(node1_sum, left_on = "node1", right_index = True, suffixes = ("", "_node1_sum"))
   node2_sum = table.groupby(by = "node2").sum()[["count"]]
   table = table.merge(node2_sum, left_on = "node2", right_index = True, suffixes = ("", "_node2_sum"))
   table.rename(columns = {"count_node1_sum": "ni.", "count_node2_sum": "n.j"}, inplace = True)
   table["n.."] = table["count"].sum()
   table["mean_prior_probability"] = ((table["ni."] * table["n.j"]) / table["n.."]) * (1 / table["n.."])
   if calculate_p_value:
      table["score"] = binom.cdf(table["count"], table["n.."], table["mean_prior_probability"])
      return table[["node1", "node2", "count", "score"]]
   table["kappa"] = table["n.."] / (table["ni."] * table["n.j"])
   table["score"] = ((table["kappa"] * table["count"]) - 1) / ((table["kappa"] * table["count"]) + 1)
   table["var_prior_probability"] = (1 / (table["n.."] ** 2)) * (table["ni."] * table["n.j"] * (table["n.."] - table["ni."]) * (table["n.."] - table["n.j"])) / ((table["n.."] ** 2) * ((table["n.."] - 1)))
   table["alpha_prior"] = (((table["mean_prior_probability"] ** 2) / table["var_prior_probability"]) * (1 - table["mean_prior_probability"])) - table["mean_prior_probability"]
   table["beta_prior"] = (table["mean_prior_probability"] / table["var_prior_probability"]) * (1 - (table["mean_prior_probability"] ** 2)) - (1 - table["mean_prior_probability"])
   table["alpha_post"] = table["alpha_prior"] + table["count"]
   table["beta_post"] = table["n.."] - table["count"] + table["beta_prior"]
   table["expected_pij"] = table["alpha_post"] / (table["alpha_post"] + table["beta_post"])
   table["variance_count"] = table["expected_pij"] * (1 - table["expected_pij"]) * table["n.."]
   table["d"] = (1.0 / (table["ni."] * table["n.j"])) - (table["n.."] * ((table["ni."] + table["n.j"]) / ((table["ni."] * table["n.j"]) ** 2)))
   table["variance_cij"] = table["variance_count"] * (((2 * (table["kappa"] + (table["count"] * table["d"]))) / (((table["kappa"] * table["count"]) + 1) ** 2)) ** 2)
   table["sdev"] = table["variance_cij"] ** .5
   if not return_self_loops:
      table = table[table["node1"] != table["node2"]]
   if undirected:
      table = table[table["node1"] <= table["node2"]]
   return table[["node1", "node2", "count", "score", "sdev"]]

def disparity_filter(table, undirected = False, return_self_loops = False):
   sys.stderr.write("Calculating DF score...\n")
   table = table.copy()
   table_sum = table.groupby(table["node1"]).sum().reset_index()
   table_deg = table.groupby(table["node1"]).count()["node2"].reset_index()
   table = table.merge(table_sum, on = "node1", how = "left", suffixes = ("", "_sum"))
   table = table.merge(table_deg, on = "node1", how = "left", suffixes = ("", "_count"))
   table["score"] = 1.0 - ((1.0 - (table["count"] / table["count_sum"])) ** (table["node2_count"] - 1))
   table["variance"] = (table["node2_count"] ** 2) * (((20 + (4.0 * table["node2_count"])) / ((table["node2_count"] + 1.0) * (table["node2_count"] + 2) * (table["node2_count"] + 3))) - ((4.0) / ((table["node2_count"] + 1.0) ** 2)))
   if not return_self_loops:
      table = table[table["node1"] != table["node2"]]
   if undirected:
      table["edge"] = table.apply(lambda x: "%s-%s" % (min(x["node1"], x["node2"]), max(x["node1"], x["node2"])), axis = 1)
      table_maxscore = table.groupby(by = "edge")["score"].max().reset_index()
      table_minvar = table.groupby(by = "edge")["variance"].min().reset_index()
      table = table.merge(table_maxscore, on = "edge", suffixes = ("_min", ""))
      table = table.merge(table_minvar, on = "edge", suffixes = ("_max", ""))
      table = table.drop_duplicates(subset = ["edge"])
      table = table.drop("edge", axis = 1)
      table = table.drop("score_min", axis = 1)
      table = table.drop("variance_max", axis = 1)
      table["sdev"] = table["variance"]**(1/2)
   return table[["node1", "node2", "count", "score", "variance", "sdev"]]

def naive(table, undirected = False, return_self_loops = False):
   sys.stderr.write("Calculating Naive score...\n")
   table = table.copy()
   table["score"] = table["count"]
   if not return_self_loops:
      table = table[table["node1"] != table["node2"]]
   if undirected:
      table["edge"] = table.apply(lambda x: "%s-%s" % (min(x["node1"], x["node2"]), max(x["node1"], x["node2"])), axis = 1)
      table_maxscore = table.groupby(by = "edge")["score"].sum().reset_index()
      table = table.merge(table_maxscore, on = "edge", suffixes = ("_min", ""))
      table = table.drop_duplicates(subset = ["edge"])
      table = table.drop("edge", axis = 1)
      table = table.drop("score_min", axis = 1)
      table["score"] = table["score"] / 2.0
   return table[["node1", "node2", "count", "score"]]

def maximum_spanning_tree(table, undirected = False):
   sys.stderr.write("Calculating MST score...\n")
   table = table.copy()
   table["distance"] = 1.0 / table["count"]
   G = nx.from_pandas_edgelist(table, source = "node1", target = "node2", edge_attr = ["distance", "count"])
   T = nx.minimum_spanning_tree(G, weight = "distance")
   table2 = nx.to_pandas_edgelist(T)
   table2 = table2[table2["count"] > 0]
   table2.rename(columns = {"source": "node1", "target": "node2", "count": "score"}, inplace = True)
   table = table.merge(table2, on = ["node1", "node2"])
   if undirected:
      table["edge"] = table.apply(lambda x: "%s-%s" % (min(x["node1"], x["node2"]), max(x["node1"], x["node2"])), axis = 1)
      table = table.drop_duplicates(subset = ["edge"])
      table = table.drop("edge", axis = 1)
   return table[["node1", "node2", "count", "score"]]
