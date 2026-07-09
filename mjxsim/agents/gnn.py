"""Compatibility imports for the Torch GNN agent."""

from mjxsim.agents.torch.gnn import (
    GNN_CFG,
    GNN_DEFAULT_CONFIG,
    GNNAgent,
    GraphConvolution,
    GraphRegressionGCN,
    normalized_chain_adjacency,
)

__all__ = [
    "GNN_CFG",
    "GNN_DEFAULT_CONFIG",
    "GNNAgent",
    "GraphConvolution",
    "GraphRegressionGCN",
    "normalized_chain_adjacency",
]
