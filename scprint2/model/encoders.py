import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MLP, GATConv, GCNConv, SAGEConv
from torch_geometric.nn.aggr import DeepSetsAggregation


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        weights: Optional[Tensor] = None,
        weights_file: Optional[str] = None,
        freeze: bool = False,
    ):
        """
        Encodes gene sequences into a continuous vector space using an embedding layer.
        Uses memory mapping for efficient access to large embedding files.

        Args:
            num_embeddings (int): The number of possible values
            embedding_dim (int): The dimension of the output vectors
            padding_idx (int, optional): The index of the padding token
            weights (Tensor, optional): The initial weights for the embedding layer
            weights_file (str, optional): Path to parquet file containing embeddings
            freeze (bool, optional): Whether to freeze the weights of the embedding layer
        """
        super(GeneEncoder, self).__init__()
        self.output_dim = embedding_dim

        if weights_file is not None:
            self.memmap = True
            if not freeze:
                raise ValueError(
                    "freeze must be True when using memory-mapped embeddings"
                )
            # Load the parquet file and create memory-mapped array
            import os

            import pandas as pd

            # Create memory-mapped file path
            self.mmap_file = f"{weights_file}.mmap"
            self.loc = None
            self.enc = None
            # Only create the memory-mapped file if it doesn't exist
            if not os.path.exists(self.mmap_file):
                print(f"Creating memory-mapped file for embeddings at {self.mmap_file}")
                df = pd.read_parquet(weights_file)
                embeddings = torch.nn.AdaptiveAvgPool1d(self.output_dim)(
                    torch.tensor(df.values)
                )

                # Create memory-mapped array
                self.embeddings = np.memmap(
                    self.mmap_file, dtype="float32", mode="w+", shape=embeddings.shape
                )
                # Copy data to memory-mapped array
                self.embeddings[:] = embeddings.numpy()
                #
                self.embeddings.flush()

                # Clean up memory
                del df
                del embeddings
            else:
                print(
                    f"Loading existing memory-mapped embeddings from {self.mmap_file}"
                )
                # Load existing memory-mapped file
                self.embeddings = np.memmap(
                    self.mmap_file,
                    dtype="float32",
                    mode="r",  # Read-only mode since we don't need to modify
                    shape=(num_embeddings, embedding_dim),
                )
        else:
            self.memmap = False
            self.embeddings = nn.Embedding(
                num_embeddings, embedding_dim, padding_idx=padding_idx, _freeze=freeze
            )
            if weights is not None:
                self.embeddings.weight.data.copy_(torch.Tensor(weights))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the encoder.

        Args:
            x (Tensor): Input tensor of indices [batch_size, seq_len]

        Returns:
            Tensor: Embedded vectors [batch_size, seq_len, embedding_dim]
        """
        if self.memmap:
            if self.loc is None or not torch.all(x.sum(1) == self.loc):
                self.enc = (
                    torch.from_numpy(
                        self.embeddings[x.reshape(-1).cpu().numpy()].copy()
                    )
                    .reshape(x.shape + (-1,))
                    .to(x.device)
                )
                self.loc = x.sum(1)
            return self.enc.clone()
        else:
            return self.embeddings(x)

    def __del__(self):
        """Cleanup method to ensure proper handling of memory-mapped file."""
        if hasattr(self, "embeddings") and self.embeddings is not None:
            try:
                self.embeddings._mmap.close()
            except:
                pass

    def _init_weights(self):
        pass


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        gene_pos_enc: list[str] = [],
    ):
        """
        The PositionalEncoding module applies a positional encoding to a sequence of vectors.
        This is necessary for the Transformer model, which does not have any inherent notion of
        position in a sequence. The positional encoding is added to the input embeddings and
        allows the model to attend to positions in the sequence.

        Args:
            d_model (int): The dimension of the input vectors.
            gene_pos_enc (list[str], optional): The gene position encoding to use.

        Note: not used in the current version of scprint-2.
        """
        super(PositionalEncoding, self).__init__()
        self.gene_pos_enc = gene_pos_enc
        max_len = max(gene_pos_enc)
        position = torch.arange(max_len).unsqueeze(1)
        token_to_pos = {token: pos for token, pos in enumerate(gene_pos_enc)}

        # Create a dictionary to convert token to position

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(float(10_000)) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # we reorder them and map them to gene_id (position)
        arr = []
        for _, v in token_to_pos.items():
            arr.append(pe[v - 1].numpy())
        pe = torch.Tensor(np.array(arr))
        # Remove the unnecessary middle dimension since pe should be [m, d]
        # pe = pe.squeeze(1)
        self.register_buffer("pe", pe)

    def forward(self, gene_pos: Tensor) -> Tensor:
        """
        Args:
            gene_pos (Tensor): Gene position indices, shape [seq_len, batch_size] or [seq_len]

        Returns:
            Tensor: Positional encodings, shape [*gene_pos.shape, embedding_dim]
        """
        return torch.index_select(self.pe, 0, gene_pos.reshape(-1)).reshape(
            gene_pos.shape + (-1,)
        )


class DPositionalEncoding(nn.Module):
    """
    The PositionalEncoding module applies a positional encoding to a sequence of vectors.
    This is necessary for the Transformer model, which does not have any inherent notion of
    position in a sequence. The positional encoding is added to the input embeddings and
    allows the model to attend to positions in the sequence.

    Args:
        d_model (int): The dimension of the input vectors.
        max_len_x (int): The maximum length in the x dimension.
        max_len_y (int): The maximum length in the y dimension.
        maxvalue_x (float, optional): Maximum value for x dimension scaling. Defaults to 10000.0.
        maxvalue_y (float, optional): Maximum value for y dimension scaling. Defaults to 10000.0.

    Note: not used in the current version of scprint-2.
    """

    def __init__(
        self,
        d_model: int,
        max_len_x: int,
        max_len_y: int,
        maxvalue_x=10000.0,
        maxvalue_y=10000.0,
    ):
        super(DPositionalEncoding, self).__init__()
        position2 = torch.arange(max_len_y).unsqueeze(1)
        position1 = torch.arange(max_len_x).unsqueeze(1)

        half_n = d_model // 2

        div_term2 = torch.exp(
            torch.arange(0, half_n, 2) * (-math.log(maxvalue_y) / d_model)
        )
        div_term1 = torch.exp(
            torch.arange(0, half_n, 2) * (-math.log(maxvalue_x) / d_model)
        )
        pe1 = torch.zeros(max_len_x, 1, d_model)
        pe2 = torch.zeros(max_len_y, 1, d_model)
        pe1[:, 0, 0:half_n:2] = torch.sin(position1 * div_term1)
        pe1[:, 0, 1:half_n:2] = torch.cos(position1 * div_term1)
        pe2[:, 0, half_n::2] = torch.sin(position2 * div_term2)
        pe2[:, 0, 1 + half_n :: 2] = torch.cos(position2 * div_term2)
        # https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
        self.register_buffer("pe1", pe1)
        self.register_buffer("pe2", pe2)

        # PE(x,y,2i) = sin(x/10000^(4i/D))
        # PE(x,y,2i+1) = cos(x/10000^(4i/D))
        # PE(x,y,2j+D/2) = sin(y/10000^(4j/D))
        # PE(x,y,2j+1+D/2) = cos(y/10000^(4j/D))

    def forward(self, x: Tensor, pos_x: Tensor, pos_y: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe1[pos_x]
        x = x + self.pe2[pos_y]
        return x


class ContinuousValueEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_value: int = 100_000,
        layers: int = 1,
        size: int = 1,
    ):
        """
        Encode real number values to a vector using neural nets projection.

        Args:
            d_model (int): The dimension of the input vectors.
            dropout (float, optional): The dropout rate to apply to the output of the positional encoding.
            max_value (int, optional): The maximum value of the input. Defaults to 100_000.
            layers (int, optional): The number of layers in the encoder. Defaults to 1.
            size (int, optional): The size of the input. Defaults to 1.
        """
        super(ContinuousValueEncoder, self).__init__()
        self.max_value = max_value
        self.encoder = nn.ModuleList()
        self.output_dim = d_model
        # self.mask_value = nn.Embedding(1, d_model)
        self.encoder.append(nn.Linear(size, d_model))
        for _ in range(layers - 1):
            self.encoder.append(nn.LayerNorm(d_model))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Dropout(p=dropout))
            self.encoder.append(nn.Linear(d_model, d_model))

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # expand last dimension
        x = x.unsqueeze(-1)
        # use the mask embedding when x=-1
        # mask = (x == -1).float()
        x = torch.clamp(x, min=0, max=self.max_value)
        for val in self.encoder:
            x = val(x)
        if mask is not None:
            x = x.masked_fill_(mask.unsqueeze(-1), 0)
            # x = x.masked_fill_(mask.unsqueeze(-1), self.mask_value(0))
        return x

    def _init_weights(self):
        pass
        # for m in self.encoder:
        #    if isinstance(m, nn.Linear):
        #        torch.nn.init.eye_(m.weight)


class ExprBasedFT(nn.Module):
    def __init__(
        self,
        d_model: int,
        gene_encoder: nn.Module,
        expr_encoder: nn.Module = nn.Identity(),
        dropout: float = 0.1,
        layers: int = 2,
        intermediary_d: int = 256 + 64,
    ):
        """
        Encode real number values to a vector using neural nets projection.

        Args:
            d_model (int): The dimension of the input vectors.
            gene_encoder (nn.Module): The gene name encoder module.
            expr_encoder (nn.Module, optional): The expression encoder module. Defaults to nn.Identity.
            dropout (float, optional): The dropout rate to apply to the output of the positional encoding.
            layers (int, optional): The number of layers in the encoder. Defaults to 2.
            intermediary_d (int, optional): The dimension of the intermediary layers. Defaults to 256 + 64.

        """
        super(ExprBasedFT, self).__init__()
        self.encoder = nn.ModuleList()
        # self.mask_value = nn.Embedding(1, d_model)
        self.add_module("gene_encoder", gene_encoder)
        self.add_module("expr_encoder", expr_encoder)
        expr_shape, gene_shape = (
            self.expr_encoder.output_dim,
            self.gene_encoder.output_dim,
        )
        self.encoder.append(nn.Linear(expr_shape + gene_shape, intermediary_d))
        for i in range(layers - 1):
            self.encoder.append(nn.LayerNorm(intermediary_d))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Dropout(p=dropout))
            self.encoder.append(
                nn.Linear(intermediary_d, intermediary_d if i < layers - 2 else d_model)
            )

    def forward(
        self,
        gene_pos: Tensor,
        expr: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        neighbors: Optional[Tensor] = None,
        neighbors_info: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            gene_pos (Tensor[batch_size, seq_len]): Gene position indices
                input to the gene encoder
            expr (Tensor[batch_size, seq_len], Optional): Expression values
                input to the expression encoder
            mask (Tensor[batch_size, seq_len], Optional): Mask for the input
                input to the expression encoder
            neighbors (Tensor[batch_size, seq_len, n_neighbors], Optional): Neighbors indices
                input to the expression encoder when it is a GNN
            neighbors_info (Tensor[batch_size, seq_len, n_neighbors], Optional):
                optional additional information about the neighbors
                input to the expression encoder when it is a GNN
        """
        # expand last dimension
        if neighbors is None and expr is None:
            expr = torch.zeros(
                (gene_pos.shape[0], gene_pos.shape[1], self.expr_encoder.output_dim),
                dtype=torch.float32,
                device=gene_pos.device,
            )
            # if no expr information: consider that it is all masked
        else:
            expr = (
                self.expr_encoder(expr, mask=mask)
                if neighbors is None
                else self.expr_encoder(expr, neighbors, neighbors_info, mask=mask)
            )
        gene_pos = self.gene_encoder(gene_pos)
        x = torch.cat([expr, gene_pos], dim=-1)
        for val in self.encoder:
            x = val(x)
        return x

    def _init_weights(self):
        pass

    #    for m in self.encoder:
    #        if isinstance(m, nn.Linear):
    #            torch.nn.init.eye_(m.weight)
    #    self.expr_encoder._init_weights()


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        """
        Encodes categorical values into a vector using an embedding layer and layer normalization.

        Args:
            num_embeddings (int): The number of possible values.
            embedding_dim (int): The dimension of the output vectors.
            padding_idx (int, optional): The index of the padding token. Defaults to None.

        Note: not used in the current version of scprint-2.
        """
        super(CategoryValueEncoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        x = self.embedding(x.long())  # (batch, seq_len, embsize)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        return x

    def _init_weights(self):
        pass


class EasyExprGNN(nn.Module):
    def __init__(
        self,
        self_dim: int = 64,
        output_dim: int = 32,
        self_layers: int = 2,
        dropout: float = 0.1,
        shared_layers: int = 2,
        neighbors_layers: int = 2,
    ):
        """
        Easy Expression Graph Neural Network

        The main GNN used in scPRINT-2 for expression encoding.
        It is inspired from the DeepSets architecture to aggregate neighbor information.

        Args:
            self_dim (int): Dimension of the self features
            output_dim (int): Output dimension
            self_layers (int): Number of layers for self features
            dropout (float): Dropout rate
            shared_layers (int): Number of shared layers
            neighbors_layers (int): Number of layers for neighbors features
        """
        super(EasyExprGNN, self).__init__()
        self.output_dim = output_dim
        self.self_dim = self_dim
        # neighbors
        self.neighbors_layers = nn.ModuleList()
        self.neighbors_layers.append(nn.Linear(2, self_dim // 2))
        for i in range(neighbors_layers - 1):
            self.neighbors_layers.append(nn.LayerNorm(self_dim // 2))
            self.neighbors_layers.append(nn.ReLU())
            self.neighbors_layers.append(nn.Dropout(p=dropout))
            self.neighbors_layers.append(nn.Linear(self_dim // 2, self_dim // 2))
        # self
        self.self_layers = nn.ModuleList()
        self.self_layers.append(nn.Linear(1, self_dim // 2))
        for i in range(self_layers - 1):
            self.self_layers.append(nn.LayerNorm(self_dim // 2))
            self.self_layers.append(nn.ReLU())
            self.self_layers.append(nn.Dropout(p=dropout))
            self.self_layers.append(nn.Linear(self_dim // 2, self_dim // 2))
        # shared
        self.shared_layers = nn.ModuleList()
        for i in range(shared_layers - 1):
            self.shared_layers.append(nn.Linear(self_dim, self_dim))
            self.shared_layers.append(nn.LayerNorm(self_dim))
            self.shared_layers.append(nn.ReLU())
            self.shared_layers.append(nn.Dropout(p=dropout))
        self.shared_layers.append(nn.Linear(self_dim, output_dim))

    def forward(
        self,
        expr: Optional[Tensor] = None,
        neighbors: Optional[Tensor] = None,
        edge_info: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the Easy Expression GNN

        Args:
            expr: Tensor of shape (batch, seq_len) representing expression values
            neighbors: Tensor of shape (batch, seq_len, n_neighbors) representing neighbor indices
            edge_info: Tensor of shape (batch, seq_len, n_neighbors) representing edge information
            mask: Tensor of shape (batch, seq_len) representing mask for the input

        Returns:
            Tensor of shape (batch, seq_len, output_dim) representing the output features
        """
        # batch, seq_len, neighbs
        if neighbors is None:
            neighbors = torch.zeros(
                (expr.shape[0], expr.shape[1], self.self_dim // 2), device=expr.device
            )
        else:
            neighbors = neighbors.transpose(1, 2)
            neighbors = torch.cat(
                [neighbors.unsqueeze(-1), edge_info.unsqueeze(-1)], dim=-1
            )
            for i, layer in enumerate(self.neighbors_layers):
                # batch, seq_len, neighbs, hidden_dim
                neighbors = layer(neighbors)
            neighbors = neighbors.sum(-2)
        if expr is None:
            expr = torch.zeros(
                (neighbors.shape[0], neighbors.shape[1], 1), device=neighbors.device
            )
        else:
            expr = expr.unsqueeze(-1)
            for i, layer in enumerate(self.self_layers):
                expr = layer(expr)
        x = torch.cat([expr, neighbors], dim=-1)
        for layer in self.shared_layers:
            # batch, seq_len, neighbs, hidden_dim
            x = layer(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        return x

    def _init_weights(self):
        pass
        # for m in self.neighbors_layers:
        #    if isinstance(m, nn.Linear):
        #        torch.nn.init.zeros_(m.weight)
        #        if m.bias is not None:
        #            torch.nn.init.constant_(m.bias, 0)
        # for m in self.self_layers:
        #    if isinstance(m, nn.Linear):
        #        torch.nn.init.eye_(m.weight)
        #        if m.bias is not None:
        #            torch.nn.init.constant_(m.bias, 0)
        # for m in self.shared_layers:
        #    if isinstance(m, nn.Linear):
        #        torch.nn.init.eye_(m.weight)
        #        if m.bias is not None:
        #            torch.nn.init.constant_(m.bias, 0)
        #


class GNN(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,  # here, 1 or 2
        merge_dim: int = 32,
        output_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        gnn_type: str = "deepset",
        add_connection_feature: bool = False,
    ):
        """
        Graph Neural Network model

        Another implementation of a GNN layer that can be used for expression encoding.
        Supports GCN, GAT, GraphSAGE, and DeepSets architectures.

        Args:
            input_dim: Dimension of input node features
            output_dim: Dimension of output node features
            num_layers: Number of GNN layers
            dropout: Dropout probability
            gnn_type: Type of GNN layer ('gcn', 'gat', 'sage', or 'deepset')
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        if num_layers == 1:
            raise ValueError("num_layers must be greater than 1")
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.add_connection_feature = add_connection_feature

        if gnn_type == "deepset":
            # Local MLP (phi) for processing individual nodes
            self.input_nn_layer = MLP(
                in_channels=input_dim,
                hidden_channels=merge_dim,
                out_channels=merge_dim,
                num_layers=num_layers,
                dropout=dropout,
                act="relu",
                norm="layer_norm",
            )

            self.input_self_layer = MLP(
                in_channels=input_dim,
                hidden_channels=merge_dim + 2,
                out_channels=merge_dim,
                num_layers=num_layers - 1,
                dropout=dropout,
                act="relu",
                norm="layer_norm",
            )

            # Global MLP (rho) for processing aggregated features
            self.output_layer = MLP(
                in_channels=(
                    (merge_dim * 2) + 1 if add_connection_feature else merge_dim * 2
                ),
                hidden_channels=output_dim,
                out_channels=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                act="relu",
                norm="layer_norm",
            )

            return

        # Select GNN layer type for other architectures
        else:
            if gnn_type == "gcn":
                gnn_layer = GCNConv
            elif gnn_type == "gat":
                gnn_layer = GATConv
            elif gnn_type == "sage":
                gnn_layer = SAGEConv
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.gnn_layer = gnn_layer(
                output_dim,
                output_dim,
                add_self_loops=False,
                normalize=False,
                aggr="mean",
            )

    def forward(
        self,
        x: Tensor,
        neighbors: Tensor,
        edge_info: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass

        Args:
            x (Tensor): Node features [minibatch_size, ngenes]
            neighbors (Tensor): Neighbor nodes [minibatch_size, ngenes, n_neighbors] or [minibatch_size, ngenes, n_neighbors, 2]
            edge_info (Tensor, optional): Graph connectivity [2, num_edges] if gnn_type != deepset,
                Edge features [num_edges, 1] if gnn_type == deepset,
                or None if gnn_type == deepset and no edge features.
            batch (Tensor, optional): Batch assignment vector [num_nodes]
            mask (Tensor, optional): Mask tensor for the nodes.

        Returns:
            Tensor: Node embeddings [num_nodes, hidden_dim]
        """

        # Standard GNN forward pass
        x = x.unsqueeze(-1)
        neighbors = neighbors.unsqueeze(-1)
        if self.gnn_type == "deepset":
            neighbors = self.input_nn_layer(neighbors).sum(dim=-3)
            x = self.input_self_layer(x)
            x = torch.cat([x, neighbors], dim=-1)
        else:
            x = self.gnn_layer(x, edge_info)
            neighbors = self.gnn_layer(neighbors, edge_info)
            for layer in self.layers:
                x = layer(x, edge_info)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # TODO: to finish

        x = self.output_layer(x)
        if mask is not None:
            x = x.masked_fill_(mask.unsqueeze(-1), 0)
        return x
