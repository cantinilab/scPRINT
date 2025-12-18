"""
Graph-regularized logit refinement implementation.

This module implements the GRIT (Graph-Regularized logIT) refinement method
for improving cell type predictions using graph structure.
"""

from typing import Optional, Union

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


def graph_regularized_logit_refinement(
    pred: np.ndarray,
    adata: anndata.AnnData,
    connectivity_key: str = "connectivities",
    lambda_reg: float = 0.1,
    use_laplacian: bool = True,
) -> np.ndarray:
    """
    Refine logits using graph-regularized optimization.
    Optimized version that solves for all classes simultaneously.

    This function implements the optimization problem:
    P̃ = arg min_P ||P - P₀||²_F + λ Tr(P^T L P)

    where P₀ are the initial logits, L is the graph Laplacian, and λ controls
    the strength of regularization.

    The solution has a closed form: P̃ = (I + λL)⁻¹P₀

    Args:
        pred (np.ndarray): Initial logits of shape (n_cells, n_classes)
        adata (anndata.AnnData): AnnData object containing graph connectivity
        connectivity_key (str): Key in adata.obsp for connectivity matrix
        lambda_reg (float): Regularization strength λ > 0
        use_laplacian (bool): If True, use graph Laplacian; if False, use adjacency matrix

    Returns:
        np.ndarray: Refined logits of same shape as input pred

    Raises:
        ValueError: If connectivity matrix is not found or dimensions don't match
        KeyError: If connectivity_key is not in adata.obsp
    """

    # Validate inputs
    if connectivity_key not in adata.obsp:
        raise KeyError(f"Connectivity key '{connectivity_key}' not found in adata.obsp")

    A = adata.obsp[connectivity_key]
    n_cells, n_classes = pred.shape

    # Check dimensions
    if A.shape[0] != n_cells or A.shape[1] != n_cells:
        raise ValueError(
            f"Connectivity matrix shape {A.shape} doesn't match number of cells {n_cells}"
        )

    # Ensure adjacency matrix is symmetric and sparse
    if not sp.issparse(A):
        A = sp.csr_matrix(A)

    # Make symmetric if not already
    A = (A + A.T) / 2

    if use_laplacian:
        # Compute graph Laplacian: L = D - A
        # where D is the diagonal degree matrix
        degrees = np.array(A.sum(axis=1)).flatten()
        D = sp.diags(degrees, format="csr")
        L = D - A
    else:
        # Use adjacency matrix directly
        L = A

    identity_matrix = sp.identity(n_cells, format="csr")
    system_matrix = identity_matrix + lambda_reg * L

    # Solve for all classes at once instead of looping
    # spsolve can handle multiple right-hand sides
    refined_pred = spsolve(system_matrix, pred)

    # Handle the case where spsolve returns 1D array for single class
    if refined_pred.ndim == 1 and n_classes == 1:
        refined_pred = refined_pred.reshape(-1, 1)
    elif refined_pred.ndim == 1:
        refined_pred = refined_pred.reshape(n_cells, n_classes)

    return refined_pred


def build_knn_graph(
    adata: anndata.AnnData,
    representation_key: str = "X_pca",
    n_neighbors: int = 15,
    metric: str = "euclidean",
) -> anndata.AnnData:
    """
    Build a k-nearest neighbor graph and store it in adata.obsp.

    Args:
        adata (anndata.AnnData): AnnData object
        representation_key (str): Key in adata.obsm for the representation to use. Defaults to "X_pca".
        n_neighbors (int): Number of nearest neighbors. Defaults to 15.
        metric (str): Distance metric for nearest neighbor search. Defaults to "euclidean".

    Returns:
        anndata.AnnData: Updated AnnData object with connectivity matrix
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for building k-NN graphs")

    # Compute neighbors
    sc.pp.neighbors(
        adata,
        use_rep=representation_key,
        n_neighbors=n_neighbors,
        metric=metric,
    )

    return adata


def zero_shot_annotation_with_refinement(
    pred: np.ndarray,
    adata: anndata.AnnData,
    connectivity_key: str = "connectivities",
    representation_key: str = "X_pca",
    n_neighbors: int = 15,
    metric: str = "euclidean",
    lambda_reg: float = 0.1,
    return_probabilities: bool = False,
    return_raw: bool = False,
) -> Union[np.ndarray, tuple]:
    """
    Perform zero-shot cell type annotation with graph refinement.

    This function first refines the logits using graph regularization,
    then performs argmax to get final predictions.

    Args:
        pred (np.ndarray): Initial logits of shape (n_cells, n_classes)
        adata (anndata.AnnData): AnnData object containing graph connectivity
        connectivity_key (str): Key in adata.obsp for connectivity matrix
        lambda_reg (float): Regularization strength
        return_probabilities (bool): If True, also return refined probabilities

    Returns:
        np.ndarray or tuple: If return_probabilities is False, returns array of
                           predicted class indices. If True, returns tuple of
                           (predictions, refined_probabilities)
    """
    if pred is type(pd.DataFrame):
        pred = pred.values
    if adata.obsp.get(connectivity_key) is None:
        # Refine logits
        adata = build_knn_graph(
            adata=adata,
            representation_key=representation_key,
            n_neighbors=n_neighbors,
            metric=metric,
        )
        connectivity_key = "connectivities"
    print(adata.obsp)
    refined_logits = graph_regularized_logit_refinement(
        pred, adata, connectivity_key, lambda_reg
    )
    if return_raw:
        return refined_logits
    # Get predictions: g(xi) = arg max_j {P̃(i)}
    predictions = np.argmax(refined_logits, axis=1)

    if return_probabilities:
        # Convert to probabilities using softmax
        refined_probs = np.exp(refined_logits)
        refined_probs = refined_probs / refined_probs.sum(axis=1, keepdims=True)
        return predictions, refined_probs

    return predictions


# Example usage and test function
def test_graph_refinement():
    """Test function for graph refinement."""

    # Create synthetic data
    n_cells, n_classes = 100, 5

    # Random logits
    np.random.seed(42)
    pred = np.random.randn(n_cells, n_classes)

    # Create synthetic AnnData with connectivity
    adata = anndata.AnnData(X=np.random.randn(n_cells, 50))

    # Create a random sparse connectivity matrix
    from scipy.sparse import random

    connectivity = random(n_cells, n_cells, density=0.1, format="csr")
    connectivity = (connectivity + connectivity.T) / 2  # Make symmetric
    adata.obsp["connectivities"] = connectivity

    # Test refinement
    refined_pred = graph_regularized_logit_refinement(pred, adata, lambda_reg=0.1)

    print(f"Original logits shape: {pred.shape}")
    print(f"Refined logits shape: {refined_pred.shape}")
    print(f"Logits changed: {not np.allclose(pred, refined_pred)}")

    # Test zero-shot annotation
    predictions, probabilities = zero_shot_annotation_with_refinement(
        pred, adata, return_probabilities=True
    )

    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Predicted classes: {np.unique(predictions)}")


if __name__ == "__main__":
    test_graph_refinement()
