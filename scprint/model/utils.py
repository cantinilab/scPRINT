import torch


def downsample_profile(mat, renoise):
    """
    This function downsamples the expression profile of a given single cell RNA matrix.

    The noise is applied based on the renoise parameter,
    the total counts of the matrix, and the number of genes. The function first calculates the noise
    threshold (tnoise) based on the renoise parameter. It then generates an initial matrix count by
    applying a Poisson distribution to a random tensor scaled by the total counts and the number of genes.
    The function then models the sampling zeros by applying a Poisson distribution to a random tensor
    scaled by the noise threshold, the total counts, and the number of genes. The function also models
    the technical zeros by generating a random tensor and comparing it to the noise threshold. The final
    matrix count is calculated by subtracting the sampling zeros from the initial matrix count and
    multiplying by the technical zeros. The function ensures that the final matrix count is not less
    than zero by taking the maximum of the final matrix count and a tensor of zeros. The function
    returns the final matrix count.

    Args:
        mat (torch.Tensor): The input matrix.
        renoise (float): The renoise parameter.
        totcounts (torch.Tensor): The total counts of the matrix.
        ngenes (int): The number of genes.

    Returns:
        torch.Tensor: The matrix count after applying noise.
    """
    totcounts = mat.sum(1)
    batch = mat.shape[0]
    ngenes = mat.shape[1]
    tnoise = 1 - (1 - renoise) ** (1 / 2)
    # we model the sampling zeros (dropping 30% of the reads)
    res = torch.poisson(
        torch.rand((batch, ngenes)).to(device=mat.device)
        * (
            (tnoise * totcounts.unsqueeze(1)) / (0.5 * ngenes)
        )  # (/ torch.Tensor([3.2,4.1]).unsqueeze(1)
    ).int()
    # we model the technical zeros (dropping 50% of the genes)
    drop = (torch.rand((batch, ngenes)) > tnoise).int().to(device=mat.device)

    mat = (mat - res) * drop
    return torch.maximum(mat, torch.Tensor([[0]]).to(device=mat.device)).int()