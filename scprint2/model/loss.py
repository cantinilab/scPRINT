import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.distributions import NegativeBinomial

# import ot
# from ot.gromov import gromov_wasserstein, fused_gromov_wasserstein


def masked_mse(input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    input = torch.log2(input + 1)
    input = (input / torch.sum(input, dim=1, keepdim=True)) * 10000
    target = torch.log2(target + 1)
    target = (target / torch.sum(target, dim=1, keepdim=True)) * 10000
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def mse(input: Tensor, target: Tensor, mask=False) -> Tensor:
    """
    Compute the MSE loss between input and target.
    """
    if mask:
        return masked_mse(input, target, (target > 0))
    input = torch.log2(input + 1)
    input = (input / torch.sum(input, dim=1, keepdim=True)) * 10000
    target = torch.log2(target + 1)
    target = (target / torch.sum(target, dim=1, keepdim=True)) * 10000
    return F.mse_loss(input, target, reduction="mean")


def masked_mae(input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """
    Compute the masked MAE loss between input and target.
    MAE = mean absolute error
    """
    mask = mask.float()
    loss = F.l1_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def masked_nb(input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """
    Compute the masked negative binomial loss between input and target.
    """
    mask = mask.float()
    nb = torch.distributions.NegativeBinomial(total_count=target, probs=input)
    masked_log_probs = nb.log_prob(target) * mask
    return -masked_log_probs.sum() / mask.sum()


# FROM SCVI
def nb(target: Tensor, mu: Tensor, theta: Tensor, eps=1e-4) -> Tensor:
    """
    Computes the negative binomial (NB) loss.

    This function was adapted from scvi-tools.

    Args:
        target (Tensor): Ground truth data.
        mu (Tensor): Means of the negative binomial distribution (must have positive support).
        theta (Tensor): Inverse dispersion parameter (must have positive support).
        eps (float, optional): Numerical stability constant. Defaults to 1e-4.

    Returns:
        Tensor: NB loss value.
    """
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + target * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(target + theta)
        - torch.lgamma(theta)
        - torch.lgamma(target + 1)
    )

    return -res.mean()


def nb_dist(x: Tensor, mu: Tensor, theta: Tensor, eps=1e-4) -> Tensor:
    """
    nb_dist Computes the negative binomial distribution.

    Args:
        x (Tensor): Torch Tensor of observed data.
        mu (Tensor): Torch Tensor of means of the negative binomial distribution (must have positive support).
        theta (Tensor): Torch Tensor of inverse dispersion parameter (must have positive support).
        eps (float, optional): Numerical stability constant. Defaults to 1e-4.

    Returns:
        Tensor: Negative binomial loss value.
    """
    loss = -NegativeBinomial(mu=mu, theta=theta).log_prob(x)
    return loss


def zinb(
    target: Tensor,
    mu: Tensor,
    theta: Tensor,
    pi: Tensor,
    eps=1e-4,
    mask=False,
) -> Tensor:
    """
    Computes zero-inflated negative binomial (ZINB) loss.

    This function was modified from scvi-tools.

    Args:
        target (Tensor): Torch Tensor of ground truth data.
        mu (Tensor): Torch Tensor of means of the negative binomial (must have positive support).
        theta (Tensor): Torch Tensor of inverse dispersion parameter (must have positive support).
        pi (Tensor): Torch Tensor of logits of the dropout parameter (real support).
        eps (float, optional): Numerical stability constant. Defaults to 1e-4.

    Returns:
        Tensor: ZINB loss value.
    """
    #  uses log(sigmoid(x)) = -softplus(-x)
    softplus_pi = F.softplus(-pi)
    # eps to make it positive support and taking the log
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (torch.log(theta + eps) - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((target < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + target * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(target + theta)
        - torch.lgamma(theta)
        - torch.lgamma(target + 1)
    )
    mul_case_non_zero = torch.mul((target > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    # we want to minize the loss but maximize the log likelyhood
    if mask:
        mask = (target > 0).float()
        res = res * mask
        return -res.sum() / mask.sum()
    return -res.mean()


def criterion_neg_log_bernoulli(input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: Tensor, target: Tensor, mask: torch.LongTensor
) -> Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-5)
    return loss.mean()


def contrastive_loss(x: Tensor, y: Tensor, temperature: float = 0.1) -> Tensor:
    """
    Computes NT-Xent loss (InfoNCE) between two sets of vectors.

    Args:
        x: Tensor of shape [batch_size, feature_dim]
        y: Tensor of shape [batch_size, feature_dim]
        temperature: Temperature parameter to scale the similarities.
            Lower values make the model more confident/selective.
            Typical values are between 0.1 and 0.5.

    Returns:
        Tensor: NT-Xent loss value

    Note:
        - Assumes x[i] and y[i] are positive pairs
        - All other combinations are considered negative pairs
        - Uses cosine similarity scaled by temperature
    """
    # Check input dimensions
    assert x.shape == y.shape, "Input tensors must have the same shape"
    batch_size = x.shape[0]

    # Compute cosine similarity matrix
    # x_unsqueeze: [batch_size, 1, feature_dim]
    # y_unsqueeze: [1, batch_size, feature_dim]
    # -> similarities: [batch_size, batch_size]
    similarities = (
        F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=2) / temperature
    )

    # The positive pairs are on the diagonal
    labels = torch.arange(batch_size, device=x.device)

    # Cross entropy loss
    return F.cross_entropy(similarities, labels)


def ecs(cell_emb: Tensor, ecs_threshold: float = 0.5) -> Tensor:
    """
    ecs Computes the similarity of cell embeddings based on a threshold.

    Args:
        cell_emb (Tensor): A tensor representing cell embeddings.
        ecs_threshold (float, optional): A threshold for determining similarity. Defaults to 0.5.

    Returns:
        Tensor: A tensor representing the mean of 1 minus the square of the difference between the cosine similarity and the threshold.
    """
    # Here using customized cosine similarity instead of F.cosine_similarity
    # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
    # normalize the embedding
    cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
    cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())

    # mask out diagnal elements
    mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
    cos_sim = cos_sim.masked_fill(mask, 0.0)
    # only optimize positive similarities
    cos_sim = F.relu(cos_sim)
    return torch.mean(1 - (cos_sim - ecs_threshold) ** 2)


def hierarchical_classification(
    pred: torch.Tensor,
    cl: torch.Tensor,
    labels_hierarchy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes the classification loss for a given batch of predictions and ground truth labels.

    Args:
        pred (Tensor): The predicted logits for the batch. Shape: (batch_size, n_labels)
        cl (Tensor): The ground truth labels for the batch. Shape: (batch_size,)
        labels_hierarchy (Tensor, optional): The hierarchical structure of the labels. Defaults to None.
            A binary tensor of shape (number of parents, n_labels)
            if not given, will act as a regular classification loss
            see gist for more details of how one can compute it
            https://gist.github.com/jkobject/5b36bc4807edb440b86644952a49781e

    Raises:
        ValueError: If the labels_hierarchy is not found while the number of predicted
            labels is smaller than the number of ground truth labels.

    Returns:
        Tensor: The computed binary cross entropy loss for the given batch.
    """
    maxsize = pred.shape[1]
    newcl = torch.zeros(
        (pred.shape[0], maxsize), device=cl.device
    )  # batchsize * n_labels
    # if we don't know the label we set the weight to 0 else to 1
    valid_indices = (cl != -1) & (cl < maxsize)
    valid_cl = cl[valid_indices]
    newcl[valid_indices, valid_cl] = 1

    weight = torch.ones_like(newcl, device=cl.device)
    # if we don't know the label we set the weight to 0 for all labels
    weight[cl == -1, :] = 0
    # if we have non leaf values, we don't know so we don't compute grad and set weight to 0
    # and add labels that won't be counted but so that we can still use them
    if labels_hierarchy is not None and (cl >= maxsize).any():
        is_parent = cl >= maxsize
        subset_parent_weight = weight[is_parent]
        # we set the weight of the leaf elements for pred where we don't know the leaf, to 0
        # i.e. the elements where we will compute the max
        # in cl, parents are values past the maxsize
        # (if there is 10 leafs labels, the label 10,14, or 15 is a parent at position
        # row 0, 4, or 5 in the hierarchy matrix
        subset_parent_weight[labels_hierarchy[cl[is_parent] - maxsize]] = 0
        weight[is_parent] = subset_parent_weight

        # we set their lead to 1 (since the weight will be zero, not really usefull..)
        subset_parent_newcl = newcl[is_parent]
        subset_parent_newcl[labels_hierarchy[cl[is_parent] - maxsize]] = 1
        newcl[is_parent] = subset_parent_newcl

        # all parental nodes that have a 1 in the labels_hierarchy matrix are set to 1
        # for each parent label / row in labels_hierarchy matrix, the addnewcl is
        # the max of the newcl values where the parent label is 1
        newcl_expanded = newcl.unsqueeze(-1).expand(-1, -1, labels_hierarchy.shape[0])
        addnewcl = torch.max(newcl_expanded * labels_hierarchy.T, dim=1)[0]

        # for their weight, it is decreasing based on number of children they have
        # it is the same here as for parental labels, we don't want to compute
        # gradients when they are 0 meaning not parents of the true leaf label.
        # for now we weight related to how many labels they contain.
        addweight = addnewcl.clone() / (labels_hierarchy.sum(1) ** 0.5)

        # except if it is the cl label we know about?
        subset_parent_weight = addweight[is_parent]
        subset_parent_weight[:, cl[is_parent] - maxsize] = 1
        addweight[is_parent] = subset_parent_weight

        # we apply the same mask to the pred but now we want to compute
        # logsumexp instead of max since we want to keep the gradients
        # we also set to -inf since it is a more neutral element for logsumexp
        pred_expanded = (
            pred.clone().unsqueeze(-1).expand(-1, -1, labels_hierarchy.shape[0])
        )
        pred_expanded = pred_expanded * labels_hierarchy.T
        pred_expanded[pred_expanded == 0] = torch.finfo(pred.dtype).min
        addpred = torch.logsumexp(pred_expanded, dim=1)

        # we add the new labels to the cl
        newcl = torch.cat([newcl, addnewcl], dim=1)
        weight = torch.cat([weight, addweight], dim=1)
        pred = torch.cat([pred, addpred], dim=1)
    elif labels_hierarchy is None and (cl >= maxsize).any():
        raise ValueError("need to use labels_hierarchy for this usecase")

    myloss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred, target=newcl, weight=weight
    )
    return myloss


class AdversarialDiscriminatorLoss(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = True,
    ):
        """
        Discriminator for the adversarial training for batch correction.

        Args:
            d_model (int): The size of the input tensor.
            n_cls (int): The number of classes.
            nlayers (int, optional): The number of layers in the discriminator. Defaults to 3.
            activation (callable, optional): The activation function. Defaults to nn.LeakyReLU.
            reverse_grad (bool, optional): Whether to reverse the gradient. Defaults
        """
        super().__init__()
        # module list
        self.decoder = nn.ModuleList()
        for _ in range(nlayers - 1):
            self.decoder.append(nn.Linear(d_model, d_model))
            self.decoder.append(nn.LayerNorm(d_model))
            self.decoder.append(activation())
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor, batch_labels: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
            batch_labels: Tensor, shape [batch_size]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self.decoder:
            x = layer(x)
        x = self.out_layer(x)
        return F.cross_entropy(x, batch_labels)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambd: float) -> Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: Tensor, lambd: float = 1.0) -> Tensor:
    """
    grad_reverse Reverses the gradient of the input tensor.

    Args:
        x (Tensor): The input tensor whose gradient is to be reversed.
        lambd (float, optional): The scaling factor for the reversed gradient. Defaults to 1.0.

    Returns:
        Tensor: The input tensor with its gradient reversed during the backward pass.
    """
    return GradReverse.apply(x, lambd)


# def embedding_independence(cell_embs, min_batch_size=32):
#    """
#    Compute independence loss between different embeddings using both
#    batch-wise decorrelation (when batch is large enough) and
#    within-sample dissimilarity
#
#    Args:
#        cell_embs: tensor of shape [batch_size, num_embeddings, embedding_dim]
#        min_batch_size: minimum batch size for using correlation-based loss
#    """
#    batch_size, num_embeddings, emb_dim = cell_embs.shape
#    # typically, 64*8*256
#    if batch_size >= min_batch_size:
#        # Compute pairwise distance matrices for each batch
#        gw_loss = 0
#        cell_embs = cell_embs.transpose(0, 1)
#        for i in range(num_embeddings):
#            # Get embeddings for this batch
#            embs = cell_embs[i]  # [num_embeddings, emb_dim]
#
#            # Compute GW distance between the two groups
#            # Compute GW distance between the two groups
#            # This measures structural differences between random subsets
#            gw_dist = gromov_wasserstein_distance(dist_mat1_np, dist_mat2_np, p, q)
#            gw_loss += torch.tensor(gw_dist, device=cell_embs.device)
#
#        return gw_loss / batch_size
#
#    else:
#        # Batch too small - use only within-sample dissimilarity
#        return within_sample(cell_embs)


def within_sample(cell_embs: Tensor):
    """
    Compute dissimilarity between embeddings within each sample
    using a combination of cosine and L2 distance

    Args:
        cell_embs: tensor of shape [batch_size, num_embeddings, embedding_dim]
    """
    batch_size, num_embeddings, emb_dim = cell_embs.shape

    # Normalize embeddings for cosine similarity
    cell_embs_norm = F.normalize(cell_embs, p=2, dim=-1)

    # Compute pairwise cosine similarities
    cos_sim = torch.bmm(cell_embs_norm, cell_embs_norm.transpose(1, 2))

    # Compute pairwise L2 distances (normalized by embedding dimension)
    l2_dist = torch.cdist(cell_embs, cell_embs, p=2) / np.sqrt(emb_dim)

    # Create mask for pairs (excluding self-similarity)
    mask = 1 - torch.eye(num_embeddings, device=cos_sim.device)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

    # Combine losses:
    # - High cosine similarity should be penalized
    # - Small L2 distance should be penalized
    cos_loss = (cos_sim * mask).pow(2).mean()
    l2_loss = 1.0 / (l2_dist * mask + 1e-3).mean()

    return 0.5 * cos_loss + 0.5 * l2_loss
