from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GraphSDEExprDecoder(nn.Module):
    def __init__(self, d_model: int, drift: nn.Module, diffusion: nn.Module):
        """
        Initialize the ExprNeuralSDEDecoder module.

        Args:
            d_model (int): The dimension of the model.
            drift (nn.Module): The drift component of the SDE.
            diffusion (nn.Module): The diffusion component of the SDE.
        """
        super().__init__()
        self.d_model = d_model
        self.drift = drift
        self.diffusion = diffusion

    def forward(self, x: Tensor, dt: float) -> Tensor:
        drift = self.drift(x)
        diffusion = self.diffusion(x)
        dW = torch.randn_like(x) * torch.sqrt(dt)
        return x + drift * dt + diffusion * dW


class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nfirst_tokens_to_skip: int = 0,
        dropout: float = 0.1,
        zinb: bool = True,
        use_depth: bool = False,
    ):
        """
        ExprDecoder Decoder for the gene expression prediction.

        Will output the mean, variance and zero logits, parameters of a zero inflated negative binomial distribution.

        Args:
            d_model (int): The dimension of the model. This is the size of the input feature vector.
            nfirst_tokens_to_skip (int, optional): The number of initial labels to skip in the sequence. Defaults to 0.
            dropout (float, optional): The dropout rate applied during training to prevent overfitting. Defaults to 0.1.
            zinb (bool, optional): Whether to use a zero inflated negative binomial distribution. Defaults to True.
            use_depth (bool, optional): Whether to use depth as an additional feature. Defaults to False.
        """
        super(ExprDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model if not use_depth else d_model + 1, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
        )
        self.pred_var_zero = nn.Linear(d_model, 3 if zinb else 1)
        self.zinb = zinb

    def forward(
        self, x: Tensor, req_depth: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        # we don't do it on the labels
        if req_depth is not None:
            x = torch.cat(
                [x, req_depth.unsqueeze(1).unsqueeze(-1).expand(-1, x.shape[1], -1)],
                dim=-1,
            )
        x = self.fc(x)
        if self.zinb:
            pred_value, var_value, zero_logits = self.pred_var_zero(x).split(
                1, dim=-1
            )  # (batch, seq_len)
            # The sigmoid function is used to map the zero_logits to a probability between 0 and 1.
            return dict(
                mean=F.softmax(pred_value.squeeze(-1), dim=-1),
                disp=torch.exp(torch.clamp(var_value.squeeze(-1), max=15)),
                zero_logits=zero_logits.squeeze(-1),
            )
        else:
            pred_value = self.pred_var_zero(x)
            return dict(mean=F.softmax(pred_value.squeeze(-1), dim=-1))


class MVCDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        tot_labels: int = 1,
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        use_depth: bool = False,
        zinb: bool = True,
    ) -> None:
        """
        MVCDecoder Decoder for masked value prediction of cell embeddings.

        Uses gene embeddings with cell embeddings to predict mean, variance, and zero logits
        parameters of a zero-inflated negative binomial distribution.

        Args:
            d_model (int): Dimension of the gene embedding.
            arch_style (str, optional): Architecture style of the decoder. Options:
                "inner product": Uses inner product between cell and gene embeddings,
                "concat query": Concatenates cell and gene embeddings,
                "sum query": Sums cell and gene embeddings.
                Defaults to "inner product".
            tot_labels (int, optional): Total number of labels in the input. Defaults to 1.
            query_activation (nn.Module, optional): Activation function for query vectors. Defaults to nn.Sigmoid.
            hidden_activation (nn.Module, optional): Activation function for hidden layers. Defaults to nn.PReLU.
            use_depth (bool, optional): Whether to use depth as an additional feature. Defaults to False.
            zinb (bool, optional): Whether to use a zero-inflated negative binomial distribution. Defaults to True.
        """
        super(MVCDecoder, self).__init__()
        if arch_style == "inner product":
            self.gene2query = nn.Linear(
                d_model if not use_depth else d_model + 1, d_model
            )
            self.norm = nn.LayerNorm(d_model)
            self.query_activation = query_activation()
            self.pred_var_zero = nn.Linear(
                d_model, d_model * (3 if zinb else 1), bias=False
            )
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(
                d_model if not use_depth else d_model + 1, d_model
            )
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model * (1 + tot_labels), d_model // 2)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(d_model // 2, (3 if zinb else 1))
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(
                d_model if not use_depth else d_model + 1, d_model
            )
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, (3 if zinb else 1))
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.d_model = d_model
        self.zinb = zinb

    def forward(
        self,
        cell_emb: Tensor,
        gene_embs: Tensor,
        req_depth: Optional[Tensor] = None,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
            req_depth: Tensor, shape (batch,), optional depth information.

        Returns:
            Dict[str, Tensor]: A dictionary containing the predicted mean, variance, and zero logits (if zinb is True).
        """
        if req_depth is not None:
            gene_embs = torch.cat(
                [
                    gene_embs,
                    req_depth.unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(-1, gene_embs.shape[1], -1),
                ],
                dim=-1,
            )
        if self.arch_style == "inner product":
            query_vecs = self.query_activation(self.norm(self.gene2query(gene_embs)))
            if self.zinb:
                pred, var, zero_logits = self.pred_var_zero(query_vecs).split(
                    self.d_model, dim=-1
                )
            else:
                pred = self.pred_var_zero(query_vecs)
            cell_emb = cell_emb.unsqueeze(2)
            if self.zinb:
                pred, var, zero_logits = (
                    torch.bmm(pred, cell_emb).squeeze(2),
                    torch.bmm(var, cell_emb).squeeze(2),
                    torch.bmm(zero_logits, cell_emb).squeeze(2),
                )
            else:
                pred = torch.bmm(pred, cell_emb).squeeze(2)
            # zero logits need to based on the cell_emb, because of input exprs
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.zinb:
                pred, var, zero_logits = self.fc2(h).split(1, dim=-1)
            else:
                pred = self.fc2(h)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.zinb:
                pred, var, zero_logits = self.fc2(h).split(1, dim=-1)
            else:
                pred = self.fc2(h)
        if self.zinb:
            return dict(
                mvc_mean=F.softmax(pred, dim=-1),
                mvc_disp=torch.exp(torch.clamp(var, max=15)),
                mvc_zero_logits=zero_logits,
            )
        else:
            return dict(mvc_mean=F.softmax(pred, dim=-1))


class ClsDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_cls: int,
        layers: List[int] = [256, 128],
        activation: Callable = nn.ReLU,
        dropout: float = 0.1,
    ):
        """
        ClsDecoder Decoder for classification task.

        Args:
            d_model (int): Dimension of the input.
            n_cls (int): Number of classes.
            layers (List[int]): List of hidden layers.
            activation (Callable): Activation function.
            dropout (float): Dropout rate.
        """
        super(ClsDecoder, self).__init__()
        # module List
        layers = [d_model] + layers
        self.decoder = nn.Sequential()
        self.n_cls = n_cls
        for i, l in enumerate(layers[1:]):
            self.decoder.append(nn.Linear(layers[i], l))
            self.decoder.append(nn.LayerNorm(l))
            self.decoder.append(activation())
            self.decoder.append(nn.Dropout(dropout))
        self.out_layer = nn.Linear(layers[-1], n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        x = self.decoder(x)
        return self.out_layer(x)


class VAEDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        layers: List[int] = [64, 64],
        activation: Callable = nn.ReLU,
        dropout: float = 0.1,
        return_latent: bool = False,
    ):
        """
        VAEDecoder for variational autoencoding of cell embeddings.

        Args:
            d_model (int): Input dimension (original embedding size)
            layers (List[int]): List of hidden layer sizes for encoder and decoder
            activation (Callable): Activation function to use
            dropout (float): Dropout rate
            return_latent (bool): Whether to return the latent vectors
        """
        super(VAEDecoder, self).__init__()

        # Encoder layers
        self.return_latent = return_latent
        encoder_layers = [d_model] + layers
        self.encoder = nn.Sequential()
        for i, (in_size, out_size) in enumerate(
            zip(encoder_layers[:-1], encoder_layers[1:])
        ):
            self.encoder.append(nn.Linear(in_size, out_size))
            self.encoder.append(nn.LayerNorm(out_size))
            self.encoder.append(activation())
            self.encoder.append(nn.Dropout(dropout))

        # VAE latent parameters
        self.fc_mu = nn.Linear(encoder_layers[-1], encoder_layers[-1])
        self.fc_var = nn.Linear(encoder_layers[-1], encoder_layers[-1])

        # Decoder layers
        decoder_layers = [encoder_layers[-1]] + list(reversed(layers[:-1])) + [d_model]
        self.decoder = nn.Sequential()
        for i, (in_size, out_size) in enumerate(
            zip(
                decoder_layers[:-1], decoder_layers[1:]
            )  # Changed to include final layer
        ):
            self.decoder.append(nn.Linear(in_size, out_size))
            if (
                i < len(decoder_layers) - 2
            ):  # Don't apply activation/norm to final layer
                self.decoder.append(nn.LayerNorm(out_size))
                self.decoder.append(activation())
                self.decoder.append(nn.Dropout(dropout))

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu (Tensor): Mean of the latent Gaussian
            log_var (Tensor): Log variance of the latent Gaussian

        Returns:
            Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Compute KL divergence between N(mu, var) and N(0, 1).

        Args:
            mu (Tensor): Mean of the latent Gaussian
            log_var (Tensor): Log variance of the latent Gaussian

        Returns:
            Tensor: KL divergence loss
        """
        # KL(N(mu, var) || N(0, 1)) = -0.5 * sum(1 + log(var) - mu^2 - var)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return kl_loss.mean()

    def forward(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """
        Forward pass through VAE.

        Args:
            x (Tensor): Input tensor of shape [batch_size, d_model]

        Returns:
            If self.return_latent is True:
                Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
                    - reconstructed_x (Tensor): Reconstructed input, shape [batch_size, d_model]
                    - mu (Tensor): Mean of the latent Gaussian, shape [batch_size, latent_dim]
                    - log_var (Tensor): Log variance of the latent Gaussian, shape [batch_size, latent_dim]
                    - kl_loss (Tensor): KL divergence loss (scalar tensor)
            Else:
                Tensor: reconstructed_x of shape [batch_size, d_model]
        """
        # Encode
        encoded = self.encoder(x)

        # Get latent parameters
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)
        log_var = torch.clamp(log_var, min=-10)

        # Sample latent vector
        kl_loss = self.kl_divergence(mu, log_var)
        # free_bits = 2.0  # per latent dim
        # kl_loss = torch.clamp(kl_loss / mu.size(-1), min=free_bits) * mu.size(-1)
        z = self.reparameterize(mu, log_var)

        # Decode
        decoded = self.decoder(z)

        if self.return_latent:
            return decoded, mu, log_var, encoded, kl_loss
        return decoded, kl_loss
