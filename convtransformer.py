import math
from typing import Tuple, List, Dict, Union, Optional

import torch
import torch.nn as nn


class ConvTransformerModel(nn.Module):
    """
    Implementation of the convtransformer model proposed by Gao et al.
    in 'Character-Level Translation with Self-attention'.

    Reference:
    Yingqiang Gao, Nikola I. Nikolov, Yuhuang Hu and Richard H.R.
    Hahnloser. 2020. Character-level translation with self-attention.
    In Proceedings of the 58th Annual Meeting of the Association for
    Computational Linguistics, pages 1591-1604, Online. Association
    for Computational Linguistics.

    http://dx.doi.org/10.18653/v1/2020.acl-main.145
    """

    def __init__(
        self,
        sz_alph: int,
        sz_emb: int,
        max_len: int,
        idx_pad: int,
        num_lay: int,
        sz_kernels: List[int],
        sz_kernel_final: int,
        nhead: int,
        dim_ff: int,
        label_smoothing: float,
        dropout: float,
    ) -> None:
        """Initializes a new ConvTransformerModel.

        Args:
            sz_alph: The size of the alphabet used.
            sz_emb: The character embedding size.
            max_len: The maximum length of a sentence (including the
                'start'- and 'end'-character).
            idx_pad: The index corresponding to the 'pad'-character.
            num_lay: The number of layers in the encoder and decoder.
            sz_kernels: The kernel sizes of the first convolutions in
                the convolutional sublayers in the encoder.
            sz_kernels_final: Kernel size of the final convolution in
                the convolutional sublayers in the encoder.
            nhead: The number of multihead attention heads.
            dim_ff: The dimension of the network in the feedforward
                sublayers.
            label_smoothing: The label smoothing factor.
            dropout: The dropout value.
        """
        super(ConvTransformerModel, self).__init__()
        self.sz_alph = sz_alph
        self.sz_emb = sz_emb
        self.max_len = max_len
        self.idx_pad = idx_pad

        # We use the same embedding layer for both the source and target
        # sequences.
        self.embedding = PositionalEmbedding(sz_alph, sz_emb, max_len, idx_pad, dropout)

        self.convtransformer = ConvTransformer(
            sz_emb, nhead, num_lay, dim_ff, sz_kernels, sz_kernel_final, dropout
        )
        self.linear = nn.Linear(sz_emb, sz_alph, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # We use cross entropy with label smoothing as loss
        # function.
        self.smooth_nllloss = SmoothNLLLoss(sz_alph, idx_pad, label_smoothing)

        # We use Xavier initialization to initialize the weights
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self, batch_src: torch.Tensor, batch_tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes the source and target sequences.

        Args:
            batch_src: Batch of indexed source sentences of shape (N, S).
            batch_tgt: Batch of indexed target sentences of shape (N, T).

        Returns:
            A tuple (loss, log_distr), where 'loss' is the 1-dimensional
            loss tensor, and where 'log_distr' is a tensor of shape
            (N, T, E) storing the log probability distributions.
        """
        src_emb = self.embedding(batch_src)
        tgt_emb = self.embedding(batch_tgt)

        batch_tgt_padding_mask = self.create_padding_mask(batch_tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb)
        output = self.convtransformer(
            src_emb.transpose(0, 1),
            tgt_emb.transpose(0, 1),
            batch_tgt_padding_mask,
            tgt_mask,
        ).transpose(0, 1)

        output = self.linear(output)
        log_distr = self.log_softmax(output)

        batch_tgt = torch.roll(batch_tgt, shifts=-1, dims=1)
        batch_tgt[:, -1] = self.idx_pad
        loss = self.smooth_nllloss(log_distr, batch_tgt)

        return loss, log_distr

    def create_padding_mask(self, batch: torch.tensor) -> torch.Tensor:
        """Creates a padding mask for the given batch.

        The resulting mask is a tensor of type bool, where all indices
        which correspond to the 'pad'-character are marked with 'True',
        while all others are marked with 'False'.

        Args:
            batch: Batch tensor of shape (N, C).

        Returns:
            Padding mask tensor of shape (N, C).
        """
        return batch == self.idx_pad

    def generate_square_subsequent_mask(self, tgt_emb: torch.Tensor) -> torch.Tensor:
        """Generates a square mask for a batch of target sequences.

        The masked positions are filled with float('-inf'), while the
        unmasked positions are filled with float(0.0).

        Args:
            tgt_emb: Embedded batch tensor of shape (N, T, E).

        Returns:
            Mask tensor of shape (T, T).
        """
        sz = tgt_emb.size(1)

        mask = torch.triu(tgt_emb.new_ones(sz, sz)) == 1
        mask = mask.transpose(0, 1).float()

        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask


class ConvTransformer(nn.Module):
    """The actual ConvTransformer module.

    Compared to the traditional transformer, the only difference lies in
    the modified encoder blocks which include an additional
    convolutional sublayer.
    """

    def __init__(
        self,
        sz_emb: int,
        nhead: int,
        num_lay: int,
        dim_ff: int,
        sz_kernels: List[int],
        sz_kernel_final: int,
        dropout: float,
    ) -> None:
        """Initializes a new ConvTransformer module.

        Args:
            sz_emb: The size of the character embeddings.
            nhead: The number of multihead attention heads.
            num_lay: The number of layers in the encoder and decoder.
            dim_ff: The dimension of the network in the feedforward
                sublayers.
            sz_kernels: The kernel sizes of the first convolutions in
                the convolutional sublayers in the encoder.
            sz_kernels_final: Kernel size of the final convolution in
                the convolutional sublayers in the encoder.
            dropout: The dropout value.
        """
        super(ConvTransformer, self).__init__()

        # Encoder layers
        enc_layer = ConvTransformerEncoderLayer(
            sz_emb, nhead, dim_ff, dropout, sz_kernels, sz_kernel_final
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_lay)

        # Decoder layers
        dec_layer = nn.TransformerDecoderLayer(sz_emb, nhead, dim_ff, dropout)
        self.decoder = nn.TransformerDecoder(dec_layer, num_lay)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Process the source and target sequences.

        Args:
            src: Batch of embedded source sentences of shape (S, N, E)
            tgt: Batch of embedded target sentences of shape (T, N, E)
            tgt_key_padding_mask: Padding mask of shape (N, T)
            tgt_mask: Target mask of shape (T, T)

        Returns:
            Tensor of shape (T, N, E)
        """
        memory = self.encoder(src)
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return output


class ConvTransformerEncoderLayer(nn.Module):
    """ConvTransformer encoder layer.

    In each encoder block of the convtransformer, we first apply
    several 1-dimensional convolutional layers with different kernel
    sizes. The outputs of these layers are then concatenated and passed
    through another 1-dimension convolutional layer. Afterwards we
    proceed as usual with multi-head attention and a feed forward
    network.
    """

    def __init__(
        self,
        sz_emb: int,
        nhead: int,
        dim_ff: int,
        dropout: float,
        sz_kernels: List[int],
        sz_kernel_final: int,
    ) -> None:
        """Initializes a new ConvTransformer encoder layer.

        Args:
            sz_emb: The size of the character embeddings.
            nhead: The number of multihead attention heads.
            dim_ff: The dimension of the network in the feedforward
                sublayers.
            dropout: The dropout value.
            sz_kernels: The kernel sizes of the first convolutions in
                the convolutional sublayers in the encoder.
            sz_kernels_final: Kernel size of the final convolution in
                the convolutional sublayers in the encoder.
        """
        super(ConvTransformerEncoderLayer, self).__init__()

        # First convolutional layer with different kernel sizes
        list_conv_layers = []
        for k in sz_kernels:
            kernel_size = (k, sz_emb)
            padding = ((k - 1) // 2, 0)
            conv2d = nn.Conv2d(
                in_channels=1,
                out_channels=sz_emb,
                kernel_size=kernel_size,
                padding=padding,
            )
            relu = nn.ReLU(inplace=True)
            list_conv_layers.append(nn.Sequential(conv2d, relu))
        self.conv_layers = nn.ModuleList(list_conv_layers)

        # Second convolutional layer which merges the outputs of the
        # first layer
        kernel_size = (sz_kernel_final, len(sz_kernels) * sz_emb)
        padding = ((sz_kernel_final - 1) // 2, 0)
        conv2d_final = nn.Conv2d(
            in_channels=1, out_channels=sz_emb, kernel_size=kernel_size, padding=padding
        )
        relu_final = nn.ReLU(inplace=True)
        self.conv_layer_final = nn.Sequential(conv2d_final, relu_final)

        # From here on the architecture is the same as in the usual
        # transformer encoder layers.
        self.enc_layer = nn.TransformerEncoderLayer(
            d_model=sz_emb, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Passes the input through the encoder layer.

        We do not make use of any masks, as this would interfere with
        the convolutional networks in the first sublayer. The arguments
        'src_mask' and 'src_key_padding_mask' are included only to
        ensure compatibility with the TransformerEncoder module.

        Arguments:
            src: Batch of embedded source sentences of shape (S, N, E).
            src_mask: Ignored.
            src_key_padding_mask: Ignored.

        Returns:
            output: Tensor of shape (S, N, E).
        """
        output_conv = []
        for layer in self.conv_layers:
            output = layer(src.permute(1, 0, 2).unsqueeze(1))
            output_conv.append(output[:, :, :, 0])
        output_conv = torch.cat(output_conv, 1).transpose(1, 2).unsqueeze(1)

        output = self.conv_layer_final(output_conv)
        output = output[:, :, :, 0].permute(2, 0, 1)
        output = self.enc_layer(src + output)

        return output


class PositionalEmbedding(nn.Module):
    """Embedding layer with added positional encoding."""

    def __init__(
        self, sz_alph: int, sz_emb: int, max_len: int, padding_idx: int, dropout: float
    ) -> None:
        """Initializes the layer.

        Args:
            sz_alph: The size of the alphabet.
            sz_emb: The size of each embedding vector.
            max_len: The maximum length of a sentence (including the
                'start'- and 'end'-character).
            padding_idx: The index of the 'pad'-character.
            dropout: The dropout value.
        """
        super(PositionalEmbedding, self).__init__()
        self.sz_emb = sz_emb
        self.embedding = nn.Embedding(sz_alph, sz_emb, padding_idx=padding_idx)
        self.positional_encoding = PositionalEncoding(sz_emb, dropout, max_len)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Computes the embedding vectors.

        Args:
            batch: Batch tensor of shape (N, C)

        Returns:
            Embedded batch tensor of shape (N, C, E)
        """
        output = self.embedding(batch) * math.sqrt(self.sz_emb)
        return self.positional_encoding(output)


class PositionalEncoding(nn.Module):
    """Positional encoding layer with dropout.

    Source:
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__(self, sz_emb: int, dropout: float, max_len: int) -> None:
        """Initializes the layer.

        Args:
            sz_emb: The size of the character embeddings.
            dropout: The dropout value.
            max_len: The maximum length of a sentence (including the
                'start'- and 'end'-character).
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_enc = torch.zeros(max_len, sz_emb)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, sz_emb, 2, dtype=torch.float)
            * (-math.log(10000.0) / sz_emb)
        )

        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term[: sz_emb // 2])
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the passed tensor.

        Args:
            batch: Batch tensor of shape (N, C, E)

        Returns:
            Tensor of shape (N, C, E) with positional encoding added.
        """
        batch_pos_enc = self.pos_enc[: batch.size(1)]
        batch_pos_enc = batch_pos_enc.unsqueeze(0).repeat(batch.size(0), 1, 1)
        return self.dropout(batch + batch_pos_enc)


class SmoothNLLLoss(nn.Module):
    """Negative log likelihood loss layer with label smoothing."""

    def __init__(self, sz_alph: int, ignore_index: int, label_smoothing: float) -> None:
        """Initializes the layer.

        Args:
            sz_alph: The size of the alphabet.
            ignore_index: The index to be ignored when computing the
                loss.
            label_smoothing: The label smoothing factor.
        """
        super(SmoothNLLLoss, self).__init__()
        self.kl_div_loss = nn.KLDivLoss(reduction="sum")
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.sz_alph = sz_alph

    def forward(self, batch: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Computes the loss.

        Args:
            batch: Batch of shape (N, T, E) containing the log
                probability distributions.
            tgt: Batch of shape (N, T) containing the targets.

        Returns:
            The computed loss tensor.
        """
        # Since we compute the loss character-wise, we can flatten the
        # passed tensors.
        batch = batch.view(-1, self.sz_alph)
        tgt = tgt.view(-1)

        # Create batch of smoothed one-hot vectors.
        smooth_labels = batch.new_empty(batch.shape)
        smooth_labels.fill_(self.label_smoothing / (self.sz_alph - 2))
        smooth_labels.scatter_(1, tgt.unsqueeze(1), 1.0 - self.label_smoothing)
        smooth_labels[:, self.ignore_index] = 0

        # Zero out the those smoothed one-hot vectors whose
        # corresponding target is equal to 'ignore_index'.
        mask = torch.nonzero(tgt == self.ignore_index)
        nr_tokens = max(batch.size(0) - mask.size(0), 1)

        if mask.size(0) > 0:
            smooth_labels.index_fill_(0, mask.squeeze(), 0.0)

        # Use the Kullback-Leibler divergence to compute the final loss.
        output = self.kl_div_loss(batch, smooth_labels) / nr_tokens

        return output
