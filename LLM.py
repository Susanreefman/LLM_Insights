#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM.py
Script to train and use Large Language model
"""
import sys

# loading dataset
import pandas as pd
import argparse
# split dataset
from sklearn.model_selection import train_test_split as tts

# pytorch build model
import torch
import collections
from collections import Counter
from torch import nn
from torch.utils.data import Dataset, DataLoader

import math
from time import perf_counter

# evaluation method
from evaluate import load
import evaluate


# Tokenize function
def tokenize(lines, token='word'):
    """
    Make from sentence a list with separate words, like ["Make", "from", "sentence"]
    Parameters:
        lines (list): input strings
        token (string): input token
    Returns:
        (list): split sentences
    """
    assert token in ('word', 'char'), 'Unknown token type: ' + token
    return [line.split() if token == 'word' else list(line) for line in lines]


# padding function
def truncate_pad(line, num_steps, padding_token):
    """
    Truncate and pad sentences
    Parameters:
        line (list): input sentence
        num_steps (int): number of steps to truncate
        token (string): input token
    Returns:
        (list): split sentences
    """
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


def build_array_sum(lines, vocab, num_steps):
    """
    function to add eos and padding and also determine valid length of each data sample
    Parameters
        lines (list): input sentences
        vocab ():
        num_steps ():
    Returns
        array ():
        valid_len ():
    """
    lines = [vocab[l] for l in lines]
    # Add end of sentence
    lines = [l + [vocab['<eos>']] for l in lines]
    # Create tensor with padding
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    # Make all lines equal length by padding
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_array(data_arrays, batch_size, is_train=True):
    """"
    Loading arrays in Tensor
    Parameters
        data_arrays (list): list of arrays including data
        batch_size (int): number of samples in each batch
        is_train (bool): mode of running the model
    Returns
        (Tensor): including the data in tensor format
    """
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


def transpose_qkv(matrix, num_heads):
    """
    Function to transpose the linearly transformed query, key and values matrices
    Parameters
        matrix (Tensor): matrix with represented values
        num_heads (int): number of attention heads
    Returns
        (Tensor): transposed matrix
    """
    matrix = matrix.reshape(matrix.shape[0], matrix.shape[1], num_heads, -1)
    matrix = matrix.permute(0, 2, 1, 3)
    return matrix.reshape(-1, matrix.shape[2], matrix.shape[3])


def transpose_output(matrix, num_heads):
    """
    Function to transpose the linearly transformed output matrices
    Parameters
        matrix (Tensor): matrix of output values
        num_heads (int): number of attention heads
    Returns
        (Tensor): transposed matrix
    """
    matrix = matrix.reshape(-1, num_heads, matrix.shape[1], matrix.shape[2])
    matrix = matrix.permute(0, 2, 1, 3)
    return matrix.reshape(matrix.shape[0], matrix.shape[1], -1)


def sequence_mask(matrix, valid_len, value=0):
    """
    Mask irrelevant padded tokens
    Parameters
        matrix (Tensor): matrix of tokens
        valid_len (int): valid length of sentence
        value (int): value of mask
    Returns
        (Tensor): output tensor
    """
    # Here masking is used so that irrelevant padding tokens are not considered while calculations
    mask = torch.arange(matrix.size(1), dtype=torch.float32)[None, :] < valid_len[:, None]  # device=X.device
    matrix[~mask] = value
    return matrix


def masked_softmax(matrix, valid_len):
    """
    Mask irrelevant padded tokens by giving it a small negative value
    Parameters
        matrix (Tensor): matrix with tokens
        valid_len (int): valid length of dimension
    Returns
        (Tensor): output tensor
    """
    if valid_len is None:
        return nn.functional.softmax(matrix, dim=-1)
    else:
        shape = matrix.shape
        if valid_len.dim() == 1:
            valid_len = torch.repeat_interleave(valid_len, shape[1])
        else:
            valid_len = valid_len.reshape(-1)
        matrix = sequence_mask(matrix.reshape(-1, shape[-1]), valid_len, value=-1e6)
        return nn.functional.softmax(matrix.reshape(shape), dim=-1)


def get_device(i=0):
    """
    Returns the specified CUDA device if available, otherwise defaults to CPU.
    Parameters:
        i (int): The index of the CUDA device to use. Defaults to 0.
    Returns:
        torch.device: CUDA device if available, otherwise the CPU device.
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')


def grad_clipping(net, theta):
    """
    Clips the gradients of a neural network to prevent gradient explosion.
    Parameters:
        net (nn.Module): The neural network whose gradients are to be clipped.
        theta (float): The threshold value for clipping
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


class Vocab:
    """A vocabulary class for mapping tokens to indices and vice versa."""

    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        """Initializes the Vocab instance with a list of tokens, a minimum frequency threshold, and reserved tokens. """
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        """Returns the number of unique tokens in the vocabulary."""
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """Returns the index of a token or a list of indices for a list of tokens.
        If the token is not found, returns the index for the unknown token. """
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """Returns the token corresponding to an index or a list of tokens for a list of indices. """
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    def unk(self):  # Index for the unknown token
        """Returns the index for the unknown token."""
        return self.token_to_idx['<unk>']

    def print_variable(self):
        """Prints the idx_to_token and token_to_idx attributes."""
        print("Variable idx_to_token:", self.idx_to_token)
        print("Variable idx_to_token:", self.token_to_idx)


class MultiHeadAttention(nn.Module):
    """mechanism that allows the model to jointly attend to information
    from different representation subspaces."""
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        """Initializes the MultiHeadAttention instance with the given parameters."""
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.w_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.w_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.w_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """Computes the multi-head attention for the given queries, keys, and values."""
        queries = transpose_qkv(self.w_q(queries), self.num_heads)
        keys = transpose_qkv(self.w_k(keys), self.num_heads)
        values = transpose_qkv(self.w_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)

        return self.w_o(output_concat)


class DotProductAttention(nn.Module):
    """Dot-product attention mechanism that scales the dot products of the query and key vectors """
    # The dot product attention scoring function
    def __init__(self, dropout, **kwargs):
        """Initializes the DotProductAttention instance with the given dropout rate."""
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_len=None):
        """Computes the dot-product attention for the given queries, keys, and values"""
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_len)

        return torch.bmm(self.dropout(self.attention_weights), values)


class PositionWiseFFN(nn.Module):
    """Position-wise feed-forward network used in transformer models."""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_output, **kwargs):
        """Initializes the PositionWiseFFN instance with the given input, hidden, and output sizes."""
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_output)

    def forward(self, X):
        """Applies the feed-forward network to the input tensor X."""
        return self.dense2(self.relu(self.dense1(X)))


class PositionalEncoding(nn.Module):
    """Adds positional encoding to the input tensor to incorporate the position information in the sequences."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        """Initializes the PositionalEncoding instance with the given parameters for number of hidden layers
        and dropout rate"""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2,
                                                                                                      dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        """Adds positional encoding to the input tensor X and applies dropout."""
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class EncoderBlock(nn.Module):
    """Encoder Block"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        """Initializes the EncoderBlock instance with the given parameters."""
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        """Applies the encoder block to the input tensor X with the given valid lengths."""
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(nn.Module):
    """Transformer encoder consisting of stacked encoder blocks"""
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout, use_bias=False, **kwargs):
        """Initializes the TransformerEncoder instance with the given parameters."""
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                                              ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        """Applies the transformer encoder to the input tensor X with the given valid lengths."""
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):
    """Decoder block"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        """Initializes the DecoderBlock instance with the given parameters."""
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        """Applies the decoder block to the input tensor X and updates the state."""
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:  # true when training the model
            key_values = X
        else:  # while decoding state[2][self.i] is decoded output of the ith block till the present time-step
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(nn.Module):
    """Transformer decoder consisting of stacked decoder blocks"""
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        """Initializes the TransformerEncoder instance with the given parameters."""
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 DecoderBlock(key_size, query_size, value_size,
                                              num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                                              dropout, i))
            self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        """Initializes the decoder state with encoder outputs and valid lengths."""
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        """Applies the transformer decoder to the input tensor X with the given state."""
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    def attention_weights(self):
        """Returns the attention weights from all decoder blocks."""
        return self._attention_weights


class AddNorm(nn.Module):
    """The residual connection followed by layer normalization."""

    def __init__(self, norm_shape, dropout):
        """Initializes the AddNorm instance with the given shape for layer normalization and dropout rate."""
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        """Applies the residual connection and layer normalization to the input tensor X with the residual Y."""
        return self.ln(self.dropout(Y) + X)


class Accumulator:
    """Accumulator class to accumulate values in a list."""
    def __init__(self, n):
        """Initializes Accumulator with n zeros in data."""
        self.data = [0.0] * n

    def add(self, *args):
        """Adds values in args to corresponding indices in data"""
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        """Resets all values in data to zero."""
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        """Returns the value at index idx in data."""
        return self.data[idx]


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """Masked softmax cross-entropy loss function."""

    def forward(self, pred, label, valid_len):
        """
         Computes the masked softmax cross-entropy loss.
        `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
        `label` shape: (`batch_size`, `num_steps`)
        `valid_len` shape: (`batch_size`,)
        """
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


class Transformer(nn.Module):
    """Transformer model composed of an encoder and a decoder."""
    def __init__(self, encoder, decoder):
        """Initializes the Transformer model with given encoder and decoder."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        """Performs forward pass of the Transformer model."""
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]


def train_model(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """
    Train a sequence-to-sequence model.
    Parameters:
    net (nn.Module): The sequence-to-sequence model to be trained.
    data_iter (iterable): The data iterator providing batches of data.
    lr (float): Learning rate for the optimizer.
    num_epochs (int): Number of training epochs.
    tgt_vocab (Vocab): Vocabulary object for the target language.
    device (torch.device): Device (CPU or GPU) on which to train the model.

    Returns:
        (list): List of losses recorded during training.
    """
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    train_losses = []
    for epoch in range(num_epochs):
        metric = Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            train_losses.append(l)
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if epoch % 5 == 0:
            print(f"Done with epoch number: {epoch}")  # optional step
    print(f'loss {metric[0] / metric[1]:.3f} on {str(device)}')
    return train_losses


def predicting_model(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    """
    Perform sequence prediction using a sequence-to-sequence model.
    Parameters:
        net (nn.Module): The trained sequence-to-sequence model.
        src_sentence (str): Source sentence to translate.
        src_vocab (Vocab): Vocabulary object for the source language.
        tgt_vocab (Vocab): Vocabulary object for the target language.
        num_steps (int): Maximum number of decoding time steps.
        device (torch.device): Device (CPU or GPU) on which to perform inference.
        save_attention_weights (bool, optional): Whether to save attention weights during decoding. Default is False.
    Returns:
        (tuple): A tuple containing the predicted target sequence (str) and attention weights (list of tensors).

    """
    # Set `net` to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    src_tokens = [x for x in src_tokens if str(x).isdigit()]
    # Unsqueeze adds another dimension that works as the the batch axis here
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis to the decoder now
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y = net.decoder(dec_X, dec_state)[0]
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
            # Once the end-of-sequence token is predicted, the generation of the output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    if len(output_seq) < 2:

        if len(output_seq) == 1:
            return ''.join(tgt_vocab.to_tokens(output_seq[0])), attention_weight_seq
        else:

            return "No output!", attention_weight_seq
    else:
        return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def evaluate_model(actual, predictions):
    """
    Evaluate the model predictions against actual references using ROUGE metric.
    Parameters:
        actual (torch.Tensor): Actual target sequences as a tensor.
        predictions (list): Predicted target sequences as a list of strings.
    Returns:
        results (dict): Dictionary containing ROUGE scores.
    """
    rouge = evaluate.load('rouge')
    y = actual.tolist()
    ref = []
    for i in y:
        ref.append([i])
    results = rouge.compute(predictions=predictions, references=ref)
    return results


def parse_args():
    """
    parse command-line arguments for input and output files

    Returns:
        parser.parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file",
                        help="""The input data in CSV format""",
                        required=True)
    return parser.parse_args()



def main():
    """Main"""
    args = parse_args()
    data = pd.read_csv(args.f, delimiter=';')

    x_train, x_test, y_train, y_test = tts(data['Text'], data['Summary'], test_size=0.1, shuffle=True, random_state=111)

    # tokenize
    src_tokens = tokenize(x_train)
    tgt_tokens = tokenize(y_train)

    # build vocabulary on dataset
    src_vocab = Vocab(src_tokens, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(tgt_tokens, reserved_tokens=['<pad>', '<bos>', '<eos>'])

    max_len_text = 150
    max_len_summary = 50

    src_array, src_valid_len = build_array_sum(src_tokens, src_vocab, max_len_text)
    tgt_array, tgt_valid_len = build_array_sum(tgt_tokens, tgt_vocab, max_len_summary)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)

    # Create tensor dataset object
    batch_size = 16
    data_iter = load_array(data_arrays, batch_size)

    device = get_device()

    # Initialize model
    num_hiddens, num_layers, dropout, num_steps = 32, 2, 0.1, 100
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]
    encoder = TransformerEncoder(len(src_vocab), key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                                 ffn_num_hiddens, num_heads, num_layers, dropout)
    decoder = TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                                 ffn_num_hiddens, num_heads, num_layers, dropout)

    net = Transformer(encoder, decoder)

    starttime = perf_counter()
    lr = 0.005
    num_epochs = 50
    train_losses = train_model(net, data_iter, lr, num_epochs, tgt_vocab, device)
    print(f'Verstreken tijd: {(perf_counter() - starttime) / 60.0:.1f} minuten.')

    sample = x_test[:10]
    actual = y_test[:10]
    predictions = []
    for s, a in zip(sample, actual):
        pred_sum, _ = predicting_model(net, s, src_vocab, tgt_vocab, max_len_summary, device)
        predictions.append(pred_sum)
        print("SAMPLE : {}".format(s))
        print("ACTUAL : {}".format(a))
        print("PREDICTED : {}".format(pred_sum))
        print('')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript terminated by the user.")
        sys.exit(1)
