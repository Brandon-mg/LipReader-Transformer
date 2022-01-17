import torch
import torch.nn as nn
import hparams
import torch.nn.functional as F
import math
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, embed_dims, heads):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.embed_dims = embed_dims
        self.depth = embed_dims // heads

        self.query = nn.Linear(self.depth, self.depth)
        self.key = nn.Linear(self.depth, self.depth)
        self.value = nn.Linear(self.depth, self.depth)

        self.fc_out = nn.Linear(self.depth * self.heads * 2, self.embed_dims)

    def forward(self, query, key, value, mask, isDecoder=False):
        batch, q_len, k_len, v_len = query.shape[0], query.shape[1], key.shape[1], value.shape[1]

        query = query.reshape(batch, q_len, self.heads, self.depth)
        key = key.reshape(batch, k_len, self.heads, self.depth)
        value = value.reshape(batch, v_len, self.heads, self.depth)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        energy = torch.einsum('bqhd, bkhd -> bhqk', [query, key])
        if isDecoder:
            print("mask", mask)
            print("mask", mask.shape)
            print("energy before mask fill", energy)
            print("energy before mask fill", energy.shape)
        if mask is not None:
            if isDecoder:
                print("Performing mask fill")
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        if isDecoder:
            print("energy after mask fill", energy)
            print("energy after mask fill", energy.shape)


        energy = torch.softmax((energy / ((self.depth ** 1 / 2))), dim=-1)

        out = torch.einsum('bhqv, bvhd -> bqhd', [energy, value])

        out = out.reshape(batch, q_len, self.heads * self.depth)
        query = query.reshape(batch, q_len, self.heads * self.depth)

        out = torch.cat([query, out], dim=-1)
        out = self.fc_out(out)

        return out, energy


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dims, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.hidden_dims = hidden_dims
        self.heads = heads
        self.multihead_attention = SelfAttention(hidden_dims, heads)
        self.feed_forward = nn.Sequential(
            nn.Conv1d(hidden_dims, hidden_dims * forward_expansion, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dims * forward_expansion, hidden_dims, kernel_size=1)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dims)
        self.layer_norm2 = nn.LayerNorm(hidden_dims)

    def forward(self, query, key, value, mask):
        attention_out, attn = self.multihead_attention(query, key, value, mask)
        add = self.dropout(self.layer_norm1(attention_out + query))
        ffn_in = add.transpose(1, 2)
        ffn_out = self.feed_forward(ffn_in)
        ffn_out = ffn_out.transpose(1, 2)
        out = self.dropout(self.layer_norm2(ffn_out + add))
        return out, attn


class EncoderPreNet(nn.Module):
    """Encoder module:
        - Three 3-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, dropout = 0.5):
        super(EncoderPreNet, self).__init__()
        self.out_channel = hparams.num_init_filters
        self.in_channel = 3
        self.dropout = nn.Dropout(dropout)
        convolutions = []

        for i in range(hparams.encoder_n_convolutions):
            if i == 0:
                conv_layer = nn.Sequential(
                    ConvNorm3D(self.in_channel, self.out_channel,
                               kernel_size=5, stride=(1, 2, 2),
                               # padding=int((hparams.encoder_kernel_size - 1) / 2),
                               dilation=1, w_init_gain='relu'),
                    ConvNorm3D(self.out_channel, self.out_channel,
                               kernel_size=3, stride=1,
                               # padding=int((hparams.encoder_kernel_size - 1) / 2),
                               dilation=1, w_init_gain='relu', residual=True),
                    ConvNorm3D(self.out_channel, self.out_channel,
                               kernel_size=3, stride=1,
                               # padding=int((hparams.encoder_kernel_size - 1) / 2),
                               dilation=1, w_init_gain='relu', residual=True)
                )
                convolutions.append(conv_layer)
            else:
                conv_layer = nn.Sequential(
                    ConvNorm3D(self.in_channel, self.out_channel,
                               kernel_size=3, stride=(1, 2, 2),
                               # padding=int((hparams.encoder_kernel_size - 1) / 2),
                               dilation=1, w_init_gain='relu'),
                    ConvNorm3D(self.out_channel, self.out_channel,
                               kernel_size=3, stride=1,
                               # padding=int((hparams.encoder_kernel_size - 1) / 2),
                               dilation=1, w_init_gain='relu', residual=True),
                    ConvNorm3D(self.out_channel, self.out_channel,
                               kernel_size=3, stride=1,
                               # padding=int((hparams.encoder_kernel_size - 1) / 2),
                               dilation=1, w_init_gain='relu', residual=True)
                )
                convolutions.append(conv_layer)

            if i == hparams.encoder_n_convolutions - 1:
                conv_layer = nn.Sequential(
                    ConvNorm3D(self.out_channel, self.out_channel,
                               kernel_size=3, stride=(1, 3, 3),
                               # padding=int((hparams.encoder_kernel_size - 1) / 2),
                               dilation=1, w_init_gain='relu'))
                convolutions.append(conv_layer)

            self.in_channel = self.out_channel
            self.out_channel *= 2
        self.convolutions = nn.ModuleList(convolutions)

    def forward(self, x):
        for conv in self.convolutions:
            x = self.dropout(conv(x))
        # for i in range(len(self.convolutions)):
        # 	if i==0 or i==1 or i ==2:
        # 		with torch.no_grad():
        # 			x = F.dropout(self.convolutions[i](x), 0.5, self.training)
        # 	else:
        # 		x = F.dropout(self.convolutions[i](x), 0.5, self.training)

        x = x.permute(0, 2, 1, 3, 4).squeeze(4).squeeze(3).contiguous()  # [bs x 90 x encoder_embedding_dim]
        # print(x.shape)

        return x


class ConvNorm3D(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', activation=torch.nn.ReLU, residual=False):
        super(ConvNorm3D, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.residual = residual
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels,
                                      kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation,
                                      bias=bias)
        self.batched = torch.nn.BatchNorm3d(out_channels)
        self.activation = activation()

        torch.nn.init.xavier_uniform_(
            self.conv3d.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        # torch.nn.init.xavier_uniform_(
        #     self.batched.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        # torch.nn.init.xavier_uniform_(
        #     self.activation.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv3d(signal)

        batched = self.batched(conv_signal)

        if self.residual:
            batched = batched + signal
        activated = self.activation(batched)

        return activated


def input_mask(x):
    mask = (x != 0).unsqueeze(1)
    return mask


class Encoder(nn.Module):
    def __init__(
            self,
            input_size,
            embed_dims,
            max_len,
            heads,
            forward_expansion,
            num_layers,
            dropout,
            hidden_dims= 384
    ):
        super(Encoder, self).__init__()
        self.prenet = EncoderPreNet(dropout)
        # self.vid_embed = nn.Linear(input_size, embed_dims)
        self.alpha = nn.Parameter(torch.ones(1))
        self.positional_embed = PositionalEncoding(hidden_dims)
        # self.positional_embed = nn.Parameter(torch.zeros(1, max_len, 384))
        self.dropout = nn.Dropout(dropout)
        self.attention_layers = nn.Sequential(
            *[
                TransformerBlock(
                    hidden_dims,
                    heads,
                    dropout,
                    forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        x = x.permute(0, 4, 1, 2, 3)
        # print("input to the prenet", x.shape)
        x = self.prenet(x)
        # print("prenet output", x.shape)
        # mask = input_mask(x)
        seq_len = x.shape[1]
        # x = self.vid_embed(x)

        x = x.permute(1, 0, 2)
    
        positional_embed = self.positional_embed(x)
        x = x.permute(1, 0, 2)
        x = positional_embed * self.alpha + x
        x = self.dropout(x)
        attn_enc_list = list()
        for layer in self.attention_layers:
            x, attn_enc = layer(x, x, x, mask)
            attn_enc_list.append(attn_enc)
        return x, attn_enc_list


class DecoderPreNet(nn.Module):
    def __init__(self, mel_dims, hidden_dims, dropout):
        super(DecoderPreNet, self).__init__()
        self.fc_out = nn.Sequential(
            nn.Linear(mel_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.fc_out(x)


class PostNet(nn.Module):
    def __init__(self, mel_dims, hidden_dims, dropout):
        # causal padding -> padding = (kernel_size - 1) x dilation
        # kernel_size = 5 -> padding = 4
        # Exclude the last padding_size output as we want only left padded output
        super(PostNet, self).__init__()
        self.conv1 = nn.Conv1d(mel_dims, hidden_dims, kernel_size=5, padding=4)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.conv_list = nn.Sequential(
            *[
                nn.Conv1d(hidden_dims, hidden_dims, kernel_size=5, padding=4)
                for _ in range(3)
            ]
        )

        self.batch_norm_list = nn.Sequential(
            *[
                nn.BatchNorm1d(hidden_dims)
                for _ in range(3)
            ]
        )

        self.dropout_list = nn.Sequential(
            *[
                nn.Dropout(dropout)
                for _ in range(3)
            ]
        )

        self.conv5 = nn.Conv1d(hidden_dims, mel_dims, kernel_size=5, padding=4)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout1(torch.tanh(self.batch_norm1(self.conv1(x)[:, :, :-4])))
        for dropout, batchnorm, conv in zip(self.dropout_list, self.batch_norm_list, self.conv_list):
            x = dropout(torch.tanh(batchnorm(conv(x)[:, :, :-4])))
        out = self.conv5(x)[:, :, :-4]
        out = out.transpose(1, 2)
        return out


class DecoderBlock(nn.Module):
    def __init__(
            self,
            embed_dims,
            heads,
            forward_expansion,
            dropout
    ):
        super(DecoderBlock, self).__init__()
        self.causal_masked_attention = SelfAttention(embed_dims, heads)
        self.attention_layer = TransformerBlock(
            embed_dims,
            heads,
            dropout,
            forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dims)

    def forward(self, query, key, value, src_mask, causal_mask):
        causal_masked_attention, attn_dec = self.causal_masked_attention(query, query, query, causal_mask)
        query = self.dropout(self.layer_norm(causal_masked_attention + query))
        out, attn_probs = self.attention_layer(query, key, value, src_mask)
        return out, attn_dec, attn_probs


class Decoder(nn.Module):
    def __init__(
            self,
            mel_dims,
            hidden_dims,
            heads,
            max_len,
            num_layers,
            forward_expansion,
            dropout
    ):
        super(Decoder, self).__init__()
        # self.positional_embed = nn.Parameter(torch.zeros(1, max_len, hidden_dims))
        self.alpha = nn.Parameter(torch.ones(1))
        self.positional_embed = PositionalEncoding(hidden_dims)
        self.prenet = DecoderPreNet(mel_dims, hidden_dims, dropout)
        self.attention_layers = nn.Sequential(
            *[
                DecoderBlock(
                    hidden_dims,
                    heads,
                    forward_expansion,
                    dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.mel_linear = nn.Linear(hidden_dims, mel_dims)
        self.stop_linear = nn.Linear(hidden_dims, 1)
        self.postnet = PostNet(mel_dims, hidden_dims, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mel, encoder_output, src_mask, casual_mask):
        # print("mel in decoder", mel.shape)
        seq_len = mel.shape[1]
        prenet_out = self.prenet(mel)
        # print("prenet mel", prenet_out.shape)
        prenet_out = prenet_out.permute(1, 0, 2)
        positional_embed = self.positional_embed(prenet_out)
        prenet_out = prenet_out.permute(1, 0, 2)
        x = positional_embed * self.alpha + prenet_out

        x = self.dropout(x)

        attn_probs_list = list()
        attn_dec_list = list()
        for layer in self.attention_layers:
            x, attn_dec, attn_probs = layer(x, encoder_output, encoder_output, src_mask, casual_mask)
            attn_probs = attn_probs.contiguous().view(-1, attn_probs.shape[0] * attn_probs.shape[1], attn_probs.shape[2],attn_probs.shape[3])[0]
            attn_dec = attn_dec.contiguous().view(-1, attn_dec.shape[0] * attn_dec.shape[1], attn_dec.shape[2], attn_dec.shape[3])[0]
            attn_probs_list.append(attn_probs)
            attn_dec_list.append(attn_dec)

        stop_linear = self.stop_linear(x)

        mel_linear = self.mel_linear(x)
        # print("mel linear", mel_linear.is_cuda)
        postnet = self.postnet(mel_linear)

        out = postnet + mel_linear

        return out, mel_linear, stop_linear, attn_dec_list, attn_probs_list

class PositionalEncoding(nn.Module):
    """
    Adds positional embedding to the input for conditioning on time.
    From the paper "Attention is all you need"
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: tensor of shape (seq_len, batch_size, embedding_size)
        Returns:
        x: tensor of shape (batch_size, seq_len, embedding_size)
        """
        x = x + self.pe[:x.size(0), :]
        x = x.permute(1, 0, 2)
        return x
        
class LSTransformer(nn.Module):
    def __init__(
            self,
            input_size = 384,
            embed_dims = hparams.emb_size,
            hidden_dims = hparams.num_hidden,
            heads = hparams.nhead,
            forward_expansion = 4,
            num_layers = 6,
            dropout = 0.1,
            mel_dims = 80,
            max_len = 240 * hparams.batch_size,
            pad_idx = 0
    ):
        super(LSTransformer, self).__init__()
        self.encoder = Encoder(
            input_size,
            embed_dims,
            max_len,
            heads,
            forward_expansion,
            num_layers,
            dropout,
            hidden_dims = 384
        )

        self.decoder = Decoder(
            mel_dims,
            hidden_dims,
            heads,
            max_len,
            num_layers,
            forward_expansion,
            dropout
        )

        self.pad_idx = 0

    def target_mask(self, mel, mel_mask):
        seq_len = mel.shape[1]
        pad_mask = (mel_mask != 0).unsqueeze(1).unsqueeze(3)
        causal_mask = torch.tril(torch.ones((1, seq_len, seq_len))).unsqueeze(1)
        return pad_mask, causal_mask

    def transform_mel(self, mel):

        for transform in self.transforms:
            if np.random.randint(0, 11) == 10:
                mel = transform(mel).squeeze(0)
        return mel


    def forward(self, vid, mel, mel_mask=None):

        # print("mel input before mask", mel.shape)
        if self.training==True:
            target_pad_mask, causal_mask = self.target_mask(mel, mel_mask)
            target_pad_mask, causal_mask = target_pad_mask.cuda(), causal_mask.cuda()
        else:
            target_pad_mask = None
            seq_len = mel.shape[1]
            causal_mask = torch.tril(torch.ones((1, seq_len, seq_len))).unsqueeze(1).cuda()
            # causal_mask = causal_mask.gt(0)
        # print("Transformer Encoder...")
        encoder_out, attns_enc = self.encoder(vid)
        # print("Transformer Decoder inputs...")
        # print("mel", mel.shape)
        # print("encoder out", encoder_out.shape)
        # print("target_pad mask", target_pad_mask.shape)
        # print("causal mask", causal_mask.shape)
        mel_postout, mel_linear, stop_linear, attns_dec, attns_probs = self.decoder(mel, encoder_out, target_pad_mask,
                                                                                    causal_mask)
        return mel_postout, mel_linear, stop_linear, attns_dec, attns_probs, attns_enc


if __name__ == "__main__":
    a = torch.randint(0, 30, (4, 60))
    mel = torch.randn(4, 128, 80)
    mask = torch.ones((4, 128))
    model = LSTransformer(
        input_size=30,
        embed_dims=512,
        hidden_dims=512,
        heads=4,
        forward_expansion=4,
        num_layers=6,
        dropout=0.1,
        mel_dims=80,
        max_len=512,
        pad_idx=0
    )
    x, y, z = model(a, mel, mask)
    # print(x.shape, y.shape, z.shape)
