import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange


class VitNet(nn.Module):
    def __init__(self, image_size, patch_size, out_channel, in_channel=3, D=1024,
                 num_layers=4, MLP_hidden=64, num_head=3, head_channel=64,
                 dropout=0.1):
        super(VitNet, self).__init__()
        self.h, self.w = image_size
        self.p1, self.p2 = patch_size
        assert self.h % self.p1 == 0 and self.w % self.p2 == 0

        self.N = (self.h // self.p1) ** 2
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.D = D
        self.num_layers = num_layers
        self.MLP_hidden = MLP_hidden
        self.num_head = num_head
        self.head_channel = head_channel
        self.dropout = dropout

        self.linearProjection = nn.Sequential(
            # [B,N,(PPC)]
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.p1, p2=self.p2),
            # [B,N,D]
            nn.LayerNorm(self.p1 * self.p2 * self.in_channel),
            nn.Linear(in_features=self.p1 * self.p2 * self.in_channel, out_features=self.D),
            nn.LayerNorm(self.D)
        )
        self.E_position = nn.Parameter(data=torch.randn(self.N + 1, self.D), requires_grad=True)
        self.X_zero = nn.Parameter(data=torch.randn(self.D))
        self.dropout_layer = nn.Identity() if dropout == 0 else nn.Dropout(p=self.dropout)
        self.transformer = TransformerNet(in_channel=self.D, out_channel=self.D,
                                          num_layers=self.num_layers, MLP_hidden=self.MLP_hidden,
                                          num_head=self.num_head, head_channel=self.head_channel,
                                          dropout=self.dropout)
        self.latent_layer = nn.Identity()
        self.MLP_head = nn.Linear(in_features=self.D, out_features=self.out_channel)

    def forward(self, input):
        # input:[B,C,H,W]
        # input_projection:[B,N,D]
        B, _, _, _ = input.size()
        input_projection = self.linearProjection(input)
        X_zero = repeat(self.X_zero, "d -> b () d", b=B)
        # transformer_input:[B,N+1,D]
        transformer_input = torch.cat([input_projection, X_zero], dim=1) + self.E_position
        transformer_input = self.dropout_layer(transformer_input)
        # transforer_output [B,N+1,D]
        transformer_output = self.transformer(transformer_input)
        # mlp_input [B,D]
        mlp_input = torch.mean(transformer_output, dim=1)

        mlp_input = self.latent_layer(mlp_input)
        output = self.MLP_head(mlp_input)
        return output


class TransformerNet(nn.Module):
    def __init__(self, in_channel, out_channel, num_layers=4,
                 MLP_hidden=64, num_head=3, head_channel=64, dropout=0.1):
        super(TransformerNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_layers = num_layers
        self.MLP_hidden = MLP_hidden
        self.num_head = num_head
        self.head_channel = head_channel
        self.dropout = dropout

        self.encoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_blocks.append(
                Encoder_block(in_channel=self.in_channel, out_channel=self.out_channel,
                              MLP_hidden=self.MLP_hidden, num_head=self.num_head,
                              head_channel=self.head_channel, dropout=self.dropout)
            )

    def forward(self, input):
        for encoder_block in self.encoder_blocks:
            input = encoder_block(input)
        return input


class Encoder_block(nn.Module):
    def __init__(self, in_channel, out_channel, MLP_hidden=64,
                 num_head=3, head_channel=64, dropout=0.1):
        super(Encoder_block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_head = num_head
        self.head_channel = head_channel
        self.MLP_hidden = MLP_hidden
        self.dropout = dropout
        self.attention_block = nn.Sequential(
            nn.LayerNorm(self.in_channel),
            MultiHeadAttention(in_channel=self.in_channel,
                               out_channel=self.out_channel,
                               num_head=self.num_head,
                               head_channel=self.head_channel)
        )
        self.mlp_block = nn.Sequential(
            nn.LayerNorm(self.out_channel),
            nn.Linear(in_features=self.out_channel, out_features=self.MLP_hidden),
            nn.GELU(),
            nn.Identity() if self.dropout == 0 else nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.MLP_hidden, out_features=self.out_channel),
            nn.Identity() if self.dropout == 0 else nn.Dropout(p=self.dropout)
        )

    def forward(self, input):
        # input [B N D]
        # attention_output [B N D]
        attention_output = self.attention_block(input)
        residual_connection = input + attention_output

        mlp_output = self.mlp_block(residual_connection)
        re = residual_connection + mlp_output
        return re


class MultiHeadAttention(nn.Module):
    def __init__(self, in_channel, out_channel, num_head=3, head_channel=64):
        super(MultiHeadAttention, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_head = num_head
        self.head_channel = head_channel
        self.scale = self.head_channel ** (-0.5)

        self.Q_layer = nn.Sequential(
            nn.Linear(in_features=self.in_channel, out_features=self.num_head * head_channel),
            Rearrange("b n (h d)->b h n d", h=self.num_head)
        )
        self.K_layer = nn.Sequential(
            nn.Linear(in_features=self.in_channel, out_features=self.num_head * head_channel),
            Rearrange("b n (h d)->b h n d", h=self.num_head)
        )
        self.V_layer = nn.Sequential(
            nn.Linear(in_features=self.in_channel, out_features=self.num_head * head_channel),
            Rearrange("b n (h d)->b h n d", h=self.num_head)
        )

        self.out_layer = nn.Linear(in_features=self.num_head * head_channel, out_features=self.out_channel)

    def forward(self, input):
        # input:[B,N,D]
        # Q:[B,N,h,h_c]
        Q = self.Q_layer(input)
        K = self.K_layer(input).permute(0, 1, 3, 2)
        V = self.V_layer(input)

        QK = torch.einsum("bhnd,bhdm->bhnm", Q, K) * self.scale
        A = torch.softmax(QK, dim=-1)
        SA = torch.einsum("bhnm,bhmd->bhnd", A, V)
        SA = rearrange(SA, "b h n d -> b n (h d)")

        return self.out_layer(SA)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images = torch.randn(size=(2, 3, 512, 512)).to(device)
    vit_net = VitNet(image_size=(512, 512), patch_size=(64, 64), out_channel=21, in_channel=3,
                     D=1024, num_layers=12, MLP_hidden=64,
                     num_head=6, head_channel=64, dropout=0.1).to(device)

    output = vit_net(images)
    print(output)
