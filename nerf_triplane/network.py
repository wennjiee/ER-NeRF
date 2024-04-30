import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from encoding import get_encoder
from .renderer import NeRFRenderer
import numpy as np

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super().__init__()
        self.pos_emb = PositionalEncoding(d_model)
        self.encode_layer = EncoderLayer(d_model, d_k, d_v, n_heads)
        self.ff_n = nn.Linear(d_model, 32)

    def forward(self, enc_inputs):
        batch_size = 1
        len = enc_inputs.shape[0]
        enc_inputs = enc_inputs.unsqueeze(0).view(batch_size, len, -1)
        ## 位置编码
        # enc_inputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1)
        enc_outputs = self.encode_layer(enc_inputs)
        enc_outputs = torch.sum(enc_outputs, dim=1)
        enc_outputs = self.ff_n(enc_outputs)
        return enc_outputs


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff=32)

    def forward(self, enc_inputs):
        ## 下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model] 需要注意的是最初始的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数 6.
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # enc_inputs to same Q,K,V
        # enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads)
        self.linear = nn.Linear(self.n_heads * self.d_v, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, Q, K, V):

        ## 首先映射分头，然后计算atten_scores，然后计算atten_value;
        ## 输入进来的数据形状： Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        ## 先映射，后分头；注意的是q和k分头之后维度是一致，这里都是dk
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(self.d_k) # [batch_size x n_heads x len_q x len_k]
        attn = nn.Softmax(dim=-1)(scores) # 对应方向softmax后，此方向sum = 1
        context = torch.matmul(attn, v_s) # [batch_size x n_heads x len_q x d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
        return context
        # output = self.linear(context)
        # return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) ## 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term) ## 这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，补长为2，其实代表的就是奇数位置

        ## pe形状是：[max_len*d_model] ---- [max_len*1*d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  ## 定缓冲区，简单理解为这个参数不更新

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :] # self.pe[:5, :]
        return self.dropout(x)


class ConditionalLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.eps = eps
        self.gamma_dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.beta_dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

        nn.init.zeros_(self.gamma_dense.weight)
        nn.init.zeros_(self.beta_dense.weight)

    # 利用CLN来将外部条件融入到预训练模型中，其直接应用是条件文本生成
    # 在Bert等Transformer模型中，主要的Normalization方法是Layer Normalization，
    # 自然想到将对应的 β 和 γ 变成输入条件的函数，来控制 Transformer 模型的生成行为，

    def forward(self, x, condition):
        '''
        self.ConditionIntegrator(x=text_emb, condition=query_emb) # hi c
        :param x: [b, t, e]
        :param condition: [b, e]
        :return:
        '''
        mean = x.mean(-1, keepdim=True)  # b*t*1
        std = x.std(-1, keepdim=True)  # b*t*1

        # gain nn.Linear(in_features=768,out=768,bias=false) + nn.Parameter(torch.ones(hidden_size))
        gamma = self.gamma_dense(condition) + self.gamma  
        
        # bias nn.Linear(in_features=768,out=768,bias=false) + nn.Parameter(torch.zeros(hidden_size))    
        beta = self.beta_dense(condition) + self.beta  
            
        x = gamma * (x - mean) / (std + self.eps) + beta
        return x

 
# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        y = self.attentionConvNet(y) 
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1)
        return torch.sum(y * x, dim=1) # [1, dim_aud]


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size/2)
        x = x[:, :, 8-half_w:8+half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, p=0.1, training=self.training)
                
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 audio_dim = 32,
                 # torso net (hard coded for now)
                 ):
        super().__init__(opt)

        # audio embedding
        self.emb = self.opt.emb

        if 'esperanto' in self.opt.asr_model:
            self.audio_in_dim = 44
        elif 'deepspeech' in self.opt.asr_model:
            self.audio_in_dim = 29
        elif 'hubert' in self.opt.asr_model:
            self.audio_in_dim = 1024
        else:
            self.audio_in_dim = 32
            
        if self.emb:
            self.embedding = nn.Embedding(self.audio_in_dim, self.audio_in_dim)

        # audio network
        self.audio_dim = audio_dim
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)

        self.att = self.opt.att
        if self.att > 0:
            self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=14, desired_resolution=512 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=14, desired_resolution=512 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=14, desired_resolution=512 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz
        # self.transformer_encoder = TransformerEncoder(d_model=self.in_dim, d_k=64, d_v=64, n_heads=8)
        # self.conditional_layer = ConditionalLayerNorm(hidden_size=32)
        ## sigma network
        self.num_layers = 3
        self.hidden_dim = 64
        self.geo_feat_dim = 64
        self.eye_att_net = MLP(self.in_dim, 1, 16, 2)
        self.eye_dim = 1 if self.exp_eye else 0
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim, 1 + self.geo_feat_dim, self.hidden_dim, self.num_layers)
        ## color network
        self.num_layers_color = 2
        self.hidden_dim_color = 64
        self.encoder_dir, self.in_dim_dir = get_encoder('spherical_harmonics')
        self.color_net = MLP(self.in_dim_dir + self.geo_feat_dim + self.individual_dim, 3, self.hidden_dim_color, self.num_layers_color)

        self.unc_net = MLP(self.in_dim, 1, 32, 2)

        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 64, 2)

        self.testing = False

        if self.torso:
            # torso deform network
            self.register_parameter('anchor_points', 
                                    nn.Parameter(torch.tensor([[0.01, 0.01, 0.1, 1], [-0.1, -0.1, 0.1, 1], [0.1, -0.1, 0.1, 1]]))) # 向网络module添加parameter
            self.torso_deform_encoder, self.torso_deform_in_dim = get_encoder('frequency', input_dim=2, multires=8)
            # self.torso_deform_encoder, self.torso_deform_in_dim = get_encoder('tiledgrid', input_dim=2, num_levels=16, level_dim=1, base_resolution=16, log2_hashmap_size=16, desired_resolution=512)
            self.anchor_encoder, self.anchor_in_dim = get_encoder('frequency', input_dim=6, multires=3)
            self.torso_deform_net = MLP(self.torso_deform_in_dim + self.anchor_in_dim + self.individual_dim_torso, 2, 32, 3)

            # torso color network
            self.torso_encoder, self.torso_in_dim = get_encoder('tiledgrid', input_dim=2, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=16, desired_resolution=2048)
            self.torso_net = MLP(self.torso_in_dim + self.torso_deform_in_dim + self.anchor_in_dim + self.individual_dim_torso, 4, 32, 3)

    # torso_alpha_mask, torso_color_mask, deform = self.forward_torso(x = bg_coords[mask], poses = poses, c = ind_code_torso)
    def forward_torso(self, x, poses, c=None):
        # x: [N, 2] in [-1, 1]
        # head poses: [1, 4, 4]
        # c: [1, ind_dim], individual code

        # test: shrink x
        x = x * self.opt.torso_shrink

        # deformation-based
        wrapped_anchor = self.anchor_points[None, ...] @ poses.permute(0, 2, 1).inverse() # self.anchor_points=3*4 registered Xkeys , wrapped_anchor=1*3*4
        wrapped_anchor = (wrapped_anchor[:, :, :2] / wrapped_anchor[:, :, 3, None] / wrapped_anchor[:, :, 2, None]).view(1, -1) # wrapped_anchor=1*6
        # print(wrapped_anchor)
        # enc_pose = self.pose_encoder(poses)
        enc_anchor = self.anchor_encoder(wrapped_anchor) # anchor_encoder = get_encoder('frequency', input_dim=6, multires=3), enc_anchor=1*42
        enc_x = self.torso_deform_encoder(x) # torso_deform_encoder = get_encoder('frequency', input_dim=2, multires=8), enc_x=65534*34

        if c is not None:
            h = torch.cat([enc_x, enc_anchor.repeat(x.shape[0], 1), c.repeat(x.shape[0], 1)], dim=-1) # sometime h=65536*84 or = 16384*84
        else:
            h = torch.cat([enc_x, enc_anchor.repeat(x.shape[0], 1)], dim=-1)

        dx = self.torso_deform_net(h) # MLP dx = 65536*2 2D deform?
        
        x = (x + dx).clamp(-1, 1)

        x = self.torso_encoder(x, bound=1) # get_encoder('tiledgrid', input_dim=2, num_levels=16,...

        # h = torch.cat([x, h, enc_a.repeat(x.shape[0], 1)], dim=-1)
        h = torch.cat([x, h], dim=-1)

        h = self.torso_net(h) # MLP in 116, out 4, contain alpha and color info.

        alpha = torch.sigmoid(h[..., :1])*(1 + 2*0.001) - 0.001 # 透明度 h*1.002 - 0.001
        color = torch.sigmoid(h[..., 1:])*(1 + 2*0.001) - 0.001 # rgb    h*1.002 - 0.001

        return alpha, color, dx


    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz) # 340608 * 2
        feat_xy = self.encoder_xy(xy, bound=bound) # 340608 * 12
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1) # it's can be found in equation (5) of paper
    
    # input: 8*1024*2, output: 1*32
    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        if self.emb:
            a = self.embedding(a).transpose(-1, -2).contiguous() # [1/8, 29, 16]

        enc_a = self.audio_net(a) # [1/8, 64]

        if self.att > 0:
            enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
            
        return enc_a

    
    def predict_uncertainty(self, unc_inp):
        if self.testing or not self.opt.unc_loss:
            unc = torch.zeros_like(unc_inp)
        else:
            unc = self.unc_net(unc_inp.detach())

        return unc


    def forward(self, x, d, enc_a, c, e=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # enc_a: [1, aud_dim]
        # c: [1, ind_dim], individual code
        # e: [1, 1], eye feature
        enc_x = self.encode_x(x, bound=self.bound)

        sigma_result = self.density(x, enc_a, e, enc_x)
        sigma = sigma_result['sigma']
        geo_feat = sigma_result['geo_feat']
        aud_ch_att = sigma_result['ambient_aud']
        eye_att = sigma_result['ambient_eye']

        # color
        enc_d = self.encoder_dir(d)

        if c is not None:
            h = torch.cat([enc_d, geo_feat, c.repeat(x.shape[0], 1)], dim=-1)
        else:
            h = torch.cat([enc_d, geo_feat], dim=-1)
                
        h_color = self.color_net(h)
        color = torch.sigmoid(h_color)*(1 + 2*0.001) - 0.001
        
        uncertainty = self.predict_uncertainty(enc_x)
        uncertainty = torch.log(1 + torch.exp(uncertainty))

        return sigma, color, aud_ch_att, eye_att, uncertainty[..., None]


    def density(self, x, enc_a, e=None, enc_x=None):
        # x: [N, 3], in [-bound, bound]
        if enc_x is None:
            enc_x = self.encode_x(x, bound=self.bound)

        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        # aud_ch_att = self.transformer_encoder(enc_x)
        aud_ch_att = self.aud_ch_att_net(enc_x)
        # enc_w = self.conditional_layer(aud_ch_att, enc_a)
        enc_w = enc_a * aud_ch_att

        if e is not None:
            # e = self.encoder_eye(e)
            eye_att = torch.sigmoid(self.eye_att_net(enc_x))
            e = e * eye_att
            # e = e.repeat(enc_x.shape[0], 1)
            h = torch.cat([enc_x, enc_w, e], dim=-1)
        else:
            h = torch.cat([enc_x, enc_w], dim=-1)

        h = self.sigma_net(h)

        sigma = torch.exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
            'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye' : eye_att,
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        # ONLY train torso
        if self.torso:
            params = [
                {'params': self.torso_encoder.parameters(), 'lr': lr},
                {'params': self.torso_deform_encoder.parameters(), 'lr': lr, 'weight_decay': wd},
                {'params': self.torso_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
                {'params': self.torso_deform_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
                {'params': self.anchor_points, 'lr': lr_net, 'weight_decay': wd}
            ]

            if self.individual_dim_torso > 0:
                params.append({'params': self.individual_codes_torso, 'lr': lr_net, 'weight_decay': wd})

            return params

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 

            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            # {'params': self.encoder_xyz.parameters(), 'lr': lr},

            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.color_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
        ]
        if self.att > 0:
            params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.emb:
            params.append({'params': self.embedding.parameters(), 'lr': lr})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        if self.train_camera:
            params.append({'params': self.camera_dT, 'lr': 1e-5, 'weight_decay': 0})
            params.append({'params': self.camera_dR, 'lr': 1e-5, 'weight_decay': 0})

        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.unc_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params