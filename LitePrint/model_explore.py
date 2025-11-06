import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath, Mlp
from lxj_utils_sys import calculate_conv_output_size
import torch.nn.functional as F

class LitePrint(nn.Module):
    """ 线性注意力模型 """
    def __init__(self, num_classes, num_tab=1, patch_size=18, **kwargs):
        super().__init__()
        convBlock_kernel = 7
        size_list = [32, 64, 128, 256]
        conv_para_list = [[8, 4, 0, 1]] * len(size_list)
        embed_dim = size_list[-1]
        num_heads = 8
        dim_feedforward = embed_dim * 4
        num_mhsa_layers = 4

        if num_tab == 1:
            max_len = calculate_conv_output_size(input_size = 1800, conv_settings=conv_para_list)
        else:
            max_len = 4 * calculate_conv_output_size(input_size = 1800, conv_settings=conv_para_list)

        print(f"需要计算注意力的序列长度：{max_len}")

        # ========================== ExploreModel 参数 ========================== #
        atten_config = {'name':'Linear',
                        'approx_dim':5,
                        'use_bias':True}
        # ========================== ExploreModel 参数 ========================== #

        self.dividing = nn.Sequential(
            Rearrange('b c (n p) -> (b n) c p', n=4),
        ) if num_tab > 1 else nn.Identity()
        self.combination = nn.Sequential(
            Rearrange('(b n) c p -> b c (n p)', n=4),
        ) if num_tab > 1 else nn.Identity()


        self.profiling = LocalProfiling(
            in_channels=patch_size,
            convBlock_kernel=convBlock_kernel,
            size_list=size_list,
            conv_para_list=conv_para_list
        )
        self.pos_embed = nn.Embedding(max_len, embed_dim).weight
        self.mhsa = MHSA(embed_dim, num_heads, num_mhsa_layers, dim_feedforward, atten_config)
        self.mlp = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = x.squeeze(dim=1)
        sliding_size = np.random.randint(0, 1 + x.shape[-1] // 4)
        x = torch.roll(x, shifts=sliding_size, dims=-1)
        x = self.dividing(x)
        x = self.profiling(x)
        x = self.combination(x)
        x = x.permute(0, 2, 1)
        x = self.mhsa(x, self.pos_embed.unsqueeze(0))
        x = x.mean(dim=1)
        x = self.mlp(x)
        return x

class LitePrint_no_LINA(nn.Module):
    """ 线性注意力模型 """
    def __init__(self, num_classes, num_tab=1, patch_size=18, **kwargs):
        super().__init__()
        convBlock_kernel = 7
        size_list = [32, 64, 128, 256]
        conv_para_list = [[8, 4, 0, 1]] * len(size_list)
        embed_dim = size_list[-1]
        num_heads = 8
        dim_feedforward = embed_dim * 4
        num_mhsa_layers = 4

        if num_tab == 1:
            max_len = calculate_conv_output_size(input_size = 1800, conv_settings=conv_para_list)
        else:
            max_len = 4 * calculate_conv_output_size(input_size = 1800, conv_settings=conv_para_list)

        print(f"需要计算注意力的序列长度：{max_len}")

        # ========================== ExploreModel 参数 ========================== #
        atten_config = {'name':'Linear',
                        'approx_dim':5,
                        'use_bias':True}
        # ========================== ExploreModel 参数 ========================== #

        self.dividing = nn.Sequential(
            Rearrange('b c (n p) -> (b n) c p', n=4),
        ) if num_tab > 1 else nn.Identity()
        self.combination = nn.Sequential(
            Rearrange('(b n) c p -> b c (n p)', n=4),
        ) if num_tab > 1 else nn.Identity()


        self.profiling = LocalProfiling(
            in_channels=patch_size,
            convBlock_kernel=convBlock_kernel,
            size_list=size_list,
            conv_para_list=conv_para_list
        )
        self.pos_embed = nn.Embedding(max_len, embed_dim).weight
        self.mhsa = MHSA(embed_dim, num_heads, num_mhsa_layers, dim_feedforward, atten_config)
        self.mlp = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = x.squeeze(dim=1)
        sliding_size = np.random.randint(0, 1 + x.shape[-1] // 4)
        x = torch.roll(x, shifts=sliding_size, dims=-1)
        x = self.dividing(x)
        x = self.profiling(x)
        x = self.combination(x)
        x = x.permute(0, 2, 1)
        # x = self.mhsa(x, self.pos_embed.unsqueeze(0))
        x = x.mean(dim=1)
        x = self.mlp(x)
        return x


class LitePrint_Sen1(nn.Module):
    """ 调整各种参数运行的敏感度测试模型：L, T, w, d, num_heads, h, r_of_lina """
    def __init__(self, num_classes, num_tab=1, patch_size=18,
                # 用于敏感度测试的参数
                 max_matrix_len=1800,
                 embed_dim=256,
                 num_heads=8,
                 r_of_lina=5,
                 **kwargs,
                 ):
        super().__init__()
        convBlock_kernel = 7
        size_list = [32, 64, 128, embed_dim]
        conv_para_list = [[8, 2, 0, 1]] * len(size_list)
        dim_feedforward = embed_dim * 4
        num_mhsa_layers = 4

        if num_tab == 1:
            max_len = calculate_conv_output_size(input_size = max_matrix_len, conv_settings=conv_para_list)
        else:
            max_len = 4 * calculate_conv_output_size(input_size = max_matrix_len//4, conv_settings=conv_para_list)

        print(f"需要计算注意力的序列长度：{max_len}")

        # ========================== ExploreModel 参数 ========================== #
        atten_config = {'name':'Linear',
                        'approx_dim':r_of_lina,
                        'use_bias':True}
        # ========================== ExploreModel 参数 ========================== #

        self.dividing = nn.Sequential(
            Rearrange('b c (n p) -> (b n) c p', n=4),
        ) if num_tab > 1 else nn.Identity()
        self.combination = nn.Sequential(
            Rearrange('(b n) c p -> b c (n p)', n=4),
        ) if num_tab > 1 else nn.Identity()


        self.profiling = LocalProfiling(
            in_channels=patch_size,
            convBlock_kernel=convBlock_kernel,
            size_list=size_list,
            conv_para_list=conv_para_list
        )
        self.pos_embed = nn.Embedding(max_len, embed_dim).weight
        self.mhsa = MHSA(embed_dim, num_heads, num_mhsa_layers, dim_feedforward, atten_config)
        self.mlp = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = x.squeeze(dim=1)
        sliding_size = np.random.randint(0, 1 + x.shape[-1] // 4)
        x = torch.roll(x, shifts=sliding_size, dims=-1)
        x = self.dividing(x)
        x = self.profiling(x)
        x = self.combination(x)
        x = x.permute(0, 2, 1)
        x = self.mhsa(x, self.pos_embed.unsqueeze(0))
        x = x.mean(dim=1)
        x = self.mlp(x)
        return x

class LitePrint_Sen2(nn.Module):
    """ 调整各种参数运行的敏感度测试模型：L, T, w, d, num_heads, h, r_of_lina """
    def __init__(self, num_classes, num_tab=1, patch_size=18,
                # 用于敏感度测试的参数
                atten_type = "Linear",
                **kwargs,
                 ):
        super().__init__()
        matrix_len = 1800
        embed_dim = 256
        num_heads = 8
        r_of_lina = 5


        convBlock_kernel = 7
        size_list = [32, 64, 128, embed_dim]
        conv_para_list = [[8, 2, 0, 1]] * len(size_list)
        dim_feedforward = embed_dim * 4
        num_mhsa_layers = 4

        if num_tab == 1:
            max_len = calculate_conv_output_size(input_size = matrix_len, conv_settings=conv_para_list)
        else:
            max_len = 4 * calculate_conv_output_size(input_size = matrix_len//4, conv_settings=conv_para_list)

        print(f"需要计算注意力的序列长度：{max_len}")

        # ========================== 注意力机制 参数 ========================== #
        if atten_type.lower() == "linear":
            atten_config = {'name':'Linear',
                            'approx_dim':r_of_lina,
                            'use_bias':True}
        elif atten_type.lower() == "base":
            atten_config = {'name':'base',}
        elif atten_type.lower() == "topm":
            atten_config = {'name': 'TopM',
                            'top_m': int(0.6 * max_len),
                            'dropout': 0.1, }
        # ========================== ExploreModel 参数 ========================== #

        self.dividing = nn.Sequential(
            Rearrange('b c (n p) -> (b n) c p', n=4),
        ) if num_tab > 1 else nn.Identity()
        self.combination = nn.Sequential(
            Rearrange('(b n) c p -> b c (n p)', n=4),
        ) if num_tab > 1 else nn.Identity()


        self.profiling = LocalProfiling(
            in_channels=patch_size,
            convBlock_kernel=convBlock_kernel,
            size_list=size_list,
            conv_para_list=conv_para_list
        )
        self.pos_embed = nn.Embedding(max_len, embed_dim).weight
        self.mhsa = MHSA(embed_dim, num_heads, num_mhsa_layers, dim_feedforward, atten_config)
        self.mlp = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = x.squeeze(dim=1)
        sliding_size = np.random.randint(0, 1 + x.shape[-1] // 4)
        x = torch.roll(x, shifts=sliding_size, dims=-1)
        x = self.dividing(x)
        x = self.profiling(x)
        x = self.combination(x)
        x = x.permute(0, 2, 1)
        x = self.mhsa(x, self.pos_embed.unsqueeze(0))
        x = x.mean(dim=1)
        x = self.mlp(x)
        return x


class MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_mhsa_layers, dim_feedforward, atten_config):
        super().__init__()
        self.nets = nn.ModuleList(
            [MHSA_Block(embed_dim, num_heads, dim_feedforward, atten_config) for _ in range(num_mhsa_layers)])
    def forward(self, x, pos_embed):
        output = x + pos_embed
        for layer in self.nets:
            output = layer(output)
        return output

class MHSA_Block(nn.Module):

    def __init__(self, embed_dim, nhead, dim_feedforward, atten_config):
        super().__init__()
        drop_path_rate = 0.1
        self.attn = eval(f'Attention_{atten_config["name"]}')(dim=embed_dim, num_heads=nhead, **atten_config)

        #self.attn = TopMAttention(embed_dim, nhead, dropout, top_m)
        #self.attn = LinearAttentionOptimized(embed_dim=embed_dim, value_dim=embed_dim, approx_dim=embed_dim)
        #self.attn = MultiHeadLinearAttention(num_heads=nhead,embed_dim=embed_dim, value_dim=embed_dim, approx_dim=embed_dim)
        self.drop_path = DropPath(drop_path_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=dim_feedforward, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class Attention_TopM(nn.Module):
    def __init__(self, dim, num_heads, dropout, top_m=-1, **kwargs):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.top_m = top_m

        self.qkv = nn.Linear(dim, dim * 3)
        # self.attn_drop = nn.Sequential(
        #     nn.Softmax(dim=-1),
        #     nn.Dropout(dropout),
        # )
        # self.proj_drop = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.Dropout(dropout),
        # )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.top_m == -1:
            pass
        else:
            mask = torch.zeros(B, self.num_heads, N, N, device=q.device, requires_grad=False)
            index = torch.topk(attn, k=min(self.top_m,attn.shape[-1]), dim=-1, largest=True)[1]
            mask.scatter_(-1, index, 1.)
            attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))

        #attn = self.attn_drop(attn)
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        #x = self.proj_drop(x)
        return x


class Attention_base(nn.Module):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        # self.attn_drop = nn.Sequential(
        #     nn.Softmax(dim=-1),
        # )
        # self.proj_drop = nn.Sequential(
        #     nn.Linear(dim, dim),
        # )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x

class Attention_Linear(nn.Module):
    def __init__(self,dim: int, num_heads: int, approx_dim: int, use_bias: bool = True, **kwargs):
        """
        多头线性注意力机制

        参数:
        embed_dim: 输入特征维度
        value_dim: 值向量维度（输出维度）
        approx_dim: 近似注意力维度
        num_heads: 注意力头数
        use_bias: 是否在特征映射中使用偏置项
        """
        super().__init__()
        embed_dim = dim
        value_dim = dim
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.approx_dim = approx_dim
        self.num_heads = num_heads
        self.use_bias = use_bias

        # 维度校验
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if value_dim % num_heads != 0:
            raise ValueError(f"value_dim ({value_dim}) must be divisible by num_heads ({num_heads})")

        # 计算每个头的维度
        self.head_dim = embed_dim // num_heads
        self.value_head_dim = value_dim // num_heads

        # QKV投影层
        self.qkv_proj = nn.Linear(embed_dim, (embed_dim * 2) + value_dim)

        # 特征映射矩阵 - 可选择是否使用偏置
        self.proj = nn.Linear(self.head_dim, approx_dim, bias=use_bias)
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化投影层权重
        nn.init.xavier_uniform_(self.proj.weight)
        # 如果使用偏置，初始化偏置项
        if self.use_bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        total_heads = batch_size * self.num_heads  # 计算总头数

        # 生成Q,K,V
        qkv = self.qkv_proj(x)
        Q, K, V = torch.split(qkv, [self.embed_dim, self.embed_dim, self.value_dim], dim=-1)

        # 分割多头并调整维度
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, self.value_head_dim).permute(0, 2, 1, 3)

        # 合并批次和头维度
        Q = Q.contiguous().view(total_heads, seq_len, self.head_dim)
        K = K.contiguous().view(total_heads, seq_len, self.head_dim)
        V = V.contiguous().view(total_heads, seq_len, self.value_head_dim)

        # 特征映射 φ(x) = √(1 + (Wx + b)^2) 或 √(1 + (Wx)^2)
        projected_Q = self.proj(Q)
        projected_K = self.proj(K)
        phi_Q = torch.sqrt(1 + projected_Q ** 2)
        phi_K = torch.sqrt(1 + projected_K ** 2)

        # 计算分子项: φ(Q)·(φ(K)^T·V)
        phi_K_t = phi_K.transpose(1, 2)  # [total_heads, approx_dim, seq_len]
        KTV = torch.bmm(phi_K_t, V)  # [total_heads, approx_dim, value_head_dim]
        numerator = torch.bmm(phi_Q, KTV)  # [total_heads, seq_len, value_head_dim]

        # 计算分母项: φ(Q)·(φ(K)^T·1)
        ones = torch.ones(total_heads, seq_len, 1, device=phi_K.device)  # [total_heads, seq_len, 1]
        KTO = torch.bmm(phi_K_t, ones)  # [total_heads, approx_dim, 1]
        denominator = torch.bmm(phi_Q, KTO)  # [total_heads, seq_len, 1]

        # 归一化输出
        epsilon = 1e-6
        head_output = numerator / (denominator + epsilon)

        # 合并多头输出
        head_output = head_output.view(batch_size, self.num_heads, seq_len, self.value_head_dim)
        output = head_output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, seq_len, self.value_dim)

        return output


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(ConvBlock1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation,
                      padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation,
                      padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        self.last_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)

class LocalProfiling(nn.Module):
    """
    ARES中的本地特征提取模块
    参数:
        in_channels: 输入通道数
        size_list: 每个卷积块的输出通道数列表
        conv_para_list: 每个卷积块的参数列表 [[kernel_size, stride, padding, dilation], ...]
    """

    def __init__(self, size_list, conv_para_list, in_channels=1, convBlock_kernel=7):
        super().__init__()

        # 验证参数长度是否匹配
        assert len(size_list) == len(conv_para_list), \
            "size_list and conv_para_list must have the same length"

        layers = []
        current_channels = in_channels

        # 动态构建网络层
        for out_channels, conv_params in zip(size_list, conv_para_list):
            kernel_size, stride, _, _ = conv_params
            # 添加卷积块
            layers.append(
                ConvBlock1d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=convBlock_kernel,
                )
            )

            # 添加池化和Dropout
            layers.append(nn.MaxPool1d(kernel_size=kernel_size, stride=stride))
            layers.append(nn.Dropout(p=0.1))

            # 更新当前通道数
            current_channels = out_channels

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
# def get_model(num_classes, num_tab, patch_size, model_name = "RF", **kwargs):
#     model = eval('ExploreModel_{}'.format(model_name))(num_classes=num_classes, num_tab=num_tab, patch_size=patch_size, **kwargs)
#     return model

def get_model(num_classes, num_tab, patch_size, model_name = "RF", **kwargs):
    model = eval('{}'.format(model_name))(num_classes=num_classes, num_tab=num_tab, patch_size=patch_size, **kwargs)
    return model

if __name__ == '__main__':
    from torchinfo import summary
    model_name = 'LitePrint'
    model = get_model(num_classes=95, num_tab=1, patch_size=5, model_name=model_name, max_matrix_len=1800)
    summary(model, input_size=(20, 1, 5, 1800), depth=float('1'), col_names=["input_size", "output_size", "num_params", "kernel_size"])

    model_name = 'LitePrint_Sen1'
    model = get_model(num_classes=95, num_tab=1, patch_size=5, model_name=model_name, max_matrix_len=450)
    summary(model, input_size=(20, 1, 5, 450), depth=float('1'), col_names=["input_size", "output_size", "num_params", "kernel_size"])
    model = get_model(num_classes=95, num_tab=1, patch_size=5, model_name=model_name, max_matrix_len=900)
    summary(model, input_size=(20, 1, 5, 900), depth=float('1'), col_names=["input_size", "output_size", "num_params", "kernel_size"])
    model = get_model(num_classes=95, num_tab=1, patch_size=5, model_name=model_name, max_matrix_len=1800)
    summary(model, input_size=(20, 1, 5, 1800), depth=float('1'), col_names=["input_size", "output_size", "num_params", "kernel_size"])
    model = get_model(num_classes=95, num_tab=1, patch_size=5, model_name=model_name, max_matrix_len=3600)
    summary(model, input_size=(20, 1, 5, 3600), depth=float('1'), col_names=["input_size", "output_size", "num_params", "kernel_size"])

    model = get_model(num_classes=95, num_tab=2, patch_size=5, model_name=model_name, max_matrix_len=2400)
    summary(model, input_size=(20, 1, 5, 2400), depth=float('1'), col_names=["input_size", "output_size", "num_params", "kernel_size"])
    model = get_model(num_classes=95, num_tab=2, patch_size=5, model_name=model_name, max_matrix_len=3600)
    summary(model, input_size=(20, 1, 5, 3600), depth=float('1'), col_names=["input_size", "output_size", "num_params", "kernel_size"])
    model = get_model(num_classes=95, num_tab=2, patch_size=5, model_name=model_name, max_matrix_len=7200)
    summary(model, input_size=(20, 1, 5, 7200), depth=float('1'), col_names=["input_size", "output_size", "num_params", "kernel_size"])
    model = get_model(num_classes=95, num_tab=2, patch_size=5, model_name=model_name, max_matrix_len=14400)
    summary(model, input_size=(20, 1, 5, 14400), depth=float('1'), col_names=["input_size", "output_size", "num_params", "kernel_size"])