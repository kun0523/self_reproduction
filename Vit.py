import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0]//patch_size[0], img_size[1]//patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        # 如果有定义 LayerNorm 就使用，如果没有 就保持不变
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        return

    def forward(self, x):
        B, C, H, W = x.shape
        assert H==self.img_size[0] and W==self.img_size[1], f"Input Image Size Error"
        # B*3*224*224  proj  B*768*14*14  faltten  B*768*196  transpose  B*196*768
        x = self.proj(x).flatten(2).transpose(1, 2)  # Embeding
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bais=False, qk_scale=None, atte_drop_ratio=0., proj_drop_ratio=0.):
        '''

        :param dim:  输入的token维度 768
        :param num_heads: 注意力头的数目
        :param qkv_bais: 生成QKV时是否添加偏置
        :param qk_scale: 用于缩放QK的系数 如果是None， 则使用 1/sqrt(embed_dim_per_head)
        :param atte_drop_ratio:  注意力分数的dropout比率
        :param proj_drop_ratio:  最终投影层的dropout比例
        '''
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bais)
        self.att_drop = nn.Dropout(atte_drop_ratio)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        # 将每个head得到输出进行concat拼接，然后通过线性变换，嵌入到原本的维度
        self.proj = nn.Linear(dim, dim)

        return

    def forward(self, x):
        B, N, C = x.shape # B batch; N num_of_patch+1 (14*14+1); C embed_dim 768
        # B, N, 3*C >> B, N, 3, num_heads, C//num_heads
        # B, N, 3, num_heads, C//num_heads >> 3, B, num_heads, N, C//num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, num_heads, N, N
        attn = attn.softmax(dim=-1)
        # (B, num_heads, N, C//num_heads)  transpose (B, N, num_heads, C//num_heads)  reshape 合并多个头的输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = (attn @ v).transpose(1, 2).flatten(2)  # 是不是一样的？
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        '''

        :param in_features: 输入维度
        :param hidden_features:  隐藏维度，通常为输入维度的4倍
        :param out_features: 输出维度 通常与输入维度一致
        :param act_layer:
        :param drop:
        '''
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        return

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def forward(self, x):
        return

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qkv_scale=None, drop_ratio=0.0, attn_drop_ratio=0.0, drop_path_ratio=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        '''

        :param dim: token 维度
        :param num_heads:  多头注意力的头数
        :param mlp_ratio:  用于计算 MLP 中 hidden_features 的维度
        :param qkv_bias:
        '''
        return

    def forward(self, x):
        return

if __name__ == "__main__":
    img = torch.ones((3, 3, 224, 224), dtype=torch.float)
    patch_embed = PatchEmbed()
    embed_vector = patch_embed(img)

    att_layer = Attention(768, 8, )
    att_layer(embed_vector)

    print()