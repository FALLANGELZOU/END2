import torch
import torch.nn as nn
import timm
from Tools.util.utils import cal_params
import torch.nn.functional as F

"""
常用模块
常见编解码器等

以及工具类
"""

class ConvTBNRelu(nn.Module):
	"""
	A sequence of TConvolution, Batch Normalization, and ReLU activation
	"""
	def __init__(self, channels_in, channels_out, stride=2, dilation=1, groups=1):
		super(ConvTBNRelu, self).__init__()

		Normalize = nn.BatchNorm2d
		Activation = nn.ReLU(True)
		if stride == 1:
			kernel_size = 3
			padding = 1
		elif stride == 2:
			kernel_size = 2
			padding = 0
		self.layers = nn.Sequential(
			nn.ConvTranspose2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups),
			Normalize(channels_out),
			Activation
		)

	def forward(self, x):
		return self.layers(x)


class ExpandNet(nn.Module):
	"""
	Network that composed by layers of ConvTBNRelu,
	上采样一次，特征图大小翻倍
	"""
	def __init__(self, in_channels, out_channels, blocks, stride=2, dilation=1, groups=1):
		super(ExpandNet, self).__init__()

		layers = [ConvTBNRelu(in_channels, out_channels, stride=stride, dilation=dilation, groups=groups)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = ConvTBNRelu(out_channels, out_channels, stride=stride, dilation=dilation, groups=groups)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
	
    
class ConvBNRelu(nn.Module):
	"""
	A sequence of Convolution, Batch Normalization, and ReLU activation
	"""
	def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, dilation=1, groups=1, padding=1):
		super(ConvBNRelu, self).__init__()
		
		Normalize = nn.BatchNorm2d
		Activation = nn.ReLU(True)
		self.layers = nn.Sequential(
			nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding=padding, dilation=dilation, groups=groups),
			Normalize(channels_out),
			Activation
		)

	def forward(self, x):
		return self.layers(x)


class CNNEncoder(nn.Module):
    """CNN特征编码器
    """
    def __init__(self, 
                 in_dim, # 输入的特征
                 patch=8, # 最终输出patch*patch个特征
                 backbone = "resnet50d", # basebone是什么
                 out_indices=[2], # 需要第几层的特征，注意可以获取多个
                 extrat = nn.Identity(),
                 only_feat = False,
                 flatten = True
        ):
        """
        Args:
            in_dim : 输入特征的通道数
            patch : 最终输出patch*patch个特征， 若为None，则不改变
            backbone : 使用的骨干模型
            out_indices : 输出哪几层的特征
            extract : 对backbone输出的最后一层的feature的操作
            only_feat : 只输出最后提取的特征，不输出中间特征
            flatten : 是否展平最后两个维度
            
        Returns:
            feat : 
                        输出形状[batch, dim, patch*patch]的特征\n 
                        dim默认为512维
            hidden_fetures: 中间特征，由out_indices决定
            
        backbone：
            resnet50d : out 512 dim
            resnet18d : out 128 dim
        """
        super(CNNEncoder, self).__init__()
        
        self.patch = patch
        self.backbone = timm.create_model(
            backbone, 
            in_chans=in_dim,
            features_only= True,
            out_indices=out_indices
            )
        self.gap = nn.AdaptiveAvgPool2d((patch, patch))
        self.extract = extrat
        self.only = only_feat
        self.flatten = flatten
        pass
    
    def forward(self, x):
        batch, channel, h, w = x.size()
        fn = self.backbone(x)
        feat = fn[-1]
        feat = self.extract(feat)
        if self.patch is not None:
            feat = self.gap(feat)
            pass
        b, c, h, w = feat.size()
        if self.flatten:
            feat = feat.view(b, c, h*w)
        
        if self.only:
            return feat
        return feat, fn
        pass

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, num_hidden_layers=1, norm='ln', act="lrelu"):
        super(MLP, self).__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else self.cal_hidden_dim(in_dim, out_dim)
        self.header = LinearBlock(in_dim, hidden_dim, norm, act)
        self.hidden_layer = nn.Sequential(*[
            LinearBlock(hidden_dim, hidden_dim, norm, act) for _ in range(num_hidden_layers)
        ])
        self.project = LinearBlock(hidden_dim, out_dim, 'none', 'none')
        pass
    
    
    def cal_hidden_dim(self, in_dim, out_dim):
        """计算隐藏层的维度
        
        若不提供隐藏层的维度，就用当前函数计算
        """
        return int((in_dim + out_dim) * (2/3))
    
    def forward(self, x):
        x = self.header(x)
        x = self.hidden_layer(x)
        x = self.project(x)
        return x
    pass

class Flatten(nn.Module):
    """用于展平tensor
    """
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)
    
    


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, activation, pad_type):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, dim,
                                    norm=norm,
                                    activation=activation,
                                    pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm='bn', activation='lrelu', pad_type='zero', kernel_size=3):
        """
        ResBlock V2
        :param in_channel: input channel
        :param out_channel: output channel
        :param norm: norm
        :param activation: activation for residual mapping
        :param pad_type: padding type
        :param kernel_size: only 3, 5, 7
        """
        super(ResBlock, self).__init__()

        padding = 1
        if kernel_size == 5:
            padding = 2
        if kernel_size == 7:
            padding = 3
        self.pre = nn.Identity() if in_channel == out_channel else nn.Conv2d(in_channel, out_channel, (1, 1),
                                                                             bias=False)

        model = []
        model += [Conv2dBlock(out_channel, out_channel, kernel_size, 1, padding,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type, pre_norm_activation=True)]
        model += [Conv2dBlock(out_channel, out_channel, kernel_size, 1, padding,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type, pre_norm_activation=True)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = self.pre(x)
        out = self.model(residual)
        out += residual
        return out

class ResBlock1D(nn.Module):
    """

    Args:
        nn (_type_): _description_
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.pre = nn.Identity() if in_channel == out_channel else nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1, 1, 0),
        )
        
        self.model = nn.Sequential(
                nn.BatchNorm1d(out_channel),
                nn.LeakyReLU(),
                nn.Conv1d(out_channel, out_channel, 3, 1, 1),
                
                nn.BatchNorm1d(out_channel),
                nn.LeakyReLU(),
                nn.Conv1d(out_channel, out_channel, 3, 1, 1),
        )
        pass
    
    def forward(self, x):
        res = self.pre(x)
        out = self.model(res)
        return out + res
        pass
    pass


class ActFirstResBlock(nn.Module):
    def __init__(self, fin, fout, fhid=None,
                 activation='lrelu', norm='none'):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.fhid = min(fin, fout) if fhid is None else fhid
        self.conv_0 = Conv2dBlock(self.fin, self.fhid, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        self.conv_1 = Conv2dBlock(self.fhid, self.fout, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        if self.learned_shortcut:
            self.conv_s = Conv2dBlock(self.fin, self.fout, 1, 1,
                                      activation='none', use_bias=False)  # 用了个1x1的线性映射，主要是调整通道

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_0(x)
        dx = self.conv_1(dx)
        out = x_s + dx
        return out


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.utils.spectral_norm(nn.Linear(in_dim, out_dim, bias=use_bias)) if norm == 'sn' \
            else nn.Linear(in_dim, out_dim, bias=use_bias)
        self.inplace = True if norm != 'sn' else False
        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'sn':
            self.norm = None
        elif norm == 'none':
            self.norm = None
        elif norm == 'ln':
            self.norm = nn.LayerNorm(norm_dim)
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=self.inplace)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=self.inplace)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False, pre_norm_activation=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        self.pre_norm_activation = pre_norm_activation
        self.inplace = True if norm != 'sn' else False
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaIN2d(norm_dim)
        elif norm == 'sn':
            # 谱归一化
            self.norm = None
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=self.inplace)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(inplace=self.inplace)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)) if norm == 'sn' \
            else nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

    def forward(self, x):
        if self.pre_norm_activation:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))

        elif self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        AdaIN
        :param num_features:
        :param eps:
        :param momentum:
        """
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class AdaIN2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True):
        super(AdaIN2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "AdaIN params are None"
        N, C, H, W = x.size()
        running_mean = self.running_mean.repeat(N)
        running_var = self.running_var.repeat(N)
        x_ = x.contiguous().view(1, N * C, H * W)
        normed = F.batch_norm(x_, running_mean, running_var,
                              self.weight, self.bias,
                              True, self.momentum, self.eps)
        return normed.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(num_features=' + str(self.num_features) + ')'


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaIN2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[:, 2 * m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaIN2d":
            num_adain_params += 2 * m.num_features
    return num_adain_params

class AdainMLP(nn.Module):
    def __init__(self, nf_in, nf_out, nf_mlp, num_blocks, norm, act):
        """
        计算adain的参数
        :param nf_in: 输入channel
        :param nf_out: 输出channel，一般用get_num_adain_params计算
        :param nf_mlp: 中间过程中的channel
        :param num_blocks: 重复几次
        :param norm: 正则
        :param act: 激活函数
        """
        super(AdainMLP, self).__init__()
        self.model = nn.ModuleList()
        nf = nf_mlp
        self.model.append(LinearBlock(nf_in, nf, norm=norm, activation=act))
        for _ in range(num_blocks - 2):
            self.model.append(LinearBlock(nf, nf, norm=norm, activation=act))
        self.model.append(LinearBlock(nf, nf_out, norm='none', activation='none'))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class TranConv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu',
                 use_bias=True, activation_first=False, pre_norm_activation=False):
        super(TranConv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        self.pre_norm_activation = pre_norm_activation
        self.inplace = True if norm != 'sn' else False
        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaIN2d(norm_dim)
        elif norm == 'sn':
            # 谱归一化
            self.norm = None
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=self.inplace)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(inplace=self.inplace)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.utils.spectral_norm(
            nn.ConvTranspose2d(in_dim, out_dim, ks, st, padding, bias=self.use_bias)) if norm == 'sn' \
            else nn.ConvTranspose2d(in_dim, out_dim, ks, st, padding, bias=self.use_bias)

    def forward(self, x):
        if self.pre_norm_activation:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
            x = self.conv(x)

        elif self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(x)
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(x)
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class UpConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, norm='in', activation='lrelu', pad_type='zero',
                 use_trans=True):
        super(UpConv2d, self).__init__()

        kernel_size = 3
        padding = 1
        if kernel_size == 5:
            padding = 2
        if kernel_size == 7:
            padding = 3

        self.up = TranConv2dBlock(
            in_channel, out_channel,
            (4, 4), 2, 1,
            norm=norm,
            activation=activation) if use_trans else nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            Conv2dBlock(in_channel, out_channel, kernel_size, 1,
                        padding=padding, norm=norm, activation=activation,
                        pad_type=pad_type)
        )
        pass

    def forward(self, x):
        x = self.up(x)
        return x
        pass


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, clamp=2, x_dim=128, y_dim=64):
        """
        水印嵌入可逆块
        :param subnet_constructor: 变换网络
        :param clamp: 超参数
        :param x_dim: cover 信息
        :param y_dim: 嵌入信息
        """
        super(InvBlock, self).__init__()
        self.clamp = clamp
        # ρ
        self.r = subnet_constructor(x_dim, y_dim)
        # η
        self.y = subnet_constructor(x_dim, y_dim)
        # φ
        self.f = subnet_constructor(y_dim, x_dim)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x1, x2, rev=False):
        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:
            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return y1, y2
    pass


class ReShape(nn.Module):
    """改变tensor形状，默认不包含batch维度
    """
    def __init__(self, *size, contain_batch=False):
        """
        Args:
            size: 改变后的形状
            contain_batch: 若为false则第一维不变，且输入的size从第二维开始
        """
        super(ReShape, self).__init__()
        self.size = size
        self.contain_batch = contain_batch
        pass
    def forward(self, x):
        if self.contain_batch:
            return x.view(*self.size).contiguous()
        batch = x.size(0)
        return x.view(batch, *self.size).contiguous()
        pass
    pass


class CNNDecoder(nn.Module):
    def __init__(self, 
                 in_dim,
                 out_dim, 
                 num_layers=4,
                 build_subnet = lambda in_dim,out_dim: nn.Sequential(
                     ResBlock(in_dim, out_dim, 'bn', 'lrelu'),
                     nn.Upsample(scale_factor=2),
                     ResBlock(out_dim, out_dim, 'bn', 'lrelu')
                 ),
                 build_tail = lambda in_dim, out_dim: nn.Sequential(
                     Conv2dBlock(in_dim, out_dim, 3, 1, 1, 'none', 'tanh')
                    #  ResBlock(in_dim, out_dim, 'none', 'tanh')
                 ),
                 only_res = True
                 ):
        """CNN解码器

        Args:
            in_dim (_type_): 输入的特征图的维度
            out_dim (_type_): 输出的特征图的维度
            num_layers (int, optional): 上采样模块有几个. 一个上采样模块中，输入特征的维度是输出特征维度的两倍
            build_subnet (_type_, optional): 上采样模块的细节，使用高阶函数构造. 传入输入特征维度和输出特征维度
            build_tail (_type_, optional): 尾部模块细节，使用高阶函数构造，传入输入特征维度和输出特征维度
            only_res (bool, optional): 是否只输出解码结果，若为False，还输出经过上采样模块后的特征图
        Returns:
            res : 最终解码的结果
            feat : 上采样后的特征图
        """
        super(CNNDecoder, self).__init__()
        self.model = []
        dim = in_dim
        for _ in range(num_layers):
            self.model.append(build_subnet(dim, dim // 2))
            dim = dim // 2
            pass
        self.model = nn.Sequential(*self.model)
        self.tail = build_tail(dim, out_dim)
        self.only = only_res
        pass
    
    
    def forward(self, x):
        """前向传播

        Args:
            x (_type_): 特征图[batch, channel, dim]
        """
        feat = self.model(x)
        if self.only:
            return self.tail(feat)
        return self.tail(feat), feat
        pass
    pass

class Permute(nn.Module):
    """重组通道顺序

    Args:
        nn (_type_): _description_
    """
    def __init__(self, *params):
        super().__init__()
        self.params = params
        pass
    
    def forward(self, x):
        return x.permute(*self.params).contiguous()
    pass

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.params = dim
        pass
    def forward(self, x: torch.Tensor):
        return x.unsqueeze(self.params)
        pass
    pass
    
class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.params = dim
        pass
    def forward(self, x: torch.Tensor):
        return x.squeeze(self.params)
        pass
    pass
if __name__ == "__main__":
    # model = ReShape(2, -1)
    # x = torch.randn(10, 10)
    # print(model(x).size())
    #model = MLP(1, 256, None, 4)
    # model = CNNEncoder(3, 8, 'resnet18d')
    # x = torch.randn(3, 3, 280, 256)
    # print(model(x)[0].size())
    # print("{:.2f}M".format(cal_params(model)))
    model = CNNDecoder(
        256, 
        3, 
        num_layers=3, 
        build_tail= lambda in_dim, out_dim: nn.Sequential(
            ResBlock(in_dim, in_dim // 2, 'bn', 'lrelu'),
            ResBlock(in_dim // 2, in_dim // 4, 'bn', 'lrelu'),
            ResBlock(in_dim // 4, out_dim, 'none', 'tanh')
        ))
    x = torch.randn(4, 256, 32, 32)
    print(model(x).size())
    print("{:.2f}M".format(cal_params(model)))
    pass
    