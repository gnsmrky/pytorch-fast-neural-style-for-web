import torch

# exports the original model, no workarounds.  The clean model.
ONNX_EXPORT_TARGET_ONNXRT  = "ONNXRT"       # exports the original PyTorch FNS model

# targets ONNX.js v0.1.3
#   exports the model with compatible 'InstanceNormalization' base ops and UpSampleBy2() (for interpolate) base ops that have issue #53 workaround.
ONNX_EXPORT_TARGET_ONNXJS013  = "ONNXJS_013"

# targets ONNX.js v0.1.4 and above, which supports 'InstanceNormalization' by 'cpu' and 'wasm' backend.
#   exports the model with compatible UpSampleBy2() (for interpolate) and 'ZeroPad' base ops ('cpu' and 'wasm' does not support 'Pad' op)
#   bug: v0.1.4 'wasm' backend has "RuntimeError: memory access out of bounds" error, while 'cpu' backend runs good.
#           the bug was posted to ONNX.js as issue #102: https://github.com/Microsoft/onnxjs/issues/102
ONNX_EXPORT_TARGET_ONNXJS_CPUWASM  = "ONNXJS_CPU"

# targets the latest ONNX.js for 'webgl' backend.
#   for v0.1.4, same as v0.1.3 but with issue #53 fixed.
ONNX_EXPORT_TARGET_ONNXJS = "ONNXJS"        
                                            
# targets PlaidML
#   exports the model with compatible 'InstanceNormalization' base ops, UpSampleBy2() (for interpolate) and ZeroPad2d().
ONNX_EXPORT_TARGET_PLAIDML = "PLAIDML"      


DEFAULT_ONNX_EXPORT_TARGET = "ONNXJS"       # ONNX_EXPORT_TARGET_ONNXRT or ONNX_EXPORT_TARGET_ONNXJS


def _instance_norm (target_fw):
    ins_norm = torch.nn.InstanceNorm2d

    if target_fw == ONNX_EXPORT_TARGET_ONNXRT:
        ins_norm = torch.nn.InstanceNorm2d

    elif target_fw == ONNX_EXPORT_TARGET_ONNXJS013:
        ins_norm = InstanceNorm2d_ONNXJS013

    elif target_fw == ONNX_EXPORT_TARGET_ONNXJS_CPUWASM:
        ins_norm = torch.nn.InstanceNorm2d

    elif target_fw == ONNX_EXPORT_TARGET_ONNXJS:
        ins_norm = InstanceNorm2d

    elif target_fw == ONNX_EXPORT_TARGET_PLAIDML:
        ins_norm = InstanceNorm2d

    return ins_norm

# functional layer used in UpsampleConvLayer()
def _upsample_by_2 (target_fw, x, c, h, w):
    if target_fw == ONNX_EXPORT_TARGET_ONNXRT:
        return torch.nn.functional.interpolate(x, mode='nearest', scale_factor=2)

    elif target_fw == ONNX_EXPORT_TARGET_ONNXJS013:
        return upsample_by_2_ONNXJS013(x, c, h, w)

    elif target_fw == ONNX_EXPORT_TARGET_ONNXJS_CPUWASM:
        return upsample_by_2(x, c, h, w)

    elif target_fw == ONNX_EXPORT_TARGET_ONNXJS:
        return upsample_by_2(x, c, h, w)

    elif target_fw == ONNX_EXPORT_TARGET_PLAIDML:
        return upsample_by_2(x, c, h, w)
    
    return torch.nn.functional.interpolate(x, mode='nearest', scale_factor=self.upsample)

# functional layer used in UpsampleConvLayer()
def _padding (target_fw, padding, channels, h, w):
    if target_fw == ONNX_EXPORT_TARGET_ONNXRT:
        return torch.nn.ReflectionPad2d(padding)

    elif target_fw == ONNX_EXPORT_TARGET_ONNXJS013:
        return torch.nn.ReflectionPad2d(padding)

    elif target_fw == ONNX_EXPORT_TARGET_ONNXJS_CPUWASM:
        return ZeroPadding(padding, channels, h, w)

    elif target_fw == ONNX_EXPORT_TARGET_ONNXJS:
        return torch.nn.ReflectionPad2d(padding)

    elif target_fw == ONNX_EXPORT_TARGET_PLAIDML:
        return torch.nn.ZeroPad2d(padding)
    
    return torch.nn.ReflectionPad2d(padding)

class TransformerNet_BaseOps(torch.nn.Module):
    def __init__(self, img_in, num_channels=32, target_framework=DEFAULT_ONNX_EXPORT_TARGET):
        super(TransformerNet_BaseOps, self).__init__()

        img_h = img_in.shape[2]
        img_w = img_in.shape[3]

        # Initial convolution layers
        self.conv1 = ConvLayer(img_h, img_w, 3, num_channels, kernel_size=9, stride=1, target_fw=target_framework)
        self.in1 = _instance_norm(target_framework)(num_channels, affine=True)
        
        self.conv2 = ConvLayer(img_h, img_w, num_channels, num_channels*2, kernel_size=3, stride=2, target_fw=target_framework)
        self.in2 = _instance_norm(target_framework)(num_channels*2, affine=True)

        self.conv3 = ConvLayer(img_h//2, img_w//2, num_channels*2, num_channels*4, kernel_size=3, stride=2, target_fw=target_framework)
        self.in3 = _instance_norm(target_framework)(num_channels*4, affine=True)

        # Residual layers
        self.res1 = ResidualBlock(img_h//4, img_w//4, num_channels*4, target_fw=target_framework)
        self.res2 = ResidualBlock(img_h//4, img_w//4, num_channels*4, target_fw=target_framework)
        self.res3 = ResidualBlock(img_h//4, img_w//4, num_channels*4, target_fw=target_framework)
        self.res4 = ResidualBlock(img_h//4, img_w//4, num_channels*4, target_fw=target_framework)
        self.res5 = ResidualBlock(img_h//4, img_w//4, num_channels*4, target_fw=target_framework)
        
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(img_h//4, img_w//4, num_channels*4, num_channels*2, kernel_size=3, stride=1, upsample=2, target_fw=target_framework)
        self.in4 = _instance_norm(target_framework)(num_channels*2, affine=True)

        self.deconv2 = UpsampleConvLayer(img_h//2, img_w//2, num_channels*2, num_channels, kernel_size=3, stride=1, upsample=2, target_fw=target_framework)
        self.in5 = _instance_norm(target_framework)(num_channels, affine=True)

        self.deconv3 = ConvLayer(img_h, img_w, num_channels, 3, kernel_size=9, stride=1, target_fw=target_framework)

        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)

        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, h, w, in_channels, out_channels, kernel_size, stride, target_fw):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = _padding(target_fw, reflection_padding, in_channels, h, w)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, h, w, channels, target_fw):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(h, w, channels, channels, kernel_size=3, stride=1, target_fw=target_fw)
        self.in1 = _instance_norm(target_fw)(channels, affine=True)

        self.conv2 = ConvLayer(h, w, channels, channels, kernel_size=3, stride=1, target_fw=target_fw)
        self.in2 = _instance_norm(target_fw)(channels, affine=True)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

# ONNX.js v0.1.4 'cpu' and 'wasm' backend does not support 'Pad' op.
# manually do 'Pad' using zeros using base ops.
class ZeroPadding(torch.nn.Module):
    def __init__(self, padding, channels, x_h, x_w):
        super(ZeroPadding, self).__init__()

        self.padding = padding
        self.x_h = x_h
        self.x_w = x_w

        self.channels  = channels
        self.zeropad_w = torch.zeros([1, self.channels, x_h, padding], dtype=torch.float)
        self.zeropad_h = torch.zeros([1, self.channels, padding, x_w + (padding*2)], dtype=torch.float)

    def forward(self, x):
        n = torch.cat ([self.zeropad_w, x, self.zeropad_w], 3)
        n = torch.cat ([self.zeropad_h, n, self.zeropad_h], 2)

        return n

# regular instance norm using base ops 
class InstanceNorm2d(torch.nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False):
        super(InstanceNorm2d, self).__init__()

        self.num_features = num_features
        self.epsilon      = eps
        self.momentum     = momentum

        self.weight  = torch.nn.Parameter(torch.FloatTensor(num_features))
        self.bias    = torch.nn.Parameter(torch.FloatTensor(num_features))

    def forward(self,x):
        num_features = self.num_features

        mu   = x.view(1,num_features,1,-1).mean(3,keepdim=True)  # mean

        diff = x - mu
        var  = (diff*diff).view(1,num_features,1,-1).mean(3,keepdim=True)  # variance
        norm = (diff)/((var + self.epsilon).sqrt())  # instance norm

        weight = self.weight.view(-1,num_features,1,1)
        bias   = self.bias  .view(-1,num_features,1,1)

        out = norm * weight + bias

        return out

# base ops for instance norm with issue #53 workarounds for ONNX.js v0.1.3
class InstanceNorm2d_ONNXJS013(torch.nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False):
        super(InstanceNorm2d_ONNXJS013, self).__init__()

        self.num_features = num_features
        self.epsilon      = eps
        self.momentum     = momentum

        self.weight  = torch.nn.Parameter(torch.FloatTensor(num_features))
        self.bias    = torch.nn.Parameter(torch.FloatTensor(num_features))

    def forward(self,x):
        num_features = self.num_features

        mu   = x.view(1,num_features,1,-1).mean(3,keepdim=True)               # mean

        diff = x - mu
        diff1= diff + 1e-9   # !!!This is to avoid a bug in onnx.js. (https://github.com/Microsoft/onnxjs/issues/53)
        #var  = (x-mu).pow(2.0).view(1,num_features,1,-1).mean(3,keepdim=True)  # variance v1, pow() + mean() is buggy and produces NaN in ONNX.js.
        #var  = (dif*dif1).mean([2,3],keepdim=True)                             # variance v2, results in 'ValueError: only one element tensors can be converted to Python scalars' error when exporting .onnx
        #var  = (dif*dif1).mean(3,keepdim=True).mean(2,keepdim=True)            # variance v3, 2 mean()s are slower than 1 view() + 1 mean()
        #var  = (dif*dif1).sum(2,keepdim=True).sum(3,keepdim=True)              # variance v4, this is still slower than 1 view() + 1 mean() in ONNX.js
        var  = (diff*diff1).view(1,num_features,1,-1).mean(3,keepdim=True)      # variance v5, this combination of 1 view() + 1 mean() is so far the fastest in ONNX.js


        #norm = (x-mu)/(var + self.epsilon)**(.5)       # instance norm, `**(0.5)` results in 'invalid input detected' in ONNX.js
        #norm = (x-mu)/((var + self.epsilon).sqrt())    # instance norm
        norm = (diff)/((var + self.epsilon).sqrt())     # instance norm

        weight = self.weight.view(-1,num_features,1,1)
        bias   = self.bias  .view(-1,num_features,1,1)

        out = norm * weight + bias

        return out

# ONNX.js does not support interpolate and upsample ops.  manually do upsample x2 for tensors
#   ONNX.js v0.1.3 has the issue #53 (https://github.com/Microsoft/onnxjs/issues/53).
#      This is to avoid issue #53:
#        Add a small value (1e-9) to avoid the same input being input to 'cat' twice, or onnx.js would result in 'output [#] already has value' error
def upsample_by_2_ONNXJS013 (x, c, h, w):
    bb  = x.view(-1,1)
    #bb1 = x.view(-1,1)
    bb1 = bb + 1e-9  # avoid view() to speed up a little in ONNX.js
    cc  = torch.cat([bb,bb1],1)

    cc1 = cc.view(-1,w*2)
    #cc2 = cc.view(-1,w*2)
    cc2 = cc1 + 1e-9
    out = torch.cat([cc1,cc2],1).view(-1,c,h*2,w*2)

    return out

# ONNX.js does not support interpolate() and upsample() ops.  manually do upsample x2 for tensors
# ONNX.js v0.1.4 has fixed iissue #53
def upsample_by_2 (x, c, h, w):

    bb  = x.view(-1,1)
    cc  = torch.cat([bb,bb],1)

    cc1 = cc.view(-1,w*2)
    out = torch.cat([cc1,cc1],1).view(-1,c,h*2,w*2)

    return out

class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, src_h, src_w, in_channels, out_channels, kernel_size, stride, upsample=None, target_fw=DEFAULT_ONNX_EXPORT_TARGET):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = _padding(target_fw, reflection_padding, in_channels, src_h*2, src_w*2)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        # for upsample_by_2()
        self.in_channels = in_channels
        self.src_h       = src_h 
        self.src_w       = src_w
        self.target_fw   = target_fw
        
    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = _upsample_by_2(self.target_fw, x_in, self.in_channels, self.src_h, self.src_w)

        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
