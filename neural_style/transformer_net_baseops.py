import torch

ONNX_EXPORT_TARGET_ONNXRT  = "ONNXRUNTIME"  # default, exports the original PyTorch FNS model
ONNX_EXPORT_TARGET_ONNXJS  = "ONNXJS"       #          exports the model with compatible InstanceNorm() and UpSampleBy2()
ONNX_EXPORT_TARGET_PLAIDML = "PLAIDML"      #          exports the model with compatible InstanceNorm(), UpSampleBy2() and padding

ONNX_EXPORT_TARGET = ONNX_EXPORT_TARGET_ONNXRT  # ONNX_EXPORT_TARGET_ONNXRT or ONNX_EXPORT_TARGET_ONNXJS


#NUM_CHANNELS = 16 # default is 32

def _instance_norm (target_fw):
    ins_norm = torch.nn.InstanceNorm2d

    if target_fw is ONNX_EXPORT_TARGET_ONNXRT:
        ins_norm = torch.nn.InstanceNorm2d

    elif target_fw is ONNX_EXPORT_TARGET_ONNXJS:
        ins_norm = InstanceNorm2d_ONNXJS

    elif target_fw is ONNX_EXPORT_TARGET_PLAIDML:
        ins_norm = InstanceNorm2d

    return ins_norm

# functional layer used in UpsampleConvLayer()
def _padding (target_fw, padding):
    if target_fw is ONNX_EXPORT_TARGET_ONNXRT:
        return torch.nn.ReflectionPad2d(padding)

    elif target_fw is ONNX_EXPORT_TARGET_ONNXJS:
        return torch.nn.ReflectionPad2d(padding)

    elif target_fw is ONNX_EXPORT_TARGET_PLAIDML:
        return torch.nn.ZeroPad2d(padding)
    
    return torch.nn.ReflectionPad2d(padding)

# functional layer used in UpsampleConvLayer()
def _upsample_by_2 (target_fw, x, c, h, w):
    if target_fw is ONNX_EXPORT_TARGET_ONNXRT:
        return torch.nn.functional.interpolate(x, mode='nearest', scale_factor=2)

    elif target_fw is ONNX_EXPORT_TARGET_ONNXJS:
        return upsample_by_2(x, c, h, w)

    elif target_fw is ONNX_EXPORT_TARGET_PLAIDML:
        return upsample_by_2(x, c, h, w)
    
    return torch.nn.functional.interpolate(x, mode='nearest', scale_factor=self.upsample)

class TransformerNet_BaseOps(torch.nn.Module):
    def __init__(self, img_in, num_channels=32):
        super(TransformerNet_BaseOps, self).__init__()

        # Initial convolution layers
        self.conv1 = ConvLayer(3, num_channels, kernel_size=9, stride=1)
        self.in1 = _instance_norm(ONNX_EXPORT_TARGET)(num_channels, affine=True)
        
        self.conv2 = ConvLayer(num_channels, num_channels*2, kernel_size=3, stride=2)
        self.in2 = _instance_norm(ONNX_EXPORT_TARGET)(num_channels*2, affine=True)

        self.conv3 = ConvLayer(num_channels*2, num_channels*4, kernel_size=3, stride=2)
        self.in3 = _instance_norm(ONNX_EXPORT_TARGET)(num_channels*4, affine=True)

        # Residual layers
        self.res1 = ResidualBlock(num_channels*4)
        self.res2 = ResidualBlock(num_channels*4)
        self.res3 = ResidualBlock(num_channels*4)
        self.res4 = ResidualBlock(num_channels*4)
        self.res5 = ResidualBlock(num_channels*4)
        
        # Upsampling Layers
        img_h = img_in.shape[2]
        img_w = img_in.shape[3]
        self.deconv1 = UpsampleConvLayer(img_h//4, img_w//4, num_channels*4, num_channels*2, kernel_size=3, stride=1, upsample=2)
        self.in4 = _instance_norm(ONNX_EXPORT_TARGET)(num_channels*2, affine=True)

        self.deconv2 = UpsampleConvLayer(img_h//2, img_w//2, num_channels*2, num_channels, kernel_size=3, stride=1, upsample=2)
        self.in5 = _instance_norm(ONNX_EXPORT_TARGET)(num_channels, affine=True)

        self.deconv3 = ConvLayer(num_channels, 3, kernel_size=9, stride=1)

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
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = _padding(ONNX_EXPORT_TARGET, reflection_padding)
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

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = _instance_norm(ONNX_EXPORT_TARGET)(channels, affine=True)

        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = _instance_norm(ONNX_EXPORT_TARGET)(channels, affine=True)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


# regular base ops for instance norm
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

# base ops for instance norm with workarounds for ONNX.JS
class InstanceNorm2d_ONNXJS(torch.nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False):
        super(InstanceNorm2d_ONNXJS, self).__init__()

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
def upsample_by_2 (x, c, h, w):
    # !!! Use a 'view' op for each input to 'cat' op to avoid the same input being input to 'cat' twice, or onnx.js would result in 'output [#] already has value' error
    # !!!      This is to avoid a bug in onnx.js. (https://github.com/Microsoft/onnxjs/issues/53)

    bb  = x.view(-1,1)
    #bb1 = x.view(-1,1)
    bb1 = bb + 1e-100  # avoid view() to speed up a little in ONNX.js
    cc  = torch.cat([bb,bb1],1)

    cc1 = cc.view(-1,w*2)
    #cc2 = cc.view(-1,w*2)
    cc2 = cc1 + 1e-100
    out = torch.cat([cc1,cc2],1).view(-1,c,h*2,w*2)

    return out

class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, src_h, src_w, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = _padding(ONNX_EXPORT_TARGET, reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        # for upsample_by_2()
        self.in_channels = in_channels
        self.src_h       = src_h 
        self.src_w       = src_w

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = _upsample_by_2(ONNX_EXPORT_TARGET, x_in, self.in_channels, self.src_h, self.src_w)

        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
