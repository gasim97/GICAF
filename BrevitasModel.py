from gicaf.interface.ModelBase import PyTorchModel
import torch
import torch.nn as nn
import torch.optim as optim
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.stats import StatsOp
from logging import info

QUANT_TYPE = QuantType.INT
SCALING_MIN_VAL = 2e-16

ACT_SCALING_IMPL_TYPE = ScalingImplType.PARAMETER
ACT_SCALING_PER_CHANNEL = False
ACT_SCALING_RESTRICT_SCALING_TYPE = RestrictValueType.LOG_FP
ACT_MAX_VAL = 6.0
ACT_RETURN_QUANT_TENSOR = False
ACT_PER_CHANNEL_BROADCASTABLE_SHAPE = None
HARD_TANH_THRESHOLD = 10.0

WEIGHT_SCALING_IMPL_TYPE = ScalingImplType.STATS
WEIGHT_SCALING_PER_OUTPUT_CHANNEL = True
WEIGHT_SCALING_STATS_OP = StatsOp.MAX
WEIGHT_RESTRICT_SCALING_TYPE = RestrictValueType.LOG_FP
WEIGHT_NARROW_RANGE = True

ENABLE_BIAS_QUANT = False

HADAMARD_FIXED_SCALE = False

def _make_quant_conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      groups,
                      bias,
                      bit_width,
                      enable_bias_quant=ENABLE_BIAS_QUANT,
                      weight_quant_type=QUANT_TYPE,
                      weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
                      weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
                      weight_scaling_per_output_channel=WEIGHT_SCALING_PER_OUTPUT_CHANNEL,
                      weight_restrict_scaling_type=WEIGHT_RESTRICT_SCALING_TYPE,
                      weight_narrow_range=WEIGHT_NARROW_RANGE,
                      weight_scaling_min_val=SCALING_MIN_VAL):
    bias_quant_type = QUANT_TYPE if enable_bias_quant else QuantType.FP
    return qnn.QuantConv2d(in_channels,
                           out_channels,
                           groups=groups,
                           kernel_size=kernel_size,
                           padding=padding,
                           stride=stride,
                           bias=bias,
                           bias_quant_type=bias_quant_type,
                           compute_output_bit_width=bias and enable_bias_quant,
                           compute_output_scale=bias and enable_bias_quant,
                           weight_bit_width=bit_width,
                           weight_quant_type=weight_quant_type,
                           weight_scaling_impl_type=weight_scaling_impl_type,
                           weight_scaling_stats_op=weight_scaling_stats_op,
                           weight_scaling_per_output_channel=weight_scaling_per_output_channel,
                           weight_restrict_scaling_type=weight_restrict_scaling_type,
                           weight_narrow_range=weight_narrow_range,
                           weight_scaling_min_val=weight_scaling_min_val)


def _make_quant_linear(in_channels,
                      out_channels,
                      bias,
                      bit_width,
                      enable_bias_quant=ENABLE_BIAS_QUANT,
                      weight_quant_type=QUANT_TYPE,
                      weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
                      weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
                      weight_scaling_per_output_channel=WEIGHT_SCALING_PER_OUTPUT_CHANNEL,
                      weight_restrict_scaling_type=WEIGHT_RESTRICT_SCALING_TYPE,
                      weight_narrow_range=WEIGHT_NARROW_RANGE,
                      weight_scaling_min_val=SCALING_MIN_VAL):
    bias_quant_type = QUANT_TYPE if enable_bias_quant else QuantType.FP
    return qnn.QuantLinear(in_channels, out_channels,
                           bias=bias,
                           bias_quant_type=bias_quant_type,
                           compute_output_bit_width=bias and enable_bias_quant,
                           compute_output_scale=bias and enable_bias_quant,
                           weight_bit_width=bit_width,
                           weight_quant_type=weight_quant_type,
                           weight_scaling_impl_type=weight_scaling_impl_type,
                           weight_scaling_stats_op=weight_scaling_stats_op,
                           weight_scaling_per_output_channel=weight_scaling_per_output_channel,
                           weight_restrict_scaling_type=weight_restrict_scaling_type,
                           weight_narrow_range=weight_narrow_range,
                           weight_scaling_min_val=weight_scaling_min_val)


def _make_quant_relu(bit_width,
                    quant_type=QUANT_TYPE,
                    scaling_impl_type=ACT_SCALING_IMPL_TYPE,
                    scaling_per_channel=ACT_SCALING_PER_CHANNEL,
                    restrict_scaling_type=ACT_SCALING_RESTRICT_SCALING_TYPE,
                    scaling_min_val=SCALING_MIN_VAL,
                    max_val=ACT_MAX_VAL,
                    return_quant_tensor=ACT_RETURN_QUANT_TENSOR,
                    per_channel_broadcastable_shape=ACT_PER_CHANNEL_BROADCASTABLE_SHAPE):
    return qnn.QuantReLU(bit_width=bit_width,
                         quant_type=quant_type,
                         scaling_impl_type=scaling_impl_type,
                         scaling_per_channel=scaling_per_channel,
                         restrict_scaling_type=restrict_scaling_type,
                         scaling_min_val=scaling_min_val,
                         max_val=max_val,
                         return_quant_tensor=return_quant_tensor,
                         per_channel_broadcastable_shape=per_channel_broadcastable_shape)


def _make_quant_hard_tanh(bit_width,
                         quant_type=QUANT_TYPE,
                         scaling_impl_type=ACT_SCALING_IMPL_TYPE,
                         scaling_per_channel=ACT_SCALING_PER_CHANNEL,
                         restrict_scaling_type=ACT_SCALING_RESTRICT_SCALING_TYPE,
                         scaling_min_val=SCALING_MIN_VAL,
                         threshold=HARD_TANH_THRESHOLD,
                         return_quant_tensor=ACT_RETURN_QUANT_TENSOR,
                         per_channel_broadcastable_shape=ACT_PER_CHANNEL_BROADCASTABLE_SHAPE):
    return qnn.QuantHardTanh(bit_width=bit_width,
                             quant_type=quant_type,
                             scaling_per_channel=scaling_per_channel,
                             scaling_impl_type=scaling_impl_type,
                             restrict_scaling_type=restrict_scaling_type,
                             scaling_min_val=scaling_min_val,
                             max_val=threshold,
                             min_val=-threshold,
                             per_channel_broadcastable_shape=per_channel_broadcastable_shape,
                             return_quant_tensor=return_quant_tensor)


def _make_quant_avg_pool(bit_width,
                        kernel_size,
                        stride,
                        signed,
                        quant_type=QUANT_TYPE):
    return qnn.QuantAvgPool2d(kernel_size=kernel_size,
                              quant_type=quant_type,
                              signed=signed,
                              stride=stride,
                              min_overall_bit_width=bit_width,
                              max_overall_bit_width=bit_width)


def _make_hadamard_classifier(in_channels,
                             out_channels,
                             fixed_scale=HADAMARD_FIXED_SCALE):
    return qnn.HadamardClassifier(in_channels=in_channels,
                                  out_channels=out_channels,
                                  fixed_scale=fixed_scale)

class _QuantVGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, bit_width=8, num_classes=1000):
        super(_QuantVGG, self).__init__()
        cfg = self.cfgs[cfg]
        self.features = self._make_layers(cfg, batch_norm, bit_width)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            _make_quant_linear(512 * 7 * 7, 4096, bias=True, bit_width=bit_width),
            _make_quant_relu(bit_width),
            nn.Dropout(),
            _make_quant_linear(4096, 4096, bias=True, bit_width=bit_width),
            _make_quant_relu(bit_width),
            nn.Dropout(),
            _make_quant_linear(4096, num_classes, bias=False, bit_width=bit_width,
                            weight_scaling_per_output_channel=False),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if (m.bias != None):
                    nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, batch_norm, bit_width):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = _make_quant_conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, groups=1,
                                        bias=not batch_norm, bit_width=bit_width)
                act = _make_quant_relu(bit_width)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), act]
                else:
                    layers += [conv2d, act]
                in_channels = v
        return nn.Sequential(*layers)

    cfgs = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

class VGG19(PyTorchModel):

    def __init__(
        self, 
        bit_width: int = 8
    ) -> None:
        self.bit_width = bit_width
        self.metadata = {'height': 224, 
                            'width': 224, 
                            'channels': 3, 
                            'bounds': (0, (bit_width**2)-1), 
                            'bgr': False, 
                            'classes': 1000, 
                            'apply_softmax': False,
                            'percision': bit_width,
                            'weight_bits': bit_width,
                            'activation_bits': bit_width}
        self.vgg19 = _QuantVGG('VGG19', bit_width=bit_width)
        info("Initialized VGG19, run 'model.train(x, y)' to train")

    def train(self, x, y):
        info("Training VGG19")
        dataset = torch.utils.data.TensorDataset(torch.tensor(x).float(), torch.tensor(y))
        data_loader = torch.utils.data.DataLoader(dataset)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.vgg19.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.vgg19(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    info('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        info('Finished training VGG19')
        super(VGG19, self).__init__(model=self.vgg19, 
                                    metadata=self.metadata)