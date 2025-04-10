import torch
import torch.nn as nn

from Tools.util.TrainUtil import Args

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out, stride=1):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)
    
class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, img_size=128, msg_len=30):
        super(Encoder, self).__init__()
        config = Args()
        config.H = img_size
        config.W = img_size
        config.encoder_channels = 64
        config.encoder_blocks = 4
        config.message_length = msg_len
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                             self.conv_channels)

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def forward(self, image, message):

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        encoded_image = self.conv_layers(image)
        # concatenate expanded message and image
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w
    



class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, msg_len=30):

        super(Decoder, self).__init__()
        config = Args()
        config.decoder_channels = 64
        config.decoder_blocks = 7
        config.message_length = msg_len
        self.channels = config.decoder_channels

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(config.decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(ConvBNRelu(self.channels, config.message_length))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(config.message_length, config.message_length)

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return x
    


class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        config = Args()
        config.discriminator_channels = 64
        config.discriminator_blocks = 3
        layers = [ConvBNRelu(3, config.discriminator_channels)]
        for _ in range(config.discriminator_blocks-1):
            layers.append(ConvBNRelu(config.discriminator_channels, config.discriminator_channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(config.discriminator_channels, 1)

    def forward(self, image):
        X = self.before_linear(image)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X)
        # X = torch.sigmoid(X)
        return X


class HiDDeN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args.img_size, args.msg_len)
        self.decoder = Decoder(args.msg_len)
        pass
    pass
if __name__ == "__main__":
    model = Encoder()
    x = torch.randn(10, 3, 128, 128)
    y = torch.randn(10, 30)
    x = model(x, y)
    print(x.size())
    model = Decoder()
    y = model(x)
    print(y.size())
    pass