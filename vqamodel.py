import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
import warnings
warnings.filterwarnings('ignore')


class VQAModel(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, embedding_tokens, vision_features, out_dim, p_dropout=0.5, question_features=1024, mid_features=512, glimpses=2):
        super(VQAModel, self).__init__()


        self.img_encoder = ResNetClassifier(
            in_size=(3,100,100),
            out_classes=2048,
            channels=[32, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024],
            pool_every=2,
            activation_type='relu',
            activation_params=dict(),
            pooling_type='avg',
            pooling_params=dict(kernel_size=2),
            batchnorm=True,
            dropout=0.1,
        )

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            drop=p_dropout,
        )

        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=mid_features,
            glimpses=2,
            drop=p_dropout,
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(glimpses * vision_features + question_features, mid_features),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(mid_features, out_dim)
        )


        
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, img, ques, q_len):
        ques = self.text(ques, list(q_len.data))

        img = img / (img.norm(p=2, dim=1, keepdim=True).expand_as(img) + 1e-8)
        a = self.attention(img, ques)
        img = apply_attention(img, a)

        combined = torch.cat([img, ques], dim=1)
        answer = self.classifier(combined)
        return answer



class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1)
        self.features = lstm_features

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform(w)

    def forward(self, ques, q_len):
        embedded = self.embedding(ques)
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        _, (_, c) = self.lstm(packed)
        return c.squeeze(0)


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img, ques):
        img = self.v_conv(self.drop(img))
        ques = self.q_lin(self.drop(ques))
        ques = tile_2d_over_nd(ques, img)
        x = self.relu(img + ques)
        x = self.x_conv(self.drop(x))
        return x


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1) # [n, 1, c, s]
    attention = attention.view(n, glimpses, -1)
    attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]
    weighted = attention * input # [n, g, v, s]
    weighted_mean = weighted.sum(dim=-1) # [n, g, v]
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}

class ResidualBlock(nn.Module):
    """
    """
    def __init__(
        self,
        in_channels: int,
        channels: list,
        kernel_sizes: list,
        batchnorm=False,
        dropout=0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):

        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None
        main_layers = []
        shortcut_layers = []

        # - extract number of conv layers
        N = len(channels)

        # - first conv layer 
        main_layers.append(
            nn.Conv2d(
                in_channels,
                channels[0],
                kernel_size= kernel_sizes[0],
                padding=(int((kernel_sizes[0]-1)/2),
                int((kernel_sizes[0]-1)/2)), bias=True))
        if dropout !=0:
            main_layers.append(torch.nn.Dropout2d(p=dropout))
        if batchnorm == True:
            main_layers.append(torch.nn.BatchNorm2d(channels[0]))
        main_layers.append(ACTIVATIONS[activation_type]())

        #middle layers
        for i in range(1,N-1):
            main_layers.append(
                nn.Conv2d(
                    channels[i-1],
                    channels[i],
                    kernel_size= kernel_sizes[i],
                    padding=(int((kernel_sizes[i]-1)/2),
                    int((kernel_sizes[i]-1)/2)), bias=True))
            if dropout !=0:
                main_layers.append(torch.nn.Dropout2d(p=dropout))
            if batchnorm == True:
                main_layers.append(torch.nn.BatchNorm2d(channels[i]))
            main_layers.append(ACTIVATIONS[activation_type]())
        if N > 1:
            main_layers.append(
                nn.Conv2d(
                    channels[N-2],
                    channels[N-1],
                    kernel_size= kernel_sizes[N-1],
                    padding=(int((kernel_sizes[N-1]-1)/2),
                    int((kernel_sizes[N-1]-1)/2)), bias=True))
        if (in_channels != channels[N-1]):
            shortcut_layers.append(nn.Conv2d (in_channels, channels[N-1], kernel_size= 1, bias=False))

        self.main_path = nn.Sequential(*main_layers)
        self.shortcut_path = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        out = self.main_path(x)
        out = out + self.shortcut_path(x)
        relu = torch.nn.ReLU()
        out = relu(out)
        return out


class ResNetClassifier(nn.Module):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
        batchnorm=False,
        dropout=0.0,
        **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        super().__init__()
#             in_size, out_classes, channels, pool_every, activation_type,
#             activation_params, pooling_type, pooling_params, batchnorm, dropout, **kwargs
#         )
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.conv_params=dict(kernel_size=3, stride=1, padding=1)
        self.out_classes = out_classes
        self.in_size = in_size
        self.channels = channels
        self.pool_every = pool_every
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params
        self.feature_extractor = self._make_feature_extractor()
        self.liner = torch.nn.Linear(100, self.out_classes)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)
        layers = []
        
        # - extract number of conv layers
        N = len(self.channels)
        
        #1st layer
        temp_in_channels = in_channels
        temp_channels = []
        temp_kernel_sizes = []
        
        #middle layers
        for i in range(1,N):
            temp_channels.append(self.channels[i-1])
            temp_kernel_sizes.append(3)
            if ((i % self.pool_every)==0 and i!=0):
                layers.append(
                    ResidualBlock(
                        in_channels=temp_in_channels,
                        channels=temp_channels,
                        kernel_sizes=temp_kernel_sizes,
                        batchnorm=self.batchnorm,
                        dropout=self.dropout,
                        activation_type=self.activation_type))
                temp_in_channels = self.channels[i-1]
                temp_channels = []
                temp_kernel_sizes = []
                layers.append(POOLINGS[self.pooling_type](self.pooling_params['kernel_size']))
        temp_channels.append(self.channels[N-1])
        temp_kernel_sizes.append(3)
        layers.append(ResidualBlock(
                in_channels=temp_in_channels,
                channels=temp_channels,
                kernel_sizes=temp_kernel_sizes,
                batchnorm=self.batchnorm,
                dropout=self.dropout,
                activation_type=self.activation_type))
        if ((N % self.pool_every)==0):
            layers.append(POOLINGS[self.pooling_type](self.pooling_params['kernel_size']))
        seq = nn.Sequential(*layers)
        return seq
    
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        features = self.liner(features)
        return features
