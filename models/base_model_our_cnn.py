"""
    Example for a simple model
"""

from abc import ABCMeta
# from nets.fc import FCNet
import torch
from torch import nn, Tensor
import itertools as it
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.init as init
import torch.nn.functional as F



#POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}

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

        """if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")"""

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
            main_layers.append(torch.nn.BatchNorm2d(channels[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        main_layers.append(nn.ReLU(inplace=True))

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
                main_layers.append(torch.nn.BatchNorm2d(channels[i], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            if (i%2 == 1):
                main_layers.append(nn.ReLU(inplace=True))
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
        channels,
        pool_every,
#         hidden_dims,
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

        self.batchnorm = batchnorm
        self.dropout = dropout
        self.conv_params=dict(kernel_size=3, stride=1, padding=1)
        self.in_size = in_size
        self.channels = channels
        self.pool_every = pool_every
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params
        self.feature_extractor = self._make_feature_extractor()
        self.fc = nn.Linear(512, self.output_dim, bias=True)


    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)
        layers = []
        self.output_dim = 2048
        # - extract number of conv layers
        N = len(self.channels)
        
        #1st layer
        temp_in_channels = 64
        temp_channels = []
        temp_kernel_sizes = []
        
        layers.append(nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
        layers.append(torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))
        
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
                        batchnorm= self.batchnorm,
                        dropout= self.dropout,
                        activation_type= self.activation_type))
                temp_in_channels = self.channels[i-1]
                temp_channels = []
                temp_kernel_sizes = []
                #layers.append(nn.AvgPool2d(self.pooling_params['kernel_size']))
        temp_channels.append(self.channels[N-1])
        temp_kernel_sizes.append(3)
        layers.append(ResidualBlock(
                in_channels=temp_in_channels,
                channels=temp_channels,
                kernel_sizes=temp_kernel_sizes,
                batchnorm= self.batchnorm,
                dropout= self.dropout,
                activation_type= self.activation_type))
        if ((N % self.pool_every)==0):
#             layers.append(nn.AvgPool2d(self.pooling_params['kernel_size']))
                layers.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))) ########################################################
        #layers.append(nn.Linear(1000, self.output_dim, bias=True))
        # add to go to 1x1
        #layers.append(nn.AvgPool2d(2))
        #layers.append(nn.AvgPool2d(15))
        seq = nn.Sequential(*layers)
        return seq
    	
    
    def forward(self, x):
        out = self.feature_extractor(x)
        batch_size = out.shape[0]
        out = out.view(batch_size, -1)
        #print("out.shape = ", out.shape)
        out = self.fc(out)
        out = out.view(batch_size,-1)
        #shortcut = self.shortcut_path(x)
        #print("shortcut.shape = ", shortcut.shape)
        #print("out.shape = ", out.shape)
        #out = out + shortcut
        #out = out.view(batch_size,-1,1,1)
        #relu = torch.nn.ReLU()
        #out = relu(out)
        return out 
    
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
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True, enforce_sorted=False)
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
        ques = self.q_lin(self.drop(ques))
        img = self.v_conv(self.drop(img))
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

class ImageNet(nn.Module):
    def __init__(self, output_dim):
        super(ImageNet, self).__init__()
        from torchvision import models
        self.model = models.resnet34(pretrained=False)
        self.fc = nn.Linear(1000, output_dim, bias=True)


    def forward(self, image_tensor):
        x = self.model(image_tensor)
        x = self.fc(x)
        return F.normalize(x, dim=1, p=2)

class QuestionNet(nn.Module):
    def __init__(self, word_embedding_dim, lstm_hidden_dim, word_vocab_size, output_dim, lstm_drop=0.0):
        super(QuestionNet, self).__init__()

        # LSTM unit
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=lstm_drop)
        self.lstm_input_dim = self.word_embedding.embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=self.lstm_hidden_dim, num_layers=2)
        self.fc = nn.Linear(output_dim, output_dim, bias=True)
        # self.fc2 = nn.Linear(21 * self.lstm_hidden_dim, q_output_dim, bias=True)
        # self.output = q_output_dim


    def forward(self, questions_tensor):
        t = self.word_embedding(questions_tensor)
        t = self.dropout(t)
        t = F.tanh(t)
        t = t.transpose(0, 1)
        _, (h, c) = self.lstm(t)
        t = torch.cat((h, c), 2)
        t = t.transpose(0, 1)
        t = t.reshape(t.size()[0], -1)
        t = F.tanh(t)
        return self.fc(t)

class MyModel(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """
    #def __init__(self,):# input_dim: int = 50, num_hid: int = 256, output_dim: int = 2, dropout: float = 0.2):
    def __init__(
        self, image_in_size=((3,224,224)), img_encoder_out_classes=1024, img_encoder_channels=[32, 128, 512, 1024],
        img_encoder_batchnorm=True, img_encoder_dropout=0.5, text_embedding_tokens=15193, text_embedding_features=100,
        text_lstm_features=512, text_dropout=0.5, attention_mid_features=128, attention_glimpses=2, attention_dropout=0.5,
        classifier_dropout=0.5,  classifier_mid_features=128,classifier_out_classes=2410
        ):
        super(MyModel, self).__init__()
        self.image_in_size=image_in_size
        self.img_encoder_out_classes=img_encoder_out_classes
        self.img_encoder_channels=img_encoder_channels
        self.img_encoder_batchnorm=img_encoder_batchnorm
        self.img_encoder_dropout=img_encoder_dropout
        self.text_embedding_tokens=text_embedding_tokens
        self.text_embedding_features=text_embedding_features
        self.text_lstm_features=text_lstm_features
        self.text_dropout=text_dropout
        self.attention_mid_features=attention_mid_features
        self.attention_glimpses=attention_glimpses
        self.attention_dropout=attention_dropout
        self.classifier_dropout=classifier_dropout 
        self.classifier_mid_features=classifier_mid_features
        self.classifier_out_classes=classifier_out_classes

        self.img_encoder = ResNetClassifier(
            in_size=image_in_size,
            channels=img_encoder_channels,
            pool_every=1,
            activation_type='relu',
            activation_params=dict(),
            pooling_type='avg',
            pooling_params=dict(kernel_size=3),
            batchnorm=img_encoder_batchnorm,
            dropout=img_encoder_dropout,
        )
        """self.text = TextProcessor(
            embedding_tokens=text_embedding_tokens,
            embedding_features=text_embedding_features, #300,
            lstm_features=text_lstm_features,
            drop=text_dropout,
        )"""
        
        """self.attention = Attention(
            v_features=img_encoder_out_classes,#2048,
            q_features=text_lstm_features,
            mid_features=attention_mid_features,
            glimpses=attention_glimpses,
            drop=attention_dropout,
        )"""
        
        """self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            #nn.Linear(2 * img_encoder_out_classes + text_lstm_features, classifier_mid_features), #(2*2048+1024,256)
            nn.Linear(4096, classifier_out_classes),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_out_classes, classifier_out_classes),
        )"""




        self.image_enc = ImageNet(4096)
        self.QP = QuestionNet(300, 1024, 15193, 4096, 0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4096, self.classifier_out_classes)
        self.fc2 = nn.Linear(self.classifier_out_classes, self.classifier_out_classes)


    def forward(self, x) -> Tensor:
        """temp_img = x[0]
        batch_size = temp_img.shape[0]
        img = self.image_enc(temp_img)
        #img = self.img_encoder(temp_img)
        #img = img.view(batch_size, -1)
        #print("img.shape = ",img.shape)
        
        ques = x[1]
        #q_len = x[2]
        #ques = self.text(ques, list(q_len.data))
        ques = self.QP(ques)
        combined = torch.mul(ques, img)
        combined = F.tanh(combined)
        #a = self.attention(img, ques)
        #img = apply_attention(img, a)
        #combined = torch.cat([img, ques], dim=1)
        #print("combined.shape = ", combined.shape)
        answer = self.classifier(combined)
        #answer = self.softmax(combined)
        return answer"""
   
        temp_img = x[0]
        ques  = x[1]
        img = self.img_encoder(temp_img)
        ques = self.QP(ques)
        combined = torch.mul(ques, img)
        combined  = F.tanh(combined)
        combined  = self.dropout(combined)
        combined  = self.fc1(combined)
        combined  = F.tanh(combined)
        combined  = self.dropout(combined)
        return self.fc2(combined)