"""基于VGG的文本识别模型（第二代）"""
import torch.nn as nn
from easyocr.model.modules import VGG_FeatureExtractor, BidirectionalLSTM


class Model(nn.Module):

    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # 高度不变（通道不变），宽度为1（h为1）

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    def forward(self, input):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # b*c*h*w -> b*w*c*h -> b*w*c*1
        visual_feature = visual_feature.squeeze(3)  # b*w*c

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction


if __name__ == "__main__":
    m = Model(3, 256, 256, 100)
    print(m)
    import torch
    x = torch.randn((16, 3, 100, 300))
    x = m(x)
    print(x.shape)
