# import torch.nn as nn
#
# from easyocr.model.modules import VGG_FeatureExtractor, BidirectionalLSTM
#
#
# class Model(nn.Module):
#
#     def __init__(self, opt):
#         super(Model, self).__init__()
#         self.opt = opt
#
#         """ FeatureExtraction """
#         self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
#         self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
#         self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
#
#         """ Sequence modeling"""
#         self.SequenceModeling = nn.Sequential(
#             BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
#             BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
#         self.SequenceModeling_output = opt.hidden_size
#
#         """ Prediction """
#         self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
#
#     def forward(self, input):
#         """ Feature extraction stage """
#         visual_feature = self.FeatureExtraction(input)
#         visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
#         visual_feature = visual_feature.squeeze(3)
#
#         """ Sequence modeling stage """
#         contextual_feature = self.SequenceModeling(visual_feature)
#
#         """ Prediction stage """
#         prediction = self.Prediction(contextual_feature.contiguous())
#
#         return prediction
#
#
# if __name__ == "__main__":
#     from easyocr.trainer.crnn.train_main import get_config
#     opt = get_config("config_files/en_filtered_config.yaml")
#     opt.num_class = 1314
#     m = Model(opt)
#     print(m)
#
#     import torch
#     img = torch.randn(1, 1, 224, 312)
#     pred = m(img)
#     print(pred.shape)
#     print(pred)
