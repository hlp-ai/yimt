import torch.nn as nn

from easyocr.model.modules import VGG_FeatureExtractor, BidirectionalLSTM


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        # self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
        #                'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        # if opt.FeatureExtraction == 'VGG':
        #     self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        # else:
        #     raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        self.SequenceModeling_output = opt.hidden_size

        # if opt.SequenceModeling == 'BiLSTM':
        #     self.SequenceModeling = nn.Sequential(
        #         BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
        #         BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        #     self.SequenceModeling_output = opt.hidden_size
        # else:
        #     print('No SequenceModeling module specified')
        #     self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)

        # if opt.Prediction == 'CTC':
        #     self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        # else:
        #     raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        # """ Transformation stage """
        # if not self.stages['Trans'] == "None":
        #     input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        # if self.stages['Seq'] == 'BiLSTM':
        #     contextual_feature = self.SequenceModeling(visual_feature)
        # else:
        #     contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        # if self.stages['Pred'] == 'CTC':
        #     prediction = self.Prediction(contextual_feature.contiguous())
        # else:
        #     prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction
