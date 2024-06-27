import time

import torch
import torch.utils.data
import torch.nn.functional as F
from nltk.metrics.distance import edit_distance

from easyocr.trainer.crnn.utils import Averager


def validation(model, criterion, evaluation_loader, converter, opt, device):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)
        
        start_time = time.time()
        preds = model(image)
        forward_time = time.time() - start_time

        # Calculate evaluation loss for CTC decoder.
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        # permute 'preds' to use CTCloss format
        cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

        if opt.decode == 'greedy':
            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = converter.decode_greedy(preds_index.data.cpu(), preds_size.data.cpu())
        elif opt.decode == 'beamsearch':
            preds_str = converter.decode_beamsearch(preds, beamWidth=2)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if pred == gt:
                n_correct += 1

            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription." 
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''
            
            # ICDAR2019 Normalized Edit Distance 
            if len(gt) == 0 or len(pred) ==0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data) # ICDAR2019 Normalized Edit Distance

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data
