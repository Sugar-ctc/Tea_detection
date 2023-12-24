import math
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]], label_smoothing = 0):
        super(YOLOLoss, self).__init__()
        #----------------------------------------------------------------------------------------------#
        #   The anchor corresponding to the feature layer of 13x13 is [142, 110],[192, 243],[459, 401]
        #   The anchor corresponding to the feature layer of 26x26 is [36, 75],[76, 55],[72, 146]
        #   The anchor corresponding to the feature layer of 52x52 is[12, 16],[19, 36],[40, 28]
        #----------------------------------------------------------------------------------------------#
        self.anchors        = [anchors[mask] for mask in anchors_mask]
        self.num_classes    = num_classes
        self.input_shape    = input_shape
        self.anchors_mask   = anchors_mask

        self.balance        = [0.4, 1.0, 4]
        self.stride         = [32, 16, 8]
        
        self.box_ratio      = 0.05
        self.obj_ratio      = 1 * (input_shape[0] * input_shape[1]) / (640 ** 2)
        self.cls_ratio      = 0.5 * (num_classes / 80)
        self.threshold      = 4

        self.cp, self.cn                    = smooth_BCE(eps=label_smoothing)  
        self.BCEcls, self.BCEobj, self.gr   = nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss(), 1

    def bbox_iou(self, box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        box2 = box2.T

        if x1y1x2y2:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        w1, h1  = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2  = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union   = w1 * h1 + w2 * h2 - inter + eps

        iou = inter / union

        if GIoU or DIoU or CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
                if DIoU:
                    return iou - rho2 / c2  # DIoU
                elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # convex area
                return iou - (c_area - union) / c_area  # GIoU
        else:
            return iou  # IoU
    
    def __call__(self, predictions, targets, imgs): 
        #-------------------------------------------#
        #   reshape the input prediction results
        #   bs, 255, 20, 20 => bs, 3, 20, 20, 85
        #   bs, 255, 40, 40 => bs, 3, 40, 40, 85
        #   bs, 255, 80, 80 => bs, 3, 80, 80, 85
        #-------------------------------------------#
        for i in range(len(predictions)):
            bs, _, h, w = predictions[i].size()
            predictions[i] = predictions[i].view(bs, len(self.anchors_mask[i]), -1, h, w).permute(0, 1, 3, 4, 2).contiguous()
            
        #-------------------------------------------#
        #   Get the equipment to work
        #-------------------------------------------#
        device              = targets.device
        #-------------------------------------------#
        #   Initialize the loss of three parts
        #-------------------------------------------#
        cls_loss, box_loss, obj_loss    = torch.zeros(1, device = device), torch.zeros(1, device = device), torch.zeros(1, device = device)
        
        #-------------------------------------------#
        #   Perform positive sample matching
        #-------------------------------------------#
        bs, as_, gjs, gis, targets, anchors = self.build_targets(predictions, targets, imgs)
        #-------------------------------------------#
        #   The height and width of the corresponding feature layer are calculated
        #-------------------------------------------#
        feature_map_sizes = [torch.tensor(prediction.shape, device=device)[[3, 2, 3, 2]].type_as(prediction) for prediction in predictions] 
    
        #-------------------------------------------#
        #   The loss is calculated and the three feature layers are processed separately
        #-------------------------------------------#
        for i, prediction in enumerate(predictions): 
            #-------------------------------------------#
            #   image, anchor, gridy, gridx
            #-------------------------------------------#
            b, a, gj, gi    = bs[i], as_[i], gjs[i], gis[i]
            tobj            = torch.zeros_like(prediction[..., 0], device=device)  # target obj


            n = b.shape[0]
            if n:
                prediction_pos = prediction[b, a, gj, gi]  # prediction subset corresponding to targets

                #-------------------------------------------#
                #   Calculate the regression loss of the positive sample on the match
                #-------------------------------------------#
                #-------------------------------------------#
                #   The grid obtains the x and y coordinates of the positive sample
                #-------------------------------------------#
                grid    = torch.stack([gi, gj], dim=1)
                #-------------------------------------------#
                #   Decode and get the predicted result
                #-------------------------------------------#
                xy      = prediction_pos[:, :2].sigmoid() * 2. - 0.5
                wh      = (prediction_pos[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                box     = torch.cat((xy, wh), 1)
                #-------------------------------------------#
                #   The real box is processed and mapped to the feature layer
                #-------------------------------------------#
                selected_tbox           = targets[i][:, 2:6] * feature_map_sizes[i]
                selected_tbox[:, :2]    -= grid.type_as(prediction)
                #-------------------------------------------#
                #   Calculate regression losses for both prediction and real boxes
                #-------------------------------------------#
                iou                     = self.bbox_iou(box.T, selected_tbox, x1y1x2y2=False, CIoU=True)
                box_loss                += (1.0 - iou).mean()
                #-------------------------------------------#
                #   The gt of the confidence loss is obtained from the iou of the predicted result
                #-------------------------------------------#
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                #-------------------------------------------#
                #   Calculate the classification loss of a positive sample on a match
                #-------------------------------------------#
                # selected_tcls               = targets[i][:, 1].long()
                # t                           = torch.full_like(prediction_pos[:, 5:], self.cn, device=device)  # targets
                # t[range(n), selected_tcls]  = self.cp
                # cls_loss                    += self.BCEcls(prediction_pos[:, 5:], t)  # BCE

                #-------------------------------------------#
                #   Calculate the classification loss of the positive sample on the match (polarity focal loss cross entropy loss)
                #-------------------------------------------#
                selected_tcls               = targets[i][:, 1].long()
                t                           = torch.full_like(prediction_pos[:, 5:], self.cn, device=device)  # targets
                t[range(n), selected_tcls]  = self.cp
                cls_loss                    = self.BCEcls(prediction_pos[:, 5:], t) + cls_loss  # BCE

                pred_prob = torch.sigmoid(prediction[..., 4])
                p_t = tobj * pred_prob + (1 - tobj) * (1 - pred_prob)          # z in the formula
                ourloss = cls_loss * (2*torch.sigmoid(20*(1-pred_prob-p_t)))    # fp = 2*torch.sigmoid(20*(1-pred_prob-p_t))
                cls_loss = ourloss.mean()
                # cls_loss = cls_loss + ourloss

            #-------------------------------------------#
            #   Calculate whether the target exists a confidence loss
            #   And multiplied by the ratio of each feature layer (cross entropy loss)
            # obj_loss += self.BCEobj(prediction[..., 4], tobj)* self.balance[i]   # obj loss
            #-------------------------------------------#

            #-------------------------------------------#
            #   Calculate whether the target exists a confidence loss
            #   And multiplied by the ratio of each feature layer (focal loss cross-entropy loss)
            #-------------------------------------------#
            # ourloss = self.BCEobj(prediction[..., 4], tobj)
            # pred_prob = torch.sigmoid(prediction[..., 4])
            # p_t = tobj * pred_prob + (1 - tobj) * (1 - pred_prob)
            # alpha_factor = tobj * 0.25 + (1 - tobj) * (1 - 0.25)
            # modulating_factor = (1.0 - p_t) ** 1.5
            # ourloss = ourloss * alpha_factor * modulating_factor
            # ourloss = ourloss.mean()
            # obj_loss = obj_loss + ourloss * self.balance[i]  # obj loss

            #-------------------------------------------#
            #   Calculate whether the target exists a confidence loss
            #   And multiplied by the ratio of each feature layer (polarity focal loss cross-entropy loss)
            # -------------------------------------------#
            obj_loss = self.BCEobj(prediction[..., 4], tobj) + obj_loss
            pred_prob = torch.sigmoid(prediction[..., 4])
            p_t = tobj * pred_prob + (1 - tobj) * (1 - pred_prob)          # z in the formula
            ourloss = obj_loss * (2*torch.sigmoid(20*(1-pred_prob-p_t)))    # fp = 2*torch.sigmoid(20*(1-pred_prob-p_t))
            obj_loss = ourloss.mean()
            # obj_loss = obj_loss + ourloss * self.balance[i]

        #-------------------------------------------#
        #   Multiply the loss of each part by proportion
        #   When you add them all up, multiply by batch_size
        #-------------------------------------------#
        box_loss    *= self.box_ratio
        obj_loss    *= self.obj_ratio
        cls_loss    *= self.cls_ratio
        bs          = tobj.shape[0]
        
        loss    = box_loss + obj_loss + cls_loss
        return loss
        




