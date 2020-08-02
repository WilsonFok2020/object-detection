import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import os, time
import numpy as np
from collections import defaultdict
import torch
import tqdm
import cv2
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from model_retinaNet import resnet18, ColorClassifier, calc_iou, place_on_cpu_gpu
import torch.optim as optim
from func import prepare_img_data_format, pair_up_bbox_color_labels, locate_top_boxes,\
 calculate_full_batch_statistics, make_clean_folder, save_checkpoint, \
 calculate_average_precision, plot_precision_recall, put_anchor_boxes_per_label_array,\
 create_sample_indices
 
from dataset import CSVDataset, Cast_to_Pytorch

CONFIDENCE = 1e-4 # have enough confidence on the detected bounding boxes to crop out patches to train a classifier
IMG_SIZE = (32,32) # cropped image size, classifier input dimension
MIN_CONFIDENCE = 0.1
IOUTHRESHOLD = 0.5 # overlap 50% or more
FREQ = 2 # check patch frequency

def locate_good_boxes(boxes_class):
    
    '''
    find good enough detected boxes and to crop patches out from images to train
    a classifier
    '''
    
    top_boxes = []
    
    '''
    maybe able to add a rule that resamples the boxes to give a more balanced dataset
    '''
    
    for _, box_scores in boxes_class.items():
        if len(box_scores) > 0:
            # score, bbox = box_score
            sorted_scores = [torch.unsqueeze(item[-1], dim=0) for item in box_scores if item[0] > MIN_CONFIDENCE]
            
            top_boxes += sorted_scores
    if len(top_boxes) == 0:
        return None
    else:
        return torch.cat(top_boxes, dim=0)

def scores_min_bar_test(scores):
    scores = scores.cpu().numpy()
    idxs = np.where(scores > CONFIDENCE)[0]
    if len(idxs) > 0:
        return idxs
    else:
        return None

def find_enough_overlap(b, shape_gt, idxs, scores, classification, transformed_anchors, labels_meaning, per_label=False):
    
    '''
    find the overlap between ground truth boxes and detected boxes
    assign the ground truth class label to the detected bounding boxes for classifier
    
    per_label is a special case used during validation / testing
    '''
    
    boxes_class = pair_up_bbox_color_labels(idxs, classification, transformed_anchors, scores, labels_meaning)
    
    if per_label:
        top_boxes = locate_top_boxes(boxes_class)
        a, a_key = put_anchor_boxes_per_label_array(top_boxes)
    else:
        a = locate_good_boxes(boxes_class)
        
    if a is not None:
        intersection, area, ratio = calc_iou(a,b, return_intersection=True)
        
        if per_label:
            # dictionary like this makes debugging far easier
            b_shape_dict_col_indices = {labels_meaning[item]:i for i, item in enumerate(shape_gt.cpu().numpy().tolist()[0])}
            
            good_overlapped_boxes = {}
            for shape, (score, bbox) in top_boxes.items():
                shape_col = b_shape_dict_col_indices[shape]
                row_index = a_key[shape]
                
                r = ratio[row_index, shape_col]
                
                if r >= IOUTHRESHOLD:
                    good_overlapped_boxes[shape] = (score, bbox)
                else:
                    print ('insufficent overlap for shape %s' %shape)
                    
            if len(good_overlapped_boxes) == 0:
                return None
            else:
                return good_overlapped_boxes
        else:
            
            # pytorch likes named tuple, therefore it is so python friendly
            max_intersection = torch.max(ratio, dim=1)
            max_areas = max_intersection.values
            max_indices = max_intersection.indices
            mask = max_areas >= IOUTHRESHOLD
            big_enough_predicted_boxes = a[mask]
            big_enough_inputs = max_indices[mask]
            
            '''
            sometimes when all detected boxes fall onto a single object,
            it may use the wrong label as it just assigns the same label for all
            '''
            
            return big_enough_predicted_boxes, big_enough_inputs
    else:
        return None
    
def prepare_patches(big_enough_predicted_boxes, big_eough_inputs, c, img_copied,
                    shape_color_dict = None):
    # make sure datatype is correct
    big_enough_predicted_boxes = big_enough_predicted_boxes.long()
    big_enough_predicted_boxes = torch.clamp(big_enough_predicted_boxes, min=0)
    
    y_start = big_enough_predicted_boxes[:,1]
    y_end = big_enough_predicted_boxes[:,3]
    x_start = big_enough_predicted_boxes[:,0]
    x_end = big_enough_predicted_boxes[:,2]
    
    L = len(big_enough_predicted_boxes)
    
    if L > 0:
        temp_img_list = []
        label_list = []
        for k in range(L):
            # may not always follow the same sequence, can contain skips
            if shape_color_dict:
                shape = big_eough_inputs[k]
                color, which = shape_color_dict[shape]
            else:
                which = big_eough_inputs[k]
                
            temp_img = img_copied[which, 
                                  :,
                                  y_start[k]:y_end[k],
                                  x_start[k]:x_end[k]]
            
            # try not to do interpolation on gpu, it can take up too much memory
            temp_img = torch.unsqueeze(temp_img, dim=0).cpu()
            patch = F.interpolate(temp_img, IMG_SIZE, mode='bilinear', align_corners=False)
            temp_img_list.append(patch)
            
            if shape_color_dict:
                label_list.append(torch.from_numpy(np.array([inverse_color_labels[color]])))
            else:
                label_list.append(torch.unsqueeze(c[big_eough_inputs[k]], dim=0))
                
        return temp_img_list, label_list
    else:
        print ('no patches')
        return None

def cat_N_items(items, dim=0):
    return [torch.cat(item, dim=dim) for item in items]

def copy_reorganize_inputs(annot, color_label_gt, img_gpu):
    
    b = annot[:,:,0:4].reshape(-1,4).cpu()
    c = color_label_gt.reshape(-1)
    shape_gt = annot[:,:,-1].cpu()
    
    # repeat 123 = 111222333
    img_copied = torch.repeat_interleave(img_gpu, annot.shape[1], dim=0)
    return b, c, shape_gt, img_copied

def check_validity_predicted_bboxes(img_gpu, annot, color_label_gt, retinanet, iter_num):
    print ('iter num %d: check_validity_predicted_bboxes' %iter_num)
    

    retinanet.calculate_focalLoss = False
    retinanet.eval()
    
    patches = None # default value 
    
    with torch.no_grad():
        img_gpu, annot, color_label_gt = cat_N_items([img_gpu, annot, color_label_gt])
        b, c, shape_gt, img_copied = copy_reorganize_inputs(annot, color_label_gt, img_gpu)
        
        scores, classification, transformed_anchors = retinanet(img_gpu.cuda())
        scores = scores.cpu()
        classification = classification.cpu()
        transformed_anchors = transformed_anchors.cpu()
        
        idxs = scores_min_bar_test(scores)
        if idxs is not None:
            found = find_enough_overlap(b, shape_gt, idxs, scores, classification, 
                                        transformed_anchors, labels_meaning, per_label=False)
            
            if found is not None:
                big_enough_predicted_boxes, big_enough_inputs = found
                patches = prepare_patches(big_enough_predicted_boxes, big_enough_inputs, c, img_copied)
    
    retinanet.train()

    
    return patches

def validate_classifier(patches, color_classifier, epoch, criterion):
    print ('validate classifier')
    color_classifier.eval()

    temp_img, label = patches
    temp_img = torch.cat(temp_img, dim=0)

    label = torch.cat(label)
    label = label.cuda()

    total = 0
    correct = 0
    
    assert temp_img.shape[0] == label.shape[0], 'not enough labels or images'
    
    with torch.no_grad():
        logit = color_classifier(temp_img.cuda())
        color_classification_loss = criterion(logit, label)
        
        _, predicted_label = torch.max(logit, 1)
        correct += (predicted_label == label).sum().item()
        total += label.size(0)
        
    return color_classification_loss.cpu().numpy().tolist(), predicted_label.data.cpu().numpy().tolist(), label.cpu().numpy()

def train_classifier(patches, color_classifier, epoch, optimizer_color, criterion):
    print ('train classifier')
    color_classifier.train()
    loss_classification = []
    
    temp_img, label = patches
    temp_img = torch.cat(temp_img, dim = 0)
    label = torch.cat(label, dim=0)


    total = 0
    correct = 0
    
    assert temp_img.shape[0] == label.shape[0], 'not enough labels or images'
    
    # very simple shuffle
    sample_indices = create_sample_indices(batch_size_color_classifier, temp_img)
    classifier_epoch = sample_indices.shape[0]
    
    for i in range(classifier_epoch):
        optimizer_color.zero_grad()
        s = sample_indices[i]
        y = label[s].cuda()
        
        logit = color_classifier(temp_img[s,...].cuda())
        color_classifier_loss = criterion(logit, y)
        
        _, predicted_label = torch.max(logit, 1)
        correct += (predicted_label == y).sum().item()
        total += y.size(0)
        
        
        loss_classification += color_classifier_loss.detach().cpu().numpy().tolist()
        color_classifier_loss = color_classifier_loss.mean()
        color_classifier_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(color_classifier.parameters(), 0.1)
        optimizer_color.step()
        
        if i ==0:
            all_y_true = label[s]
            all_pred = predicted_label.cpu().detach().numpy()
        else:
            all_y_true = np.concatenate([all_y_true, label[s]], axis=0)
            all_pred = np.concatenate([all_pred, predicted_label.cpu().detach().numpy()], axis=0)
            
    return loss_classification, all_y_true, all_pred
            
def train(dataloader_train, retinanet, color_classifier, epoch, optimizer, optimizer_color, criterion):
    
    retinanet.train()
    loss_hist = []
    loss_classification = []
    loss_regression = []
    
    img_list, labels_list = [], []
    
    
    check_img = []
    check_annot = []
    check_color_label_gt = []
    
    tbar = tqdm.tqdm(dataloader_train)
    for iter_num, data in enumerate(tbar):
        optimizer.zero_grad()
        img = data['img'].cuda()
        annot, color_label_gt = data['annot'][:,:,0:-1], data['annot'][:,:,-1]
        color_label_gt = color_label_gt.long()
        
        annot = annot.cuda()
        color_label_gt = color_label_gt.cuda()
        
        classification_loss, regression_loss = retinanet([img, annot])
                
        loss_classification += classification_loss.detach().cpu().numpy().tolist()
        loss_regression += regression_loss.detach().cpu().numpy().tolist()
        
        total_loss = classification_loss + regression_loss
        loss_hist += total_loss.detach().cpu().numpy().tolist()
        
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        loss = classification_loss + regression_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
        optimizer.step()
        
        tbar.set_description('epoch: {}, step:{},loss:{loss:.3f}, classification_loss:{classification_loss:.3f}, regression_loss:{regression_loss:.3f}'.format(
                epoch, iter_num, loss=np.mean(loss_hist),
                classification_loss=np.mean(loss_classification),
                regression_loss=np.mean(loss_regression)))
        
        check_img.append(img.cpu())
        check_annot.append(annot.cpu())
        check_color_label_gt.append(color_label_gt.cpu())
        
        '''
        the limit on how frequently we can check depends on how much memory the gpu has
        
        
        '''
        
        if iter_num % FREQ == 0 and iter_num > 0 or iter_num == len(tbar)-1: # catch the last round
            print ('checking at %d' %iter_num)
            patches = check_validity_predicted_bboxes(check_img, check_annot, check_color_label_gt, retinanet, iter_num)
            
            check_img = []
            check_annot = []
            check_color_label_gt = []
            
            if patches:
                img_list += patches[0]
                labels_list += patches[1]
    assert len(img_list) == len(labels_list), 'number of images should agree with the number of labels'
    
    if len(img_list) > 1:
        loss_classification_color, all_y_true, all_pred = train_classifier((img_list,
                                                                            labels_list),
            color_classifier, epoch, optimizer_color, criterion)
            
        p_dict, r_dict, f_dict, s_dict, overall_acc = calculate_full_batch_statistics(all_y_true, all_pred, epoch, 'train', color_labels,
                                                                                      save_dir, tag=str(epoch))
        
        writer.add_scalar('train/classification_color_loss', np.mean(loss_classification_color), epoch)
        writer.add_scalar('train/correct_per', overall_acc, epoch)
                
    writer.add_scalar('train/classification_loss', np.mean(loss_classification), epoch)
    writer.add_scalar('train/regression_loss', np.mean(loss_regression), epoch)
    writer.add_scalar('train/total_loss', np.mean(loss_hist), epoch)
    
    return np.mean(loss_hist)

def visualise(img_cpu, top_boxes, epoch, iter_num, patches, predicted_label):
    '''
    draw the detected bounding boxes and write the predictions on the image to see how good they are
    
    '''
    
    _,label = patches
    
    label = torch.cat(label)
    label = label.cpu().numpy()
    
    for to_draw in range(img_cpu.shape[0]):
        img = prepare_img_data_format(img_cpu, to_draw)
        caption_text_colors = []
        
        for (label_name, box_score), pred, gt_label in zip(top_boxes.items(),
             predicted_label, label):
            
            score, bbox = box_score 
            
            # draw the corners of the boxes
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
        
            caption = '{}/{.2f} pred: {} gt: {}'.format(label_name, score, color_labels[pred],
                       color_labels[gt_label])
            color= colors[inverse_label_meaning[label_name]]
            # make sure the datatype is correct for openCV
            color = (int(color[0]), int(color[1]), int(color[2]))
            
            caption_text_colors.append((caption, [c / 255.0 for c in color]))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        
        plt.figure()
        plt.imshow(img)
        for ie, item in enumerate(caption_text_colors):
            plt.text(0, ie*15, item[0], color=item[1])
        
        plt.savefig(os.path.join(save_dir_bbox, str(epoch) + '_'+ str(iter_num)+'_bbox.png'))
        plt.close()
        
        del img
        
def pair_up_gt_bbox_labels(annot, labels_meaning):
    a = annot.cpu().numpy()[0,...]
    gt_class = {labels_meaning[a[i, -1]]: a[i, 0:4] for i in range(a.shape[0])}
    return gt_class

def calculate_truePositive_falsePositive(TP_dict, FP_dict, total_dict, idxs, classification, transformed_anchors,
                                         scores, labels_meaning, annot):
    
    boxes_class = pair_up_bbox_color_labels(idxs, classification,
                                            transformed_anchors, scores, 
                                            labels_meaning)
    
    gt_boxes_class = pair_up_gt_bbox_labels(annot, labels_meaning)
    
    for shape, gt in gt_boxes_class.items():
        gt = torch.from_numpy(gt[None,...])
        detected = boxes_class.get(shape, None)
        
        if detected:
            # sort the boxes in descending order of confidence
            detected = sorted(detected, key=lambda x:x[0], reverse=True)
            detected_boxes = detected[0][-1].cpu().numpy()
            detected_boxes = detected_boxes[None,...]
            detected_boxes = torch.from_numpy(detected_boxes.astype(np.float32))
            
            _,_,ratio = calc_iou(detected_boxes, gt, return_intersection=True)
            
            ratio = ratio.cpu().numpy()
            
        else:
            ratio = 0
            
        
        number_of_positive = 1 # each image should have only one object of certain shape
        if ratio >= IOUTHRESHOLD:
            TP_dict[shape].append(1)
            FP_dict[shape].append(0)
        else:
            TP_dict[shape].append(0) # placeholder for cumsum
            FP_dict[shape].append(1)
            
        total_dict[shape] = total_dict[shape] + number_of_positive
    return TP_dict, FP_dict, total_dict

def create_shape_color_groundtruth_dict(shape_gt, c):
    shape_color_dict = {}
    for ii, (si, ci) in enumerate(zip(shape_gt.cpu().numpy().tolist()[0],
                                  c.cpu().numpy().tolist())):
        shape_color_dict[labels_meaning[si]] = (color_labels[ci], ii)
        
    return shape_color_dict

def unpack_good_overlapped_boxes(good_overlapped_boxes):
    
    big_enough_predicted_boxes, big_enough_inputs = [], []
    
    for shape, (score, bbox) in good_overlapped_boxes.items():
        big_enough_inputs.append(shape)
        big_enough_predicted_boxes.append(torch.unsqueeze(bbox, dim=0))
    
    return big_enough_inputs, torch.cat(big_enough_predicted_boxes)

def calculate_mAP(total_dict, TP_dict, FP_dict, epoch):
    ap_dict = {}
    for shape in total_dict:
        print ('\t', shape)
        acc_TP = np.cumsum(TP_dict[shape])
        acc_FP = np.cumsum(FP_dict[shape])
        rec = acc_TP / total_dict[shape]
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        
        ap, mpre, mrec, ii = calculate_average_precision(rec, prec)
        ap_dict[shape] = ap
        plot_precision_recall(rec, prec, ap, shape, tag=str(epoch), savePath=save_dir)
        
    mAP = np.mean([ i for i in ap_dict.values()])
    print ('average AP {:.2f}'.format(mAP))
    return ap_dict , mAP

def validate(dataloader_test, retinanet, color_classifier, epoch, criterion):
    retinanet.calculate_focalLoss = True
    retinanet.eval()
    color_classifier.eval()
    
    loss_hist = []
    loss_classification = []
    loss_regression = []
    loss_classification_color = []
    
    # initialise default value
    all_y_true = None
    all_pred = None
    
    TP_dict = defaultdict(list)
    FP_dict = defaultdict(list)
    total_dict = defaultdict(int)
    record_results = False # assume we have no patches/ predictions
    
    tbar = tqdm.tqdm(dataloader_test)
    with torch.no_grad():
        for iter_num, data in enumerate(tbar):
            
            img_gpu = data['img'].cuda()
            img_cpu = data['img']
            
            
            annot, color_label_gt = data['annot'][:,:,0:-1], data['annot'][:,:,-1]
            
            color_label_gt = color_label_gt.long()
            
            annot = annot.cuda()
            color_label_gt = color_label_gt.cuda()
            
            [scores, classification, transformed_anchors], focalLoss_values = retinanet(
                    [img_gpu, annot])
            classification_loss, regression_loss = focalLoss_values
            
          
            loss_classification += classification_loss.detach().cpu().numpy().tolist()
            loss_regression += regression_loss.detach().cpu().numpy().tolist()
            
            total_loss = classification_loss + regression_loss
            loss_hist += total_loss.detach().cpu().numpy().tolist()
            
            ####### calculate scores #############
            b,c,shape_gt, img_copied = copy_reorganize_inputs(annot, color_label_gt, img_gpu)
            shape_color_dict = create_shape_color_groundtruth_dict(shape_gt, c)
            
            idxs = scores_min_bar_test(scores)
            if idxs is not None:
                TP_dict, FP_dict, total_dict = calculate_truePositive_falsePositive(TP_dict, FP_dict, total_dict,
                                                                                    idxs,
                                                                                    classification,
                                                                                    transformed_anchors,
                                                                                    scores,
                                                                                    labels_meaning,
                                                                                    annot)
                good_overlapped_boxes = find_enough_overlap(b, shape_gt, idxs, scores, classification,
                                                            transformed_anchors, labels_meaning, per_label=True)
                
                if good_overlapped_boxes is not None:
                    
                    big_enough_inputs, big_enough_predicted_boxes =  unpack_good_overlapped_boxes(good_overlapped_boxes)
                    patches = prepare_patches(big_enough_predicted_boxes, big_enough_inputs,c, img_copied, shape_color_dict=shape_color_dict)
                    
                    if patches is not None:
                        color_classification_loss, predicted_label, gt_label = validate_classifier(
                                patches, color_classifier, epoch, criterion)
                        visualise(img_cpu, good_overlapped_boxes, epoch, iter_num, patches, predicted_label)
                        loss_classification_color += color_classification_loss
                        
                        if all_y_true is not None and all_pred is not None:
                            all_y_true = np.concatenate([all_y_true, gt_label], axis=0)
                            all_pred = np.concatenate([all_pred, predicted_label], axis=0)
                        else:
                            all_y_true = gt_label
                            all_pred = predicted_label
                        record_results = True
        print ('calculate mAP')
        ap_dict, mAP = calculate_mAP(total_dict, TP_dict, FP_dict, epoch)
        
        if record_results:
        
            p_dict, r_dict, f_dict, s_dict, overall_acc = calculate_full_batch_statistics(all_y_true, all_pred, epoch, 'valid', color_labels,
                                                                                          save_dir, tag=str(epoch))
            
            writer.add_scalar('valid/classification_color_loss', np.mean(loss_classification_color), epoch)
            writer.add_scalar('valid/correct_per', overall_acc, epoch)
            [writer.add_scalar('valid/{}_AP'.format(s), ap_dict[s], epoch) for s in  ap_dict]
                
    writer.add_scalar('valid/classification_loss', np.mean(loss_classification), epoch)
    writer.add_scalar('valid/regression_loss', np.mean(loss_regression), epoch)
    writer.add_scalar('valid/total_loss', np.mean(loss_hist), epoch)
    
    return np.mean(loss_hist), mAP, np.mean(loss_classification_color)


def get_files(p):
    return [os.path.join(p, 'csv', 'shapes.csv'),
            os.path.join(p, 'csv', 'class_shapes.csv'),
            os.path.join(p, 'csv', 'class_colors.csv')]
    
            
    
if __name__ == "__main__":
    
    # restore to previous model
    restore = False
    
    colors = np.array([[217,177,69],
                       [5,241,53],
                       [26,38,251]])
    train_source = None
    test_source = None
    
    transform = torchvision.transforms.Compose([
            Cast_to_Pytorch(255.0)])
        
    file, class_list, color_list = get_files(train_source)
                
    dataset_train = CSVDataset(train_file = file,
                               class_list = class_list,
                               color_list = color_list,
                               transform =transform)
    
    file, class_list, color_list = get_files(test_source)
                
    dataset_test = CSVDataset(train_file = file,
                               class_list = class_list,
                               color_list = color_list,
                               transform =transform)
    
    labels_meaning = dataset_train.labels
    inverse_label_meaning = {value:key for key, value in labels_meaning.items()}
    num_classes_shapes = len(labels_meaning)
    num_classes_colors = len(dataset_test.color_labels)
    color_labels = dataset_train.color_labels
    inverse_color_labels = {value:key for key, value in color_labels.items()}
    
    print ('labels_meaning {}'.format(labels_meaning))
    print ('color_labels {}'.format(color_labels))
    
    batch_size = 50
    batch_size_color_classifier = 20
    num_workers = 1
    
    '''
    in this toy example, all classes are well balanced.
    Otherwise, we may need to add a sampler to oversample and undersample minority and majority classes
    
    
    '''
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers = num_workers,
                                               pin_memory=False,
                                               drop_last = False)
    
    # during testing, look at the result one by one
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers = 1,
                                               pin_memory=False,
                                               drop_last = False)
    description = 'toy'
    print ('description {}'.format(description))
    
    print ('%d training images.' %len(dataset_train))
    print ('%d test images.' %len(dataset_test))
    
    save_dir = None
    save_dir_bbox = None
    logdir = None
    
    if not restore:
        # clean up
        make_clean_folder(save_dir)
        make_clean_folder(save_dir_bbox)
        make_clean_folder(logdir)
    writer = SummaryWriter(logdir=logdir, flush_secs=2)
    
    t0 = time.time()
    alpha = 0.25
    gamma = 2.0
    retinanet = resnet18(num_classes = num_classes_shapes, alpha=alpha, gamma=gamma)
    color_classifier = ColorClassifier(3,3,3)
    
    # uniform weight becuase no class imbalance
    class_weight = [1.0]* num_classes_colors
    print ('class_weight {}'.format(class_weight))
    
    criterion = torch.nn.CrossEntropyLoss(torch.Tensor(class_weight), reduction='none')
    criterion = place_on_cpu_gpu(criterion)
    
    #### restore retinaNet #####
    if restore:
        model_checkpoint = None
        model_basefolder = None

        path = os.path.join(model_basefolder, 'epoch_'+str(model_checkpoint)+'.pth')
        print ('restore model from {} at checkpoint {}'.format(model_basefolder,
               model_checkpoint))
        retinanet.load_state_dict(torch.load(path)['model.state_dict'])
        start_epoch = model_checkpoint + 1
    else:
        start_epoch = 0
    
    retinanet = place_on_cpu_gpu(retinanet)
    color_classifier = place_on_cpu_gpu(color_classifier)
    
    print ('building model took {:.2f}'.format(time.time()- t0))
    
    learning_rate = 1e-4
    
    optimizer = optim.Adam(retinanet.parameters(), lr=learning_rate)
    optimizer_color = optim.Adam(color_classifier.parameters(), lr=learning_rate)
    
    MAX_EPOCH = 100
    
    for epoch in range(start_epoch, MAX_EPOCH):
        total_loss_train = train(train_loader, retinanet, color_classifier, epoch, optimizer, optimizer_color, criterion)
        total_loss_valid, mAP, loss_classification_color = validate(test_loader, retinanet, color_classifier, epoch, criterion)
        
        if epoch == start_epoch:
            best_train_loss = total_loss_train
            
            best_valid_loss = total_loss_valid
            best_mAP = mAP
            best_loss_classification_color = loss_classification_color
            
        else:
            if total_loss_train < best_train_loss:
                save_checkpoint(retinanet, epoch, prefix=save_dir+'/retinanet_train')
                print ('training loss improves from {:.3f} down to {:.3f}'.format(best_train_loss,
                       total_loss_train))
                best_train_loss = total_loss_train
                
            if total_loss_valid < best_valid_loss:
                save_checkpoint(retinanet, epoch, prefix=save_dir+'/retinanet_valid')
                print ('validation loss improves from {:.3f} down to {:.3f}'.format(best_valid_loss,
                       total_loss_valid))
                best_valid_loss = total_loss_valid
                
            if mAP > best_mAP:
                save_checkpoint(retinanet, epoch, prefix=save_dir+'/retinanet_valid')
                print ('validation mAP improves from {:.3f} up to {:.3f}'.format(best_mAP,
                       mAP))
                best_mAP = mAP
                
            if loss_classification_color < best_loss_classification_color:
                save_checkpoint(color_classifier, epoch, prefix=save_dir+'/classifier_valid')
                print ('validation classification_color loss improves from {:.3f} down to {:.3f}'.format(
                        best_loss_classification_color,
                       loss_classification_color))
                best_loss_classification_color = loss_classification_color
                
            
                
    writer.close()
    print ('end')
                
                
    
        
                        
                        
            

            
                
        
    



