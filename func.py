import matplotlib as mpl
mpl.use('Agg') # use this to draw figures when there is no display screen
from matplotlib import pyplot as plt

import os, time, sys, random
import numpy as np
import glob, torch
import torch.nn.functional as F # convention
import torchvision
# we use tensorboard to see the results of training and validation
from tensorboardX import SummaryWriter 
from model_retinaNet import resnet18, ColorClassifier, calc_iou
import torch.optim as optim
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
import  shutil

leftmargin = 0.5
rightmargin = 0.5
categorysize = 0.5

def create_sample_indices(batch_size, x):
    N = x.shape[0]
    sample_ind = np.arange(N)
    random.shuffle(sample_ind)
    
    if N > batch_size:
        m = int(N/ batch_size)
        sample_ind = sample_ind[0:m*batch_size]
        return sample_ind.reshape(-1, batch_size)
    else:
        # not enough samples to fill up a single batch
        # occurs at the beginning, detector is poor
        # need to have axis 0 dimension to train
        return sample_ind.reshape(1,N)

def put_anchor_boxes_per_label_array(top_boxes):
    
    a = torch.zeros([len(top_boxes), 4])
    keys = sorted(top_boxes.keys())
    for j, key in enumerate(keys):
        box_score = top_boxes[key]
        _, bbox = box_score
        a[j,:] = bbox
    return a, {key:i for i, key in enumerate(keys)}

def prepare_img_data_format(img_cpu, to_draw):
    '''
    take a minibatch of images on the cpu
    use to_draw to select a single image
    '''
    
    # matplotlib takes H,W,C but pytorch uses C,H,W
    img = img_cpu[to_draw,...].permute(1,2,0).numpy()
    img = img.astype(np.float64)
    img *= 255.0
    img = np.ascontiguousarray(img, dtype=np.uint8) # CV prefers contiguousarray and uint8 format
    return img

def pair_up_bbox_color_labels(idxs, classification, transformed_anchors, scores,
                             labels_meaning):
    '''
    group the detected boxes by the predicted box classes
    '''
    
    boxes_class = defaultdict(list)
    for j in range(len(idxs)):
        i = idxs[j]
        bbox = transformed_anchors[i,:]
        label_name = labels_meaning[int(classification[i])]
        score = scores[j]
        boxes_class[label_name].append((score, bbox))
        
    return boxes_class

def locate_top_boxes(boxes_class):
    '''
    find detected boxes with high confidence scores
    '''
    
    top_boxes = {}
    for label_name, box_scores in boxes_class.items():
        if len(box_scores)>1:
            sorted_scores = sorted(box_scores, key=lambda x:x[0]) # sort by scores
            top_boxes[label_name] = sorted_scores[-1]
        else:
            top_boxes[label_name] = box_scores[0]
        
    return top_boxes

def cal_acc_F1_scores(all_pred_combined, all_y_true_combined, abridged, tag=None):
    
    acc = accuracy_score(all_y_true_combined, all_pred_combined)
    
    precision, recall, f1_score_arr, support = precision_recall_fscore_support(all_y_true_combined,
                                                                               all_pred_combined,
                                                                               labels=[i for i in range(len(abridged))],
                                                                               average=None)
    
    p_dict = {'p_'+abridged[i]:precision[i] for i in range(len(precision))}
    r_dict = {'r_'+abridged[i]: recall[i] for i in range(len(recall))}
    f_dict = {'f1_'+abridged[i]: f1_score_arr[i] for i in range(len(f1_score_arr))}
    s_dict = {'s_'+abridged[i]: support[i] for i in range(len(support))}
    
    return p_dict, r_dict, f_dict, s_dict, acc

def calculate_full_batch_statistics(all_y_true, all_pred, epoch, stage, one_hot_dict_meaning, this_trial, tag=None):
    
    draw_confusion_matrix(all_y_true, all_pred, one_hot_dict_meaning, epoch, stage, this_trial, tag)
    p_dict, r_dict, f_dict, s_dict, acc = cal_acc_F1_scores(all_pred, all_y_true, one_hot_dict_meaning, tag=tag)
    return p_dict, r_dict, f_dict, s_dict, acc

def draw_confusion_matrix(all_y_true, all_pred, inverse_one_hot_dict, epoch, mode, this_trial, tag=None):
    
    def put_matrix_on_paper(cm, normalize, have_labels, tag):
        title = mode
        if tag is not None:
            if not isinstance(tag, str):
                tag = str(tag)
            title = title + '_' + tag
            
        if normalize:
            title = title + '_norm'
            
        figwidth = leftmargin + rightmargin + (cm.shape[0]*categorysize)
        
        fig, ax = plt.subplots(figsize = (figwidth, figwidth))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # only annoate classes that are present
        txt = [inverse_one_hot_dict[value] for value in have_labels]
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks = np.arange(cm.shape[1]),
               xticklabels = txt,
               yticklabels = txt,
               title = title,
               ylabel = 'True labels',
               xlabel = 'predicted labels')
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max()/2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j,i, format(cm[i,j], fmt),
                        ha='center',
                        va='center',
                        color='white' if cm[i,j] > thresh else 'black')
        plt.xticks(rotation = 90)
        fig.tight_layout()
        
        if not os.path.exists(this_trial):
            os.makedirs(this_trial)
        plt.savefig(os.path.join(this_trial, title), dpi=100)
        plt.close()
        
    have_labels = sorted(np.unique(all_y_true))
    cm = confusion_matrix(all_y_true, all_pred, labels=have_labels)
    
    for normalize in [True, False]:
        if normalize:
            cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        else:
            cm2 = cm
        put_matrix_on_paper(cm2, normalize, have_labels, tag)
        
def save_checkpoint(model, epoch, optimizer=None,
                    prefix='./checkpoints'):
    filename = os.path.join(prefix, 'epoch_'+str(epoch)+ '.pth')
    if not os.path.exists(prefix):
        os.makedirs(prefix)
           
    save_dict = {'model.state_dict':model.state_dict()}
    if optimizer:
        
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
    try:
        torch.save(save_dict, filename)
        print ('saved checkpoint to {}'.format(filename))
        return filename
    except OSError as e:
        print ('skip saving becuase {}'.format(e))
        return None
    
def make_clean_folder(this_trial):
    
    if not os.path.exists(this_trial):
        os.mkdir(this_trial)
    else:
        # empty folder
        if os.path.isdir(this_trial):
        
            try:
                shutil.rmtree(this_trial)
            except NotADirectoryError:
                pass
            
            # make a brandnew one
            os.mkdir(this_trial)

        else:
            print ('the model folder is NOT a directory {}'.format(this_trial))
       
                        
def calculate_average_precision(rec, prec):
    assert len(rec) == len(prec), 'recall and precision should have the same number of operating points {}, {}'.format(
            len(rec), len(prec))
    
    mrec = []
    # the value should start at 0
    mrec.append(0)
    [mrec.append(e) for e in rec]
    # the value should end at 1
    mrec.append(1)
    
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    
    ii = []
    for i in range(len(mrec)-1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i+1)
    ap = 0
    for i in ii:
        ap = ap + (mrec[i] - mrec[i-1]) * mpre[i]
        
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

def plot_precision_recall(recall, precision, average_precision, class_name, tag=None, savePath=None, showAP=True):
    plt.figure()
    plt.plot(recall, precision, label='precision')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.ylim([0,1.05])
    plt.xlim([0, 1.05])

    if showAP:
        ap_str = '{0:.2f}%'.format(average_precision * 100)
        plt.title('Precision x Recall curve \n Class: %s, AP: %s' %(str(class_name),
                                                                    ap_str))
    plt.legend(loc='best')                                
    plt.grid()
    
    if savePath:
        plt.savefig(os.path.join(savePath, tag+'_'+class_name+'.png'))
    plt.close()
    
    
    