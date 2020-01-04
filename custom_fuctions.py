import numpy as np
import torch
cls_nums = {'cifar10': 10, 'cifar100': 100}
img_maxs = {'cifar10': 5000, 'cifar100': 500}


def get_img_num_per_cls(imb_factor, dataset):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    cls_num = cls_nums[dataset]
    img_max = img_maxs[dataset]
    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    CARDINITY = torch.tensor(img_num_per_cls)
    return img_num_per_cls


def get_imb_subset(targets, cifar_imb_factor, dataset):
    img_num_per_cls = get_img_num_per_cls(cifar_imb_factor, dataset)
    indices = []
    y = np.array(targets)
    for class_idx in range(len(img_num_per_cls)):
        # position of samples belonging to class_idx
        current_class_all_idx = np.argwhere(y == class_idx)
        # convert the result into a 1-D list
        current_class_all_idx = list(current_class_all_idx[:,0])
        current_class_selected_idx = list(np.random.choice(current_class_all_idx, img_num_per_cls[class_idx], replace=False))
        indices = indices + current_class_selected_idx
    return indices


def shot_acc(preds, labels, data, many_shot_thr=100, low_shot_thr=20):
    
    training_labels = np.array(data.dataset.labels).astype(int)
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append(np.array(preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))          
    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)

