import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import glob, json
from PIL import Image, ImageFile
from skimage import measure
import scipy
import imageio
from scipy.ndimage import map_coordinates

from setup import *

ImageFile.LOAD_TRUNCATED_IMAGES = True
POLYP_ONLY = True


@DATASETS.register_module()
class PolypDatasetTest(CustomDataset):

    # CLASSES = ('polyp', 'instrument')

    def get_image_bbox(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        sample = {'image': _img, 'label': _target}
        augmented_mask = sample["label"]
        gt_image = sample['image']
        gt_bboxs = self._mask_to_bbox_region(augmented_mask)

        return gt_image, gt_bboxs, augmented_mask

    def _mask_to_bbox(self, coco_mask):
        class_id = np.unique(coco_mask)
        bboxs = []
        for i in class_id:
            binary_mask = np.zeros(coco_mask.shape[:2], dtype=np.uint8)
            if i == 0:
                continue
            binary_mask[coco_mask == i] = 1
            contours = measure.find_contours(binary_mask, 0.9, fully_connected='high')
            for contour in contours:
                contour = np.flip(contour, axis=1)
                min_x = np.min(contour[:, 0])
                min_y = np.min(contour[:, 1])
                max_x = np.max(contour[:, 0])
                max_y = np.max(contour[:, 1])
                area = (max_x - min_x) * (max_y - min_y)
                if area < self.mask_min_size:
                    continue
                bbox = [min_x, min_y, max_x, max_y, i]
                bboxs.append(bbox)

        return np.array(bboxs, dtype=np.int)

    def _mask_to_bbox_region(self, coco_mask):
        class_id = np.unique(coco_mask)
        bboxs = []
        for i in class_id:
            binary_mask = np.zeros(coco_mask.shape[:2], dtype=np.uint8)
            if i == 0:
                continue
            binary_mask[coco_mask == i] = 1
            labeled_mask = measure.label(binary_mask)
            regions = measure.regionprops(labeled_mask)
            for prop in regions:
                if prop.area < self.mask_min_size:
                    continue
                min_x = prop.bbox[1]
                min_y = prop.bbox[0]
                max_x = prop.bbox[3]
                max_y = prop.bbox[2]
                bbox = [min_x, min_y, max_x, max_y, i]
                bboxs.append(bbox)

        return np.array(bboxs, dtype=np.int)

    def _mask_to_bbox_scipy(self, coco_mask):
        class_id = np.unique(coco_mask)
        bboxs = []
        for i in class_id:
            if i == 0:
                continue
            binary_mask = np.zeros(coco_mask.shape[:2], dtype=np.uint8)
            binary_mask[coco_mask == i] = 1
            label_im, nb_labels = scipy.ndimage.label(binary_mask)
            sizes = scipy.ndimage.sum(binary_mask, label_im, range(nb_labels + 1))

            valid_seg_indices = []
            for seg_index, seg_size in enumerate(sizes):
                if seg_size > 1:
                    valid_seg_indices.append(seg_index)
            mask_size = sizes < 10
            remove_pixel = mask_size[label_im]
            label_im[remove_pixel] = 0
            new_label_im, new_nb_labels = scipy.ndimage.label(label_im)
            rois = np.array([(new_label_im == ii) * 1 for ii in range(1, new_nb_labels + 1)])

            for rix, r in enumerate(rois):
                if np.sum(r != 0) > 0:  # check if the lesion survived data augmentation
                    seg_ixs = np.argwhere(r != 0)
                    coord_list = [int(np.min(seg_ixs[:, 1]) - 1), int(np.min(seg_ixs[:, 0]) - 1),
                                  int(np.max(seg_ixs[:, 1]) + 1), int(np.max(seg_ixs[:, 0]) + 1), i]
                    bboxs.append(coord_list)
        return np.array(bboxs, dtype=np.int)

    def _make_img_gt_point_pair(self, index):

        _image = imageio.imread(self.image_paths[index])

        _target = imageio.imread(self.mask_paths[index])
        if POLYP_ONLY:
            _target[_target > 1] = 0

        return _image, _target

    # def __getitem__(self, index):
    #     sample = {}
    #     gt_image, gt_bboxs, augmented_mask = self.get_image_bbox(index)
    #     gt_targets = torch.FloatTensor(gt_bboxs)
    #     sample['gt_image'] = torch.from_numpy(gt_image)
    #     sample['gt_bbox'] = gt_targets
    #     return sample

    def load_annotations(self, ann_file):
        # ann_list = mmcv.list_from_file(ann_file)

        base_dir = ann_file
        anno_files = glob.glob(os.path.join(base_dir, "annos/{}".format('test'), '*.json'))
        assert len(anno_files) > 0, 'No annotation files locat at: {}'.format(
            os.path.join(base_dir, "annos/{}".format('test')))

        # minimum mask size
        self.mask_min_size = 0
        self.img_dir = os.path.join(base_dir, "images/")
        self.split = 'test'

        self.image_paths = []
        self.mask_paths = []
        public_dataset = [
            'cvc300',
            'CVC-ClinicDB',
            'ETIS',
            'Segmentation'
        ]
        for anno_path in anno_files:
            with open(anno_path, 'r') as f:
                annotation = json.load(f)

            im_path = annotation['images']
            if len(im_path) > 0:
                im_file = im_path[0]['file_name']
                base_name = os.path.dirname(im_file)
            else:
                base_name = os.path.basename(anno_path)
            file_name_without_extention = base_name.split('.')[0]
            if 'polyp' in file_name_without_extention:
                _fsplit = [file_name_without_extention]
            elif '_p' in file_name_without_extention:
                _fsplit = file_name_without_extention.split('_p')
            else:
                _fsplit = file_name_without_extention.split('_P')

            base_name_without_p_index = _fsplit[0]
            folder_num = '' if len(_fsplit) == 1 else _fsplit[1]
            im_dir = os.path.join(base_dir, 'images', base_name_without_p_index, folder_num)

            assert os.path.isdir(im_dir), im_dir
            for dirName, subdirList, fileList in os.walk(im_dir):
                # assert len(fileList) > 0
                for file in fileList:
                    self.image_paths.append(os.path.join(dirName, file))

                    file_name, ext = file.split('.')
                    if ext == 'tif':
                        mask_file = file_name + '.tif'
                    elif ext == 'tiff':
                        if 'ColonDB' in dirName:
                            file_name = 'p' + file_name
                        mask_file = file_name + '.tiff'
                    elif ext == 'bmp':
                        mask_file = file_name + '.bmp'
                        if 'Segmentation' in dirName:
                            mask_file = file_name + '_mask.tif'
                    else:
                        if 'cvc' in dirName:
                            mask_file = file_name + '.png'
                        elif 'blur' in dirName:
                            mask_file = file_name + '.jpg'
                        elif 'image_without_polyp' in dirName:
                            mask_file = file_name + '.jpg'
                        else:
                            mask_file = file_name + '_mask.png'

                    def check_if_public(dname):
                        for i in public_dataset:
                            if i in dname:
                                return True
                        return False

                    if check_if_public(dirName):
                        mask_dirName = os.path.dirname(os.path.normpath(dirName))
                    else:
                        mask_dirName = dirName
                    mask_dir = os.path.relpath(mask_dirName, base_dir).replace('images', 'mask')
                    mask_path = os.path.join(base_dir, mask_dir, mask_file)
                    assert os.path.isfile(mask_path), mask_path
                    self.mask_paths.append(mask_path)

        assert len(self.image_paths) == len(self.mask_paths)
        print('{} set contains {} images'.format('test', len(self.image_paths)))
        # print('\n', '*' * 80, '\n', 'image path: ', self.image_paths)

        data_infos = []
        for i, file_name in enumerate(self.image_paths):
            # print('\n', '*' * 80, '\n', 'file_name in image_path: ', file_name)
            gt_image, gt_bboxs, augmented_mask = self.get_image_bbox(i)
            img_shape = gt_image.shape
            width = int(img_shape[0])
            height = int(img_shape[1])

            bboxes = []
            labels = []
            for j in range(len(gt_bboxs)):
                bboxes.append(gt_bboxs[j][:4])
                labels.append(gt_bboxs[j][4])

            data_infos.append(
                dict(
                    filename=file_name,
                    width=width,
                    height=height,
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64))
                ))
            # print('data_info appended: ', data_infos)

        # print('\n', '*' * 80, '\n', 'data_infos in total: ', data_infos)

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']
