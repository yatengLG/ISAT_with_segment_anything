# -*- coding: utf-8 -*-
# @Author  : LG

from ISAT.scripts.isat import ISAT
import cv2
import tqdm
import numpy as np
import os


class YOLO(ISAT):
    """
    YOLO use txt to save annotations. Every line container a annotation.
    format: class_index, x1, y1, x2, y2, x3, y3, ....

    save_to_YOLO:
        针对多个一个目标包含多个多边形的情况，参考了yolo8 coco2yolo的转换代码，在多边形之间拉一条直线，将多个多边形组合为单个多边形进行保存。
        源代码地址: https://github.com/ultralytics/JSON2YOLO/blob/c38a43f342428849c75c103c6d060012a83b5392/general_json2yolo.py
    """
    def __init__(self):
        self.keep_crowd = True

    def read_from_YOLO(self, img_root, txt_root, class_dict=None):
        img_files = os.listdir(img_root)

        pbar = tqdm.tqdm(img_files)
        for img_name in pbar:
            name_without_suffix = self.remove_file_suffix(img_name)
            txt_path = os.path.join(txt_root, name_without_suffix+'.txt')
            image_path = os.path.join(img_root, img_name)
            if not os.path.exists(txt_path):
                continue

            anno = self._load_one_yolo_txt(image_path, txt_path, class_dict)
            self.annos[name_without_suffix] = anno
            pbar.set_description('Load yolo txt {}'.format(name_without_suffix+'.txt'))

        # cates
        if class_dict is not None:
            self.cates = tuple(class_dict.values())
        else:
            class_set = set()
            for _, anno in self.annos.items():
                for obj in anno.objs:
                    class_set.add(obj.category)
            class_set = list(class_set)
            class_set.sort()
            self.cates = tuple(class_set)
        return True

    def save_to_YOLO(self, txt_root):
        os.makedirs(txt_root, exist_ok=True)
        cates_index_dict = {cat:index for index, cat in enumerate(self.cates)}

        with open(os.path.join(txt_root, 'classification.txt'), 'w') as f:
            for cat in self.cates:
                f.write('{}\n'.format(cat))

        pbar = tqdm.tqdm(self.annos.items())
        for name_without_suffix, anno in pbar:
            txt_path = os.path.join(txt_root, name_without_suffix+'.txt')
            pbar.set_description('Integrate {}'.format(name_without_suffix))
            try:
                self._save_one_yolo_txt(anno, txt_path, cates_index_dict)
                pbar.set_description('Save yolo to {}'.format(name_without_suffix+'.txt'))

            except Exception as e:
                raise '{} {}'.format(name_without_suffix, e)
        return True

    @staticmethod
    def merge_multi_segment(segments):
        """
        https://github.com/ultralytics/JSON2YOLO/blob/c38a43f342428849c75c103c6d060012a83b5392/general_json2yolo.py#L324

        Merge multi segments to one list.
        Find the coordinates with min distance between each segment,
        then connect these coordinates with one thin line to merge all
        segments into one.

        Args:
            segments(List(List)): original segmentations in coco's json file.
                like [segmentation1, segmentation2,...],
                each segmentation is a list of coordinates.
        """

        def min_index(arr1, arr2):
            """
            https://github.com/ultralytics/JSON2YOLO/blob/c38a43f342428849c75c103c6d060012a83b5392/general_json2yolo.py#L324
            Find a pair of indexes with the shortest distance.
            Args:
                arr1: (N, 2).
                arr2: (M, 2).
            Return:
                a pair of indexes(tuple).
            """
            dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
            return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

        s = []
        segments = [np.array(i).reshape(-1, 2) for i in segments]
        idx_list = [[] for _ in range(len(segments))]

        # record the indexes with min distance between each segment
        for i in range(1, len(segments)):
            idx1, idx2 = min_index(segments[i - 1], segments[i])
            idx_list[i - 1].append(idx1)
            idx_list[i].append(idx2)

        # use two round to connect all the segments
        for k in range(2):
            # forward connection
            if k == 0:
                for i, idx in enumerate(idx_list):
                    # middle segments have two indexes
                    # reverse the index of middle segments
                    if len(idx) == 2 and idx[0] > idx[1]:
                        idx = idx[::-1]
                        segments[i] = segments[i][::-1, :]

                    segments[i] = np.roll(segments[i], -idx[0], axis=0)
                    segments[i] = np.concatenate([segments[i], segments[i][:1]])
                    # deal with the first segment and the last one
                    if i in [0, len(idx_list) - 1]:
                        s.append(segments[i])
                    else:
                        idx = [0, idx[1] - idx[0]]
                        s.append(segments[i][idx[0]:idx[1] + 1])

            else:
                for i in range(len(idx_list) - 1, -1, -1):
                    if i not in [0, len(idx_list) - 1]:
                        idx = idx_list[i]
                        nidx = abs(idx[1] - idx[0])
                        s.append(segments[i][nidx:])
        return s

    @staticmethod
    def yolo2isat_segmentation(yolo_seg, img_width, img_height):
        """Convert YOLO segmentation format to ISAT segmentation format"""
        return [[round(x * img_width), round(y * img_height)] for x, y in zip(yolo_seg[::2], yolo_seg[1::2])]

    @staticmethod
    def get_isat_bbox(segmentation):
        """Calculate the bbox from the ISAT segmentation"""
        xs = [point[0] for point in segmentation]  # x-coordinates
        ys = [point[1] for point in segmentation]  # y-coordinates
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

    @staticmethod
    def bbox_within(bbox_1, bbox_2):  # 这个函数查看两个物体的边框，如果是包含关系的话分到一个组
        """Check if two objects belong to the same group"""
        return all(bbox_1[idx] >= bbox_2[idx] for idx in [0, 1]) and all(bbox_1[idx] <= bbox_2[idx] for idx in [2, 3])

    def _load_one_yolo_txt(self, image_path, txt_path, class_dict=None):

        anno = self.ANNO()
        anno.info = self.ANNO.INFO()

        image = cv2.imread(image_path)  # load the image in BRG scale

        image_width, image_height = image.shape[1], image.shape[0]  # get the image dimensions

        if image.ndim == 2:
            image_depth = 1
        elif image.ndim == 3:
            image_depth = image.shape[2]
        else:
            image_depth = None

        img_root, img_name = os.path.split(image_path)
        anno.info.description = ''
        anno.info.folder = img_root
        anno.info.name = img_name
        anno.info.width = image_width
        anno.info.height = image_height
        anno.info.depth = image_depth

        objects = []
        group, layer = 1, 1.0  # initialize layer as a floating point number

        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                obj = self.ANNO.OBJ()

                parts = line.split()  # split each line
                class_index = int(parts[0])  # get the class index
                yolo_segmentation = list(map(float, parts[1:]))  # get the yolo_segmentation
                isat_segmentation = self.yolo2isat_segmentation(yolo_segmentation, image_width,
                                                                image_height)  # convert yolo_segmentation to isat_segmentation
                bbox = self.get_isat_bbox(isat_segmentation)  # calculate the bbox from segmentation
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[
                    1])  # roughly calculate the bbox area as segmentation area, it will be replaced anyway

                obj.category = class_index if class_dict is None else class_dict.get(class_index, 'UNKNOW')
                obj.group = group
                obj.segmentation = isat_segmentation
                obj.area = area
                obj.layer = layer
                obj.bbox = bbox
                obj.iscrowd = False

                group += 1
                objects.append(obj)

        anno.objs = tuple(objects)
        return anno

    def _save_one_yolo_txt(self, anno, save_path, cates_index_dict):
        with open(save_path, 'w') as f:
            objects = anno.objs

            objects_groups = [obj.group for obj in objects]
            objects_groups.sort()
            objects_groups = set(objects_groups)
            # 同group
            for group_index, group in enumerate(objects_groups):
                objs_with_group = [obj for obj in objects if obj.group == group]
                cats = [obj.category for obj in objs_with_group]
                cats = set(cats)

                # 同category
                for cat in cats:
                    objs_with_cat = [obj for obj in objs_with_group if obj.category == cat]
                    crowds = [obj.iscrowd for obj in objs_with_group]
                    crowds = set(crowds)
                    # 同iscrowd
                    class_index = cates_index_dict.get(cat)
                    segmentations = []

                    for obj in objs_with_cat:
                        if not self.keep_crowd and obj.iscrowd:
                            continue
                        segmentation = [[x / anno.info.width, y / anno.info.height] for x, y in obj.segmentation]
                        segmentation = [c for xy in segmentation for c in xy]
                        segmentations.append(segmentation)

                    if len(segmentations) > 1:
                        segmentations = self.merge_multi_segment(segmentations)
                        segmentations = (np.concatenate(segmentations, axis=0)).reshape(-1).tolist()
                    else:
                        segmentations = segmentations[0]

                    s = '{}' + ' {}' * len(segmentations) + '\n'
                    f.write(s.format(class_index, *segmentations))
        return True