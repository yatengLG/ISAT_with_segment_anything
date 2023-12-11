# -*- coding: utf-8 -*-
# @Author  : LG

from ISAT.scripts.isat import ISAT
from xml.etree import ElementTree as ET
import tqdm
import os


class VOCDetect(ISAT):
    def __init__(self):
        self.keep_crowd = True

    def save_to_XML(self, xml_root):
        os.makedirs(xml_root, exist_ok=True)

        pbar = tqdm.tqdm(self.annos.items())
        for name_without_suffix, anno in pbar:
            xml_path = os.path.join(xml_root, name_without_suffix + '.xml')
            try:
                self._save_one_voc_xml(anno, xml_path)
                pbar.set_description('Save xml to {}'.format(name_without_suffix + '.xml'))
            except Exception as e:
                raise '{} {}'.format(name_without_suffix, e)
        return True

    def _save_one_voc_xml(self, anno, xml_path):
        annotation = ET.Element('annotation')
        tree = ET.ElementTree(annotation)
        folder = ET.Element('folder')
        folder.text = '{}'.format(anno.info.folder)
        annotation.append(folder)

        filename = ET.Element('filename')
        filename.text = '{}'.format(anno.info.name)
        annotation.append(filename)

        explain = ET.Element('explain')
        explain.text = '{}'.format('XML from ISAT')
        annotation.append(explain)

        size = ET.Element('size')
        width = ET.Element('width')
        width.text = '{}'.format(anno.info.width)
        size.append(width)

        height = ET.Element('height')
        height.text = '{}'.format(anno.info.height)
        size.append(height)

        depth = ET.Element('depth')
        depth.text = '{}'.format(anno.info.depth)
        size.append(depth)
        annotation.append(size)

        objects = anno.objs
        objects_groups = [obj.group for obj in objects]
        objects_groups.sort()
        objects_groups = set(objects_groups)

        # 同group为同一个目标，需合并包围盒
        for group_index, group in enumerate(objects_groups):
            objs_with_group = [obj for obj in objects if obj.group == group]
            cats = [obj.category for obj in objs_with_group]
            cats = set(cats)

            # 同category
            for cat in cats:
                objs_with_cat = [obj for obj in objs_with_group if obj.category == cat]
                crowds = [obj.iscrowd for obj in objs_with_group]
                crowds = set(crowds)

                segmentations = []

                for obj in objs_with_cat:
                    if not self.keep_crowd and obj.iscrowd:
                        continue

                    segmentations.extend(obj.segmentation)

                xs = [x for x, y in segmentations]
                ys = [y for x, y in segmentations]

                object = ET.Element('object')
                name = ET.Element('name')
                name.text = cat
                object.append(name)

                pose = ET.Element('pose')
                pose.text = 'Unspecified'
                object.append(pose)

                truncated = ET.Element('truncated')
                truncated.text = '0'
                object.append(truncated)

                difficult = ET.Element('difficult')
                difficult.text = '0'
                object.append(difficult)

                bndbox = ET.Element('bndbox')
                xmin = ET.Element('xmin')
                xmin.text = '{}'.format(int(min(xs)))
                bndbox.append(xmin)

                ymin = ET.Element('ymin')
                ymin.text = '{}'.format(int(min(ys)))
                bndbox.append(ymin)

                xmax = ET.Element('xmax')
                xmax.text = '{}'.format(int(max(xs)))
                bndbox.append(xmax)

                ymax = ET.Element('ymax')
                ymax.text = '{}'.format(int(max(ys)))
                bndbox.append(ymax)
                object.append(bndbox)
                annotation.append(object)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        return True