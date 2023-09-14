import os
import sys
import cv2
import argparse


sys.path.insert(0, 'lib')
from lib.utils import misc_utils, visual_utils
#test
img_root = '../tools/outputs/Images-val'
def eval_all(args):
    # json file
    assert os.path.exists(args.json_file), "Wrong json path!"
    misc_utils.ensure_dir('outputs')
    records = misc_utils.load_json_lines(args.json_file)[:args.number]
    for record in records:
        dtboxes = misc_utils.load_bboxes(
                record, key_name='dtboxes', key_box='fbox', key_score='score', key_tag='tag')
        # gtboxes = misc_utils.load_bboxes(record, 'gtboxes', 'box')
        dtboxes = misc_utils.xywh_to_xyxy(dtboxes)
        # gtboxes = misc_utils.xywh_to_xyxy(gtboxes)
        keep = dtboxes[:, -2] > args.visual_thresh      #visual_thresh=0.3
        dtboxes = dtboxes[keep]
        len_dt = len(dtboxes)
        # len_gt = len(gtboxes)
        line = "{}: dt:{}.".format(record['ID'], len_dt)
        print(line)
        img_path = img_root + '/' + record['ID'] + '.jpg'
        # print("visulize_json.py....29l....img_path:", img_path)
        img = misc_utils.load_img(img_path)
        print("img_path: ", img_path)
        #print("img ",img)
        visual_utils.draw_boxes(img, dtboxes, line_thick=1, line_color='blue')
        # visual_utils.draw_boxes(img, gtboxes, line_thick=1, line_color='white')
        fpath = 'outputs/{}.jpg'.format(record['ID'])
        cv2.imwrite(fpath, img)



def run_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', '-f', default='../model/rcnn_emd_simple/outputs/eval_dump/dump-40.json', required=False, type=str)
    parser.add_argument('--number', '-n', default=4, type=int)
    parser.add_argument('--visual_thresh', '-v', default=0.3, type=int)
    args = parser.parse_args()
    eval_all(args)

if __name__ == '__main__':
    run_eval()
