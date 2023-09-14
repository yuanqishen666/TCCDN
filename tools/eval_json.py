import os
import sys
import argparse

sys.path.insert(0, 'lib')
from lib.utils import misc_utils
from lib.evaluate import compute_JI, compute_APMR

def eval_all(args):
    # ground truth file
    gt_path = '/media/rtx2080ti/6c33c0e1-2391-4b12-8cf8-7a2cdc25e009/yqs/CrowdDet-V4_modify_3/tools/lib/data/CrowdHuman/annotation_val.odgt'
    assert os.path.exists(gt_path), "Wrong ground truth path!"
    misc_utils.ensure_dir('outputs')
    # output file
    eval_path = os.path.join('outputs', 'result_eval.md')
    eval_fid = open(eval_path,'a')
    eval_fid.write((args.json_file+'\n'))
    # eval JI
    res_line, JI = compute_JI.evaluation_all(args.json_file, 'fbox')  #modify:'box' --> 'fbox'
    for line in res_line:
        eval_fid.write(line+'\n')
    line = 'JI:{:.4f}.'.format(JI)
    # eval AP, MR
    AP, MR = compute_APMR.compute_APMR(args.json_file, gt_path, 'fbox') #modify:'box' --> 'fbox'
    line = 'AP:{:.4f}, MR:{:.4f}'.format(AP, MR)
    line = 'AP:{:.4f}, MR:{:.4f}, JI:{:.4f}.'.format(AP, MR, JI)


    print(line)
    eval_fid.write(line+'\n\n')
    eval_fid.close()

def run_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', '-f', default='/media/rtx2080ti/6c33c0e1-2391-4b12-8cf8-7a2cdc25e009/yqs/CrowdDet-V4_modify_3/tools/lib/data/CrowdHuman/annotation_val.odgt', required=False, type=str)
    args = parser.parse_args()
    eval_all(args)

if __name__ == '__main__':
    run_eval()
