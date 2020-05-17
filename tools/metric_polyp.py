import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
#from scipy.interpolate import spline
#from scipy.interpolate import make_interp_spline, BSpline
from matplotlib import colors

# import json
class Metric(object):
    def __init__(self, mode='center',iou_thresh=0,visualize = True,visualization_root='/data1/zinan_xiong/'):

        self.TPs = []
        self.FNs = []
        self.FPs = []
        self.pred_bbox_count = 0
        assert mode == 'center' or mode =='iou' , '({}) mode is not supported'
        self.mode = mode
        self.iou_thresh=iou_thresh

        #in BGR order
        # Blue
        self.FP_color= (255, 0, 0)
        # Green
        self.Detection_color = (0, 255, 0)
        # Red
        self.GT_color = (0, 0, 255)
        self.visualize=visualize
        self.total_gt =0.0
        if visualize:
            #  create image folder for saving detection result
            self.detection_folder = visualization_root + 'ALL/'
            self.false_positive_folder = visualization_root + 'FP/'
            self.false_negative_folder = visualization_root + 'FN/'
            os.makedirs(self.detection_folder, exist_ok=True)
            os.makedirs(self.false_positive_folder, exist_ok=True)
            os.makedirs(self.false_negative_folder, exist_ok=True)
            os.popen('rm -r ' + self.detection_folder + '*')
            os.popen('rm -r ' + self.false_positive_folder + '*')
            os.popen('rm -r ' + self.false_negative_folder + '*')

    def eval_add_result(self, ground_truth:list, pred_points:list, image:np.ndarray = None, image_name= None):
        if self.visualize:
            FPimage = image.copy()
            FNimage = image.copy()
            Detectionimage = image.copy()


            for pt in pred_points:
                pt1 = tuple([int(pt[0]), int(pt[1])])
                pt2 = tuple([int(pt[2]), int(pt[3])])
                cv2.rectangle(Detectionimage, pt1, pt2,self.Detection_color, 2)

        missing = False
        self.total_gt += len(ground_truth)
        for index_gt_box, gt_box in enumerate(ground_truth):
            hasTP = False
            gt = gt_box[0]
            TP_Count = 0
            # print("gt_box 0: ", gt[0], "\n")

            not_matched = []
            for pred_point in pred_points:
                for j in pred_point:
                    self.pred_bbox_count += 1
                    print("pred_boxes type: {}".format([type(i) for i in j], "\n"))
                    if self.mode == 'center':
                        ctx = j[0] + (j[2] - j[0]) * 0.5
                        cty = j[1] + (j[3] - j[1]) * 0.5
                        bbox_matched=gt[0] < ctx < gt[2] and gt[1] < cty < gt[3]

                    elif self.mode =='iou':
                        query_area =(j[2] - j[0])*(j[3] - j[1])
                        gt_area = (gt[2] - gt[0])*(gt[3] - gt[1])
                        iw = (min(j[2],gt[2])-max(j[0],gt[0]))
                        ih =(min(j[3],gt[3])-max(j[1],gt[1]))
                        iw =max(0, iw)
                        ih = max(0, ih)
                        ua = query_area+gt_area -(iw*ih)
                        overlaps = (iw*ih)/float(ua)
                        bbox_matched= overlaps>self.iou_thresh

                    if bbox_matched:
                        TP_Count += 1
                        # TODO seems not that right here.
                        # if not hasTP, only pick one TP from the list, discard others since they won't intercept:
                        if not hasTP:
                            self.TPs.append(j)
                            hasTP = True
                    else:
                        not_matched.append(j)
            pred_points = not_matched
            # if TP_Count > 0:
            #     hasTP = True

            #pred_points = not_matched
            #self.FPs += len(not_matched)

            if not hasTP:
                self.FNs.append(gt)

                if self.visualize:
                    # Draw False negative rect
                    missing = True
                    pt1 = tuple([int(gt[0]), int(gt[1])])
                    pt2 = tuple([int(gt[2]), int(gt[3])])
                    cv2.rectangle(FNimage, pt1, pt2, self.GT_color, 2)


            if self.visualize:
                # Draw groundturth on detection and FP images
                pt1 = tuple([int(gt[0]), int(gt[1])])
                pt2 = tuple([int(gt[2]), int(gt[3])])
                cv2.rectangle(Detectionimage, pt1, pt2, self.GT_color, 2)
                cv2.rectangle(FPimage, pt1, pt2, self.GT_color, 2)
        # print("FP: ", len(self.FPs), self.FPs, '\n', '*' * 80)
        # print('TP: ', len(self.TPs), self.TPs, '\n', '*' * 80)
        # for idx, fp in enumerate(self.FPs):
        #     if fp in self.TPs:
        #         self.FPs.pop(idx)

        if self.visualize:
            if missing:
                cv2.imwrite(self.false_negative_folder+str(image_name)+'.jpg', FNimage)
            cv2.imwrite(self.detection_folder + str(image_name) + '.jpg', Detectionimage)
        if len(pred_points)>0 and self.visualize:
            # Draw false positive rect
            for fp in pred_points:
                pt1 = tuple([int(fp[0]), int(fp[1])])
                pt2 = tuple([int(fp[2]), int(fp[3])])
                cv2.rectangle(FPimage, pt1, pt2, self.FP_color, 2)
            cv2.imwrite(self.false_positive_folder + str(image_name) + '.jpg', FPimage)
        #  add FP here
        self.FPs += pred_points
        

    def get_result(self):
        if float(len(self.TPs) + len(self.FPs))==0:
            precision=0
        else:
            precision = float(len(self.TPs)) / float(len(self.TPs) + len(self.FPs))
            #print("length of TPs", len(self.TPs), "\n")
            #print("length of FPs", len(self.TPs), "\n")
            #print("length of FNs", len(self.FNs), "\n")
        if float (len(self.TPs) + len(self.FNs))==0:
            recall=0
        else:
            recall = float(len(self.TPs)) /float (len(self.TPs) + len(self.FNs))
        return precision, recall, self.pred_bbox_count

    def reset(self):
        self.TPs = []
        self.FNs = []
        self.FPs = []
        self.total_gt = 0.0

    def get_tpr_fpr(self):
        return len(self.TPs)/self.total_gt,len(self.FPs)/self.total_gt


class Result_writer(object):
    def __init__(self,path):


        self.myfile = open(path, "a")

    def append_detection(self,predection:list,groud_truth:list,image_name):
        self.myfile.write('image {} \n'.format(image_name))
        self.myfile.write('predection# {} \n'.format(len(predection)))
        for i in predection:
            self.myfile.write('{} {} {} {} {} \n'.format(i[0],i[1],i[2],i[3],i[4]))
        self.myfile.write('ground_truth# {} \n'.format(len(groud_truth)))
        for i in groud_truth:
            self.myfile.write('{} {} {} {} \n'.format(i[0],i[1],i[2],i[3]))


class Result_writer_multi_class(object):
    def __init__(self,path):


        self.myfile = open(path, "a")

    def append_detection(self,predection:list,groud_truth:list,image_name):
        self.myfile.write('image {} \n'.format(image_name))
        self.myfile.write('predection# {} \n'.format(len(predection)))
        for i in predection:
            self.myfile.write('{} {} {} {} {} {} \n'.format(i[0],i[1],i[2],i[3],i[4],i[5]))
        self.myfile.write('ground_truth# {} \n'.format(len(groud_truth)))
        for i in groud_truth:
            self.myfile.write('{} {} {} {} {} \n'.format(i[0],i[1],i[2],i[3],i[4]))


class Result_plot(object):
    def __init__(self, path):
        self.precisions = []
        self.recalls = []
        self.pred_lists = []
        self.target_lists = []
        self.score_list = []

        self.file_name =os.path.splitext(os.path.basename(path))[0]
        self.file_dir = os.path.dirname(path)
        with open(path, 'r') as myfile:
            results = myfile.readlines()
        count = 0
        while count < len(results):
            count += 1
            n_predicted = int(results[count].split(' ')[1])
            pred_list=[]
            for i in range(n_predicted):
                count += 1
                bbox = results[count].rstrip().split(' ')
                bbox = np.array([float(i) for i in bbox])
                score =bbox[0]
                if score!=0:
                    self.score_list.append(score)
                pred_list.append(bbox)
            self.pred_lists.append(np.vstack(pred_list))
            count += 1
            n_ground = int(results[count].split(' ')[1])

            target_list=[]
            for i in range(n_ground):
                count += 1
                bbox = results[count].rstrip().split(' ')[0:4]
                bbox = np.array([float(i) for i in bbox])
                target_list.append(bbox)
            self.target_lists.append(target_list)
            count += 1

    def get_result(self):
        threshs = np.arange(0.01, 1.0, 0.05).tolist()

        for thresh in threshs:
            evals = Metric(visualize=False, mode='center', visualization_root='demo/{:0.3f}/'.format(thresh))
            for index in range(len(self.pred_lists)):
                pred_list=self.pred_lists[index]
                pred_list=pred_list[pred_list[:,0]>thresh][:,1:]
                evals.eval_add_result(self.target_lists[index],pred_list)

            precision, recall = evals.get_result()
            F1 = 2*(precision*recall)/(precision+recall)
            F2= 5*(precision*recall)/(4*precision+recall)
            self.precisions.append(precision)
            self.recalls.append(recall)
            out ="precision: {:.4f}  recall:  {:.4f} F1: {:.4f} F2: {:.4f} thresh: {:.4f} TP: {} FP: {} FN: {}"\
                .format(precision,recall,F1,F2,thresh,len(evals.TPs),len(evals.FPs),len(evals.FNs))
            print(out)


    def get_plot(self):

        precisions,recalls = self.get_pr()

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(recalls, precisions)
        # ax.set_yticks(ticks)
        plt.ylim(0.9, 1)
        plt.xlim(0.9, 1)
        ax.set(xlabel='recall', ylabel='precision')
        ax.grid()
        plt.tight_layout()
        plt.savefig(self.file_dir + '/{}.jpg'.format(self.file_name, dpi=fig.dpi, pad_inches=10))
        plt.show()



    def get_pr(self):
        precisions = []
        recalls = []
        threshs = list(set(self.score_list))
        threshs.sort()
        FPR = []
        TPR = []
        for thresh in threshs:
            evals = Metric(visualize=False, mode='center', visualization_root='demo/{:0.3f}/'.format(thresh))
            for index in range(len(self.pred_lists)):
                target = self.target_lists[index]

                pred_list = self.pred_lists[index]
                pred_list = pred_list[pred_list[:, 0] >= thresh][:, 1:]
                evals.eval_add_result(self.target_lists[index], pred_list)

            precision, recall = evals.get_result()
            precisions.append(precision)
            recalls.append(recall)
            tpr, fpr = evals.get_tpr_fpr()
            FPR.append(fpr)
            TPR.append(tpr)

        return precisions, recalls

    def get_result_scale(self):
        threshs = np.arange(0.01, 1.0, 0.05).tolist()

        for thresh in threshs:
            evals = Metric(visualize=False, mode='center', visualization_root='demo/{:0.3f}/'.format(thresh))
            for index in range(len(self.pred_lists)):
                pred_list = self.pred_lists[index]
                pred_list = pred_list[pred_list[:, 0] > thresh][:, 1:]
                evals.eval_add_result(self.target_lists[index], pred_list)

            precision, recall = evals.get_result()
            F1 = 2 * (precision * recall) / (precision + recall)
            F2 = 5 * (precision * recall) / (4 * precision + recall)
            self.precisions.append(precision)
            self.recalls.append(recall)
            out = "precision: {:.4f}  recall:  {:.4f} F1: {:.4f} F2: {:.4f} thresh: {:.4f} TP: {} FP: {} FN: {}" \
                .format(precision, recall, F1, F2, thresh, len(evals.TPs), len(evals.FPs), len(evals.FNs))
            print(out)

            bboxes = np.vstack(evals.FNs)
            width = bboxes[:, 2] - bboxes[:, 0]
            height = bboxes[:, 3] - bboxes[:, 1]

            wd_stack = np.vstack((width, height)).transpose()
            max_bbox_side = np.amax(wd_stack, axis=1)
            unique_number, unique_count = np.unique(max_bbox_side, return_counts=True)
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(2, 1, 1)
            ax.plot(unique_number, unique_count)
            ax.set(xlabel='box size', ylabel='number of boxes')
            ax.grid()
            plt.xticks(np.arange(0, max(unique_number) + 1, 20.0))
            ax.set_title(self.file_name + ' ' + str(thresh))
            # display second plot which shows histergram of plot 1
            num_of_bins = 200
            ax = fig.add_subplot(2, 1, 2)
            # n is the count in each bin, bins is the lower-limit of the bin
            n, bins, patches = ax.hist(max_bbox_side, num_of_bins)
            # We'll color code by height, but you could use any scalar
            fracs = n / n.max()
            # we need to normalize the data to 0..1 for the full range of the colormap
            norm = colors.Normalize(fracs.min(), fracs.max())
            # Now, we'll loop through our objects and set the color of each accordingly
            for thisfrac, thispatch in zip(fracs, patches):
                color = plt.cm.viridis(norm(thisfrac))
                thispatch.set_facecolor(color)
            ax.grid()
            ax.set(xlabel='box size', ylabel='number of boxes')


            plt.tight_layout()

            # plt.savefig(save_location, dpi=fig.dpi, pad_inches=10)
            plt.show()


def plot_best():
    with plt.style.context('fast'):
        fig = plt.figure(figsize=(4.5, 3.75))
        ax = fig.add_subplot(1, 1, 1)
        file_dic = {
            'resnet-50':'/home/dwang/data0/SynologyDrive/ssd.pytorch-master/aws_result/resnet_50_wrong_gaussian/95.txt',
            'resnet-101': '/home/dwang/data0/SynologyDrive/ssd.pytorch-master/aws_result/resnet_101_wrong_gaussian/25.txt',
            'SSD_baseline':'/home/dwang/data0/SynologyDrive/ssd.pytorch-master/aws_result/SSD_baseline/65.txt',
            'Ours:VGG16':'/home/dwang/data0/SynologyDrive/ssd.pytorch-master/aws_result/VGG16_COSINE_GASSUIAN_NO_FPN_FEM/75.txt',
            'Ours:VGG16+FPN':'/home/dwang/data0/SynologyDrive/ssd.pytorch-master/aws_result/VGG16_FPN_COSINE_GASSUIAN_NO_FEM/55.txt',
            'Ours:VGG16+FPN+CEM': '/home/dwang/data0/SynologyDrive/ssd.pytorch-master/aws_result/0.txt'
        }

        for k in file_dic.keys():
            r = Result_plot(file_dic[k])

            precisions, recalls = r.get_pr()

            ax.plot(recalls, precisions)

            precisions, recalls = np.array(precisions),np.array(recalls)

            # sorted_index = precisions.argsort()
            # precisions = precisions[sorted_index]
            # recalls = recalls[sorted_index]
            # xnew = np.linspace(precisions.min(), precisions.max(), 1000)  # 300 represents number of points to make between T.min and T.max
            #
            # power_smooth = spline(precisions, recalls, xnew)
            # plt.plot(xnew, power_smooth)

        plt.legend(file_dic.keys(), ncol=1, loc='lower left');
        # ax.set_yticks(ticks)
        plt.ylim(0.9, 1)
        plt.xlim(0.9, 1)
        ax.set(xlabel='recall', ylabel='precision')
        ax.grid()
        plt.tight_layout()
        plt.savefig('aws_result' + '/{}.pdf'.format('best', dpi=fig.dpi, pad_inches=10))
        plt.show()

if __name__ == '__main__':
    # for i in range(10,80,5):
    #     print('epoc:{}'.format(i))
    #     r =Result_plot('aws_result/correct_gaussian_alpha_1/{}.txt'.format(i))
    #     r.get_result()
    #     r.get_plot()
    # i=0
    # print('epoc:{}'.format(i))
    # r = Result_plot('demo/{}.txt'.format(i))
    # r.get_result()
    # r.get_plot()

    # r = Result_plot('aws_result/correct_gaussian_alpha_1/{}.txt'.format(65))
    # r.get_result_scale()





    plot_best()



# def pf():
# dataset = Polyp('test')
# result_writer = Result_writer('aws_result/centerNet-104/{}.txt'.format('converted'))
# eva= Matrix()
# with open('aws_result/centerNet-104/results.json') as json_file:
#     bbox = json.load(json_file)
#
# for i in range(len(dataset)):
#
#     anno =dataset.annos[i]
#
#     file_name =os.path.splitext(os.path.basename(anno['filename']))[0]
#
#     gt_box=anno["gt_bboxes"]
#     preds = []
#     preds.append(np.array([0,0,0,0,0]))
#     for bbox_item in bbox:
#         if bbox_item['image_id'] == int(file_name):
#             bbox_item['bbox'][2] += bbox_item['bbox'][0]
#             bbox_item['bbox'][3] += bbox_item['bbox'][1]
#
#             # bbox_item['bbox'][0]*=384/288
#             # bbox_item['bbox'][1]*=288/288
#             # bbox_item['bbox'][2] *= 384/288
#             # bbox_item['bbox'][3] *= 288 / 288
#
#
#             bbox_item['bbox'].append(bbox_item['score'])
#
#             box =np.array([bbox_item['score'],bbox_item['bbox'][0],bbox_item['bbox'][1],bbox_item['bbox'][2],bbox_item['bbox'][3]])
#             preds.append(box)
#
#
#     result_writer.append_detection(preds,gt_box,file_name)
#
# result_writer.myfile.close()
# r = Result_plot('aws_result/centerNet-104/{}.txt'.format('converted'))
# r.get_result()
#
#
#






