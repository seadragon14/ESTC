import math
import sys
import warnings
import cv2
import numpy as np
import torch
from tqdm import tqdm

from imageHandle import preProcess, non_max_suppression, scale_boxes, removeSplicing, getBoundingBoxInfo, cut, \
    preProcessForImage, readImgFormPath, handleVisualize, map_non_zero_values


# 1.通过YOLO模型预测得出图像中的故障信号的位置信息（注意位置信息有多个的情况）
def getImgLoc(img_path,YOLOv8):
    orig_imgs,pre_imgs,img_names=preProcess(img_path,True)

    pred_imgs=YOLOv8(pre_imgs)

    # 进行非极大值抑制
    pred_results = non_max_suppression(pred_imgs,conf_thres=0.5,iou_thres=0.7,classes=None,max_det=300,agnostic=False)

    imgLoc=[]

    # 将预测得到的结果缩放到原始图像尺寸（original,preprocess,pred）
    for pred, orig_img, img_name in zip(pred_results, orig_imgs, img_names):
        img_loc=[]
        pred[:, :4] = scale_boxes(pre_imgs.shape[2:], pred[:, :4], orig_img.shape)
        # 转化为数组
        for i in pred:
            img_loc.append(i.cpu().numpy().tolist())
        imgLoc.append(img_loc)

    return orig_imgs,imgLoc,img_names


# 2.通过该位置信息可以设定剔除块和截断块的位置、大小信息,每次扰动后进行拼接，以及进行补偿
def dealBoundingBox(orig_imgs,imgLocs,img_names,removeRate,cutRate):
    if len(imgLocs)==0:
        print("待测图像中未出现断丝")
        sys.exit(0)
    result=[]
    # 针对每个图像进行
    for img,img_name,orig_img in zip(imgLocs,img_names,orig_imgs):

        times=[x for x in range(len(img))]

        raodong=[]
        # 针对每个图像中的boundingbox
        for no,boundingbox in zip(times,img):

            name=img_name.split(".")[0]+"-"+str(no+1)
            # 确定boundingbox的信息
            x1,y1,x2,y2,Wsignal,Hsignal,Hcenter,edge_Up,edge_Center,edge_Down=getBoundingBoxInfo(boundingbox)

            # 设置剔除块的信息,进行剔除拼接
            Hremove=Hcenter*removeRate
            removeResult=removeSplicing(int(Hremove), orig_img, name, int(edge_Up), int(edge_Center), int(edge_Down),
                           int(x1),int(x2), int(y1), int(y2))

            # 进行截断补偿
            Hcut=Hsignal*cutRate
            cutResult=cut(Hcut,orig_img, name,x1,x2,edge_Up,edge_Down,int(y1),int(y2))

            Result=removeResult+cutResult
            raodong.append(Result)
        result.append([raodong,img_name.split(".")[0]])
    return result

# 3.得到该图像中所有故障信号对应的所有区域的权重
def dealPredict(imgInfo,imgLocs):
    result=[]

    for img,orig_info in zip(imgInfo,imgLocs):

        img_name=img[1]

        # 标记值，表示现在进行到了哪个检测框
        count=0
        weight_box=[]
        # 针对每个检测框的不同扰动位置
        for box,orig in zip(img[0],orig_info):
            weights_loc=[]
            for signleBox in box:
                # 获取单张图像
                signalImg=signleBox[0]

                # 对图像进行预处理
                pre_img=preProcessForImage(signalImg,half=True)
                # 进行推理
                pred_img = YOLOv8(pre_img)
                # 进行非极大值抑制
                pred_result = non_max_suppression(pred_img, conf_thres=0.5, iou_thres=0.7, classes=None, max_det=300,agnostic=False)
                # 缩放回原尺寸
                for pred, orig_img in zip(pred_result, [signalImg]):
                    pred[:, :4] = scale_boxes(pre_img.shape[2:], pred[:, :4], orig_img.shape)

                # 扰动后和扰动前的置信分数
                if pred_result[0].shape[0]-count==0:
                    after=0
                else:
                    after=pred_result[0][count][4].item()
                before=orig[4]
                isNegative=0 if (after-before)>=0 else 1
                res=math.fabs(after-before)/before

                weights_loc.append([res,isNegative,signleBox[1:]])
            count+=1
            weight_box.append(weights_loc)
        result.append([weight_box,img_name])
    return result

# 4.通过该权重进行可视化体现出故障信号对模型的影响权重
def visulizeWeight(weights,img_path):

    imgs=readImgFormPath(img_path)

    pbar=tqdm(total=len(imgs))
    for weightinfo,img in zip(weights,imgs):
        pbar.update(1)
        width=img.shape[1]
        height=img.shape[0]
        img_name=weightinfo[1]
        # heatmap = np.zeros((height, width),dtype=np.float32)

        count_boundingbox=len(weightinfo[0])
        heatmapList=[np.zeros((height, width),dtype=np.float32) for x in range(count_boundingbox) ]

        # 针对每个检测框
        for weight_box,heatmap in zip(weightinfo[0],heatmapList):

            # 针对每个检测框中扰动的每个区域
            for weight in weight_box:
                x1,y1,x2,y2=weight[2][:]
                final_weight=weight[0]
                isNegative=weight[1]
                heatmap=handleVisualize(heatmap,x1,y1,x2,y2,final_weight)

        # 改进点：针对每个检测框单独做成热图，最后将多个heatmap拼接

        for heatmap in heatmapList:
            # 对热图进行高斯模糊，让边缘平滑
            blur_kernel_size = (15, 15)  # 可以调整核的大小来控制模糊程度
            heatmap = cv2.GaussianBlur(heatmap, blur_kernel_size, 0)



        fusion=0
        for num in range(count_boundingbox):
            heatmap=heatmapList[num]
            heatmap_normalized_float=cv2.normalize(heatmap,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_32F)
            heatmap_normalized = np.round(heatmap_normalized_float).astype(np.uint8)
            heatmap_colored=cv2.applyColorMap(heatmap_normalized,cv2.COLORMAP_JET)

            if num==0:
                fusion=heatmap_colored
            else:
                fusion=map_non_zero_values(fusion,heatmap_colored)

        # overlay=img.copy()
        # for num in range(count_boundingbox):
        #     heatmap=heatmapList[num]
        #     heatmap_normalized_float=cv2.normalize(heatmap,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        #     heatmap_normalized = np.round(heatmap_normalized_float).astype(np.uint8)
        #     heatmap_colored=cv2.applyColorMap(heatmap_normalized,cv2.COLORMAP_JET)
        #     if num==0:
        #         alpha=0.5
        #         overlay=cv2.addWeighted(overlay,1-alpha,heatmap_colored,alpha,0)
        #     else:
        #         alpha = 0.5-0.1*num
        #         overlay = cv2.addWeighted(overlay, 0.9, heatmap_colored, alpha, 0)


        # 创建颜色条，使其宽度与 overlay 的宽度一致
        overlay = cv2.addWeighted(img, 0.5, fusion, 0.5, 0)
        color_bar_height = 50
        color_bar_width = overlay.shape[1]
        color_bar = np.zeros((color_bar_height, color_bar_width, 3), dtype=np.uint8)
        for i in range(color_bar_width):
            intensity = int(255 * (i / (color_bar_width - 1)))
            color = cv2.applyColorMap(np.array([[intensity]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
            color_bar[:, i] = color

        # 添加标签
        text1 = "Low Weight"
        text2 = "High Weight"
        cv2.putText(color_bar, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(color_bar, text2, (color_bar_width - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 将颜色条添加到叠加图像下方
        combined_image = np.vstack((overlay, color_bar))

        # 保存图像
        cv2.imwrite(r"F:\ultralytics-main\interpretability\NewMethod\version3\\"+img_name+".png", combined_image)
    pbar.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    img_path = r'F:\ultralytics-main\interpretability\NewMethod\readpic'
    # img_path=r'F:\ultralytics-main\onlypric'
    YOLOv8 = torch.load(r'F:\ultralytics-main\correctBest.pt')['model']
    YOLOv8 = YOLOv8.eval().cuda()

    original_imgs,imgLocs,img_names=getImgLoc(img_path,YOLOv8=YOLOv8)

    removeRate=1/30  # 设置剔除块相对于boundingbox的大小
    cutRate=1/50    # 设置截断块相对于boundingbox的大小
    result=dealBoundingBox(original_imgs,imgLocs,img_names,removeRate,cutRate)
    weights=dealPredict(result,imgLocs)
    visulizeWeight(weights,img_path)