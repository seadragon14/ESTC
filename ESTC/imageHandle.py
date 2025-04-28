import math
import time
import cv2
import os
import torch
import numpy as np
from pathlib import Path


# is a file
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}  # image suffixes
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def preTransform(image,new_shape=(640, 640),stride=32,isCenter=True):
    """对每张图片进行处理，包括下采样以及padding，图像画边框
        Args:
            image:单张需要处理的图像
            new_shape:该图片要下采样的尺寸
            isCenter:padding是四周还是单侧
    """
    img_shape=image.shape[:2]
    ratio=min(new_shape[0]/img_shape[0],new_shape[1]/img_shape[1])
    new_unpad = int(round(img_shape[1] * ratio)), int(round(img_shape[0] * ratio))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    if isCenter:
        dw /= 2  # divide padding into 2 sides
        dh /= 2
    if img_shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)) if isCenter else 0, int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)) if isCenter else 0, int(round(dw + 0.1))

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border
    return image

def preProcess(path,half=False):
    """模拟v8进行预处理，包括判断是图片文件还是图片路径，对其进行相应的处理
    Args:
        path:图像路径
        half:将张量设置为全精度或者是半精度，模型全精度
    """
    img_List=[]
    fileNameList=[]
    is_file = Path(path).suffix[1:] in (IMG_FORMATS)
    if is_file:
        img_List.append(cv2.imread(path))
        fileNameList.append(path.split('\\')[-1])
    else:
        img_paths =[]
        for filename in os.listdir(path):
            img_paths.append(os.path.join(path,filename))
            fileNameList.append(filename)
        for file_pth in img_paths:
            img_List.append(cv2.imread(file_pth))

    not_tensor = not isinstance(img_List, torch.Tensor)
    if not_tensor:
        im = np.stack([preTransform(img) for img in img_List])
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous 返回连续数组
        im = torch.from_numpy(im)
    im = im.to(DEVICE)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return img_List,im,fileNameList

def map_non_zero_values(heatmap1, heatmap2):
    # 确保两个 heatmap 的形状相同
    assert heatmap1.shape == heatmap2.shape, "两个 heatmap 的形状必须相同"

    # 创建一个布尔掩码，标记 heatmap1 中的非零值位置
    mask = heatmap1 != 0

    # 将 heatmap1 中的非零值映射到 heatmap2 的对应位置
    result = heatmap2.copy()
    result[mask] = heatmap1[mask]

    return result

def preProcessForImage(image,half=False):

    not_tensor = not isinstance(image, torch.Tensor)
    if not_tensor:
        im = np.stack([preTransform(image)])
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous 返回连续数组
        im = torch.from_numpy(im)
    im = im.to(DEVICE)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im

def handleVisualize(heatmap,x1,y1,x2,y2,weight):
    height=y2-y1
    width=x2-x1

    y_indices = np.arange(height)
    x_indices = np.arange(width)
    # 计算行和列的中心位置
    center_y = height // 2
    center_x = width // 2

    # 计算行和列的距离矩阵
    y_dist = np.abs(y_indices - center_y)
    x_dist = np.abs(x_indices - center_x)

    # 创建二维的距离矩阵
    y_dist_matrix = np.tile(y_dist, (width, 1)).T
    x_dist_matrix = np.tile(x_dist, (height, 1))

    # 计算总距离矩阵
    total_dist_matrix = np.sqrt(y_dist_matrix ** 2 + x_dist_matrix ** 2)

    # 计算最大距离
    max_dist = np.sqrt(center_y ** 2 + center_x ** 2)

    # 计算权重矩阵，使得中间权重最大，向四周递减
    weight_matrix = 1 - (total_dist_matrix / max_dist)

    # 应用权重到 heatmap 区域
    heatmap[y1:y2, x1:x2] = weight_matrix * weight

    return heatmap

def readImgFormPath(path):
    img_List=[]
    fileNameList=[]
    is_file = Path(path).suffix[1:] in (IMG_FORMATS)
    if is_file:
        img_List.append(cv2.imread(path))
        fileNameList.append(path.split('\\')[-1])
    else:
        img_paths =[]
        for filename in os.listdir(path):
            img_paths.append(os.path.join(path,filename))
            fileNameList.append(filename)
        for file_pth in img_paths:
            img_List.append(cv2.imread(file_pth))
    return img_List

def compensateUp(image,Xcut1,Xcut2,Ycut1,Ycut2):

    # 步骤1：提取非白色像素点
    line_points = []
    lastPoint = -1  # 用于捕捉非白色像素点间距离的关系
    enhance_points = []
    # 循环遍历截断处像素
    count=0 #  用于判断连续的非白色像素个数，以此来判断厚度属于波峰还是其他
    exist=False
    for x in range(Xcut1, Xcut2 + 1):
        color = image[Ycut2, x]
        if (color != [255, 255, 255]).any():
            count+=1
            # 设置边界条件,捕捉中间像素浅两边像素深的特征进行加强
            if (x - 1 >= Xcut1) and (x + 1 <= Xcut2):
                preColor = image[Ycut2, x - 1][0]
                afterColor = image[Ycut2, x + 1][0]
                if (preColor < color[0] and color[0] > afterColor):
                    enhance_points.append((x - 1, Ycut2, [180, 119, 34]))
                    enhance_points.append((x + 1, Ycut2, [180, 119, 34]))

            # 捕捉非白色像素的距离特征
            if lastPoint == -1:
                lastPoint = x
                continue
            # 如果非白色像素不相邻视为两个交点
            if math.fabs(lastPoint - x) > 1 and math.fabs(lastPoint - x) < 20:
                # 防止两个波峰距离很近被连起来
                if len(line_points)!=0 and math.fabs(line_points[-1][0]-lastPoint)<4:continue
                # 错误将两个波峰尖端识别成断丝两部分（特征在于厚）
                if  exist:continue
                line_points.append((lastPoint - 1, Ycut2, [180, 119, 34]))
                line_points.append((x + 1, Ycut2, [180, 119, 34]))
                count = 0
            lastPoint = x
            if count>6: exist=True


    # 步骤2：配对
    # print(line_points)
    if len(line_points) % 2 != 0:
        line_points = line_points[:-1]

    pairs = [(line_points[i], line_points[i + 1]) for i in range(0, len(line_points), 2)]
    enhance_pairs = [(enhance_points[i], enhance_points[i + 1]) for i in range(0, len(enhance_points), 2)]

    # 步骤3：绘制
    cv2.rectangle(image, (Xcut1, Ycut1), (Xcut2, Ycut2), (255, 255, 255), -1)

    for pt1, pt2 in pairs:
        x1, y1, color1 = pt1
        x2, y2, color2 = pt2
        cv2.line(image, (x1, y1), (x2, y2), color1, 2, cv2.LINE_AA)

    for pt1, pt2 in enhance_pairs:
        x1, y1, color1 = pt1
        x2, y2, color2 = pt2
        cv2.line(image, (x1, y1), (x2, y2), color1, 2, cv2.LINE_AA)
    return image

def compensateDown(orig_img,Xcut1,Xcut2,Ycut1,Ycut2):
    # 步骤1：提取非白色像素点
    line_points = []
    lastPoint = -1  # 用于捕捉非白色像素点间距离的关系
    enhance_points = []
    # 循环遍历截断处像素
    for x in range(Xcut1, Xcut2 + 1):
        color = orig_img[Ycut1, x]
        if (color != [255, 255, 255]).any():
            # 设置边界条件,捕捉中间像素浅两边像素深的特征进行加强
            if (x - 1 >= Xcut1) and (x + 1 <= Xcut2):
                preColor = orig_img[Ycut1, x - 1][0]
                afterColor = orig_img[Ycut1, x + 1][0]
                if (preColor < color[0] and color[0] > afterColor):
                    enhance_points.append((x - 1, Ycut1, [180, 119, 34]))
                    enhance_points.append((x + 1, Ycut1, [180, 119, 34]))

            # 捕捉非白色像素的距离特征
            if lastPoint == -1:
                lastPoint = x
                continue
            # 如果非白色像素不相邻视为两个交点
            if math.fabs(lastPoint - x) > 1 and math.fabs(lastPoint - x) < 20:
                line_points.append((lastPoint - 1, Ycut1, [180, 119, 34]))
                line_points.append((x + 1, Ycut1, [180, 119, 34]))
            lastPoint = x

    # 步骤2：配对
    # print(line_points)
    if len(line_points) % 2 != 0:
        line_points = line_points[:-1]

    pairs = [(line_points[i], line_points[i + 1]) for i in range(0, len(line_points), 2)]
    enhance_pairs = [(enhance_points[i], enhance_points[i + 1]) for i in range(0, len(enhance_points), 2)]

    # 步骤3：绘制
    image = orig_img.copy()
    cv2.rectangle(image, (Xcut1, Ycut1), (Xcut2, Ycut2), (255, 255, 255), -1)

    for pt1, pt2 in pairs:
        x1, y1, color1 = pt1
        x2, y2, color2 = pt2
        cv2.line(image, (x1, y1), (x2, y2), color1, 2, cv2.LINE_AA)

    for pt1, pt2 in enhance_pairs:
        x1, y1, color1 = pt1
        x2, y2, color2 = pt2
        cv2.line(image, (x1, y1), (x2, y2), color1, 2, cv2.LINE_AA)
    return image

def cut(Hcut,orig_img, img_name,x1,x2,edge_Up,edge_Down,Ystart,Yend):
    """
    Args:
        Hcut: 截断块高
        orig_img: 原始图像
        img_name: 图像名称
        x1: 检测框左上角横坐标
        x2: 检测框右下角横坐标
        edge_Up: 中心区域上边界
        edge_Down: 中心区域下边界
        Ystart: 检测框起始y坐标
        Yend: 检测框结束y坐标
    Returns:

    """
    Result=[]
    Xcut1=int(x1)
    Xcut2=int(x2)
    Ycut1=int(Ystart)
    Ycut2=int(Hcut)+Ycut1

    # 对boudingbox上下两端额外区域进行清空，防止模型标注不准确残留像素带来的影响
    cv2.rectangle(orig_img, (Xcut1, Ystart-20), (Xcut2, Ystart), (255, 255, 255), -1)
    cv2.rectangle(orig_img, (Xcut1, Yend), (Xcut2, Yend+20), (255, 255, 255), -1)

    # 故障信号上半区域
    while Ycut2<edge_Up:
        tempImg1=orig_img.copy()
        image=compensateUp(tempImg1, Xcut1, Xcut2, Ycut1, Ycut2)
        save_name=r"F:\ultralytics-main\cut\\"+img_name+"-"+str(Ycut2)+".png"
        Ycut2=Ycut2+int(Hcut)
        # cv2.imwrite(save_name,image)
        Result.append([image,Xcut1,Ycut1,Xcut2,Ycut2])


    # 更换为故障信号下班区域
    Ycut2=int(Yend)
    Ycut1=Ycut2-int(Hcut)

    while Ycut1>edge_Down:
        tempImg2=orig_img.copy()
        image=compensateDown(tempImg2,Xcut1,Xcut2,Ycut1,Ycut2)
        save_name = r"F:\ultralytics-main\cut\\" + img_name + "-" + str(Ycut1) + ".png"
        Ycut1=Ycut1-int(Hcut)
        # cv2.imwrite(save_name, image)
        Result.append([image,Xcut1,Ycut1,Xcut2,Ycut2])

    return Result

def removeSplicing(Hremove, orig_img, img_name, edge_Up, edge_Center, edge_Down,x1,x2,Ystart,Yend):
    """
    用于剔除故障信号中的指定部分
    Args:
        orig_img: 原始图像
        img_name: 图像名称
        x1: 剔除块左上角横坐标
        y1:
        x2: 剔除块右下角横坐标
        y2:
        edgeCenter: 中心区域中心线
        Ystart: 检测框起始位置
        Yend: 检测框结束位置

    Returns:

    """
    Result=[]
    # 计算剔除框的坐标
    y1=edge_Up
    y2=y1+Hremove

    # 计算移动距离
    dy = Hremove

    while y2<=edge_Center:
        image = orig_img.copy()
        # 填充检测框为白色
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # 将整个信号区域由上部分拼接到剩余部分
        movey1 = Ystart
        movey2 = y1

        region = image[movey1:movey2, x1:x2]
        new_image = np.zeros_like(image)
        # 这里多移动1是因为剩了点空隙，和下面不同是因为下面的不是剩而是自带的
        new_movey1 = movey1 + dy+1
        new_movey2 = movey2 + dy+1
        new_image[:] = image[:]
        new_image[new_movey1:new_movey2, x1:x2] = region[:new_movey2 - new_movey1, :]
        # 清空复制过来的原始图像中保留的像素(-20是防止预测框不准，剩了些像素)
        new_image[movey1-20:new_movey1, x1:x2] = 255
        image = new_image
        save_name = r"F:\ultralytics-main\remov\\" + img_name + "-" + str(y2) + ".png"
        # cv2.imwrite(save_name, image)
        res=[image,x1,y1,x2,y2]
        Result.append(res)
        y1=y2
        y2=y2+Hremove

    y2 = edge_Down
    y1 = y2 - Hremove

    while y1 >= edge_Center:

        image1 = orig_img.copy()
        # 填充检测框为白色
        cv2.rectangle(image1, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # 将整个信号区域由下部分拼接到剩余半部分
        movey1 = y2
        movey2 = Yend

        # 此处+1是因为这块该移动的区域顶层包含一层纯白像素
        region = image1[movey1+1:movey2+1, x1:x2]
        new_movey1 = movey1 - dy
        new_movey2 = movey2 - dy

        new_image = np.zeros_like(image1)
        new_image[:] = image1[:]

        new_image[new_movey1:new_movey2, x1:x2] = region[:new_movey2 - new_movey1, :]

        new_image[new_movey2:movey2+20, x1:x2] = 255
        image1 = new_image

        save_name = r"F:\ultralytics-main\remov\\" + img_name + "-" + str(y1) + ".png"
        # cv2.imwrite(save_name, image1)
        res=[image1,x1,y1,x2,y2]
        Result.append(res)
        y2=y1
        y1=y1-Hremove

    return Result


def getBoundingBoxInfo(boundingbox):
    # 计算boundingbox的大小和一些边界
    x1, y1, x2, y2 = boundingbox[:4]
    Wsignal = math.fabs(x2 - x1)
    Hsignal = math.fabs(y2 - y1)
    Hcenter = Hsignal * 0.6
    edge_Up = y1 + Hsignal * 0.2
    edge_Down = y2 - Hsignal * 0.2
    edge_Center = y1 + Hsignal * 0.5
    return x1,y1,x2,y2,Wsignal,Hsignal,Hcenter,edge_Up,edge_Center,edge_Down

def nms_rotated(boxes, scores, threshold=0.45):
    """
    NMS for oriented bounding boxes using probiou and fast-nms.

    Args:
        boxes (torch.Tensor): Rotated bounding boxes, shape (N, 5), format xywhr.
        scores (torch.Tensor): Confidence scores, shape (N,).
        threshold (float, optional): IoU threshold. Defaults to 0.45.

    Returns:
        (torch.Tensor): Indices of boxes to keep after NMS.
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes).triu_(diagonal=1)
    pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
    return sorted_idx[pick]

def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd

def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    import torchvision  # scope for faster 'import ultralytics'

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
        boxes (torch.Tensor): the bounding boxes to clip
        shape (tuple): the shape of the image

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped boxes
    """
    if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
    specified in (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        xywh (bool): The box format is xywh or not, default=False.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)