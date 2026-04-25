import cv2
from api_infer import *
import numpy as np

classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
           "fire", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
           "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
           "hair drier", "toothbrush"]

def resize_image_pad(srcimg, keep_ratio, input_width, input_height):
    top, left, newh, neww = 0, 0, input_height, input_width
    if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
        hw_scale = srcimg.shape[0] / srcimg.shape[1]
        if hw_scale > 1:
            newh, neww = input_height, int(input_width / hw_scale)
            img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
            left = int((input_width - neww) * 0.5)
            img = cv2.copyMakeBorder(img, 0, 0, left, input_width - neww - left, cv2.BORDER_CONSTANT,
                                     value=(114, 114, 114))  # add border
        else:
            newh, neww = int(input_height * hw_scale), input_width
            img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
            top = int((input_height - newh) * 0.5)
            img = cv2.copyMakeBorder(img, top, input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                     value=(114, 114, 114))
    else:
        img = cv2.resize(srcimg, (input_width, input_height), interpolation=cv2.INTER_AREA)
    return img, newh, neww, top, left

def preprocess_yolo_det(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, newh, neww, padh, padw = resize_image_pad(rgb_img, keep_ratio=True, input_width=640, input_height=640)
    img = img.astype(np.float32) / 255.0
    return img, newh, neww, padh, padw

def make_grid(nx=20, ny=20):
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)


def drawPred(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=1)

    label = '%.2f' % conf
    label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
    return frame

def yolov5_anchor_decode(outs, inpHeight=640, inpWidth=640):
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    stride = np.array([8., 16., 32.])
    nl = len(anchors)
    na = len(anchors[0]) // 2
    grid = [np.zeros(1)] * nl
    anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, -1, 2)
    row_ind = 0

    for i in range(nl):
        h, w = int(inpHeight / stride[i]), int(inpWidth / stride[i])
        length = int(na * h * w)
        if grid[i].shape[2:4] != (h, w):
            grid[i] = make_grid(w, h)

        outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
            grid[i], (na, 1))) * int(stride[i])
        outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
            anchor_grid[i], h * w, axis=0)
        row_ind += length

def postprocess_yolov5_det(img, outs, newh, neww, padh, padw, anchor_decode=True):
    output = outs['output']
    N = len(output) // 85
    output = np.array(output).reshape(N, 85)
    # # print(output.shape)
    # # print(output.size)
    
    confThreshold = 0.25
    nmsThreshold = 0.45
    objThreshold = 0.3
    if anchor_decode:
        yolov5_anchor_decode(output)

    frameHeight = img.shape[0]
    frameWidth = img.shape[1]
    ratioh, ratiow = frameHeight / newh, frameWidth / neww
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.

    confidences = []
    boxes = []
    classIds = []
    for detection in output:
        if detection[4] > objThreshold:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId] * detection[4]
            if confidence > confThreshold:
                center_x = int((detection[0] - padw) * ratiow)
                center_y = int((detection[1] - padh) * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width * 0.5)
                top = int(center_y - height * 0.5)

                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                classIds.append(classId)
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    print("before nms, confidences num:", len(confidences))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    print("after nms, indices num:", len(indices))
    if len(indices) > 0:
        indices = np.array(indices).flatten()
    else:
        indices = []
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        img = drawPred(img, classIds[i], confidences[i], left, top, left + width, top + height)
        print("class:{}({})".format(classes[classIds[i]], confidences[i]))
        print("left:{}, top:{}, width:{}, height:{}".format(left, top, width, height))
    return img


def postprocess_yolov8_det(img, outs, newh, neww, padh, padw):
    out_boxes = outs['boxes']
    N = len(out_boxes) // 4
    out_boxes = np.array(out_boxes).reshape(N, 4)
    out_scores = np.array(outs['scores'])
    out_ids = np.array(outs['class_idx'])

    conf_threshold = 0.25
    nms_threshold = 0.45
    img_height, img_width = img.shape[:2]
    ratioh = img_height / newh
    ratiow = img_width / neww
    rows = out_boxes.shape[0]

    boxes = []
    scores = []
    class_ids = []
    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = out_scores[i]
        class_id = int(out_ids[i])
        if classes_scores >= conf_threshold:
            box = [
                int((out_boxes[i][0] - padw) * ratiow),
                int((out_boxes[i][1] - padh) * ratioh),
                (out_boxes[i][2] - padw) * ratiow - (out_boxes[i][0] - padw) * ratiow,
                (out_boxes[i][3] - padh) * ratioh - (out_boxes[i][1] - padh) * ratioh,
            ]
            boxes.append(box)
            scores.append(classes_scores)
            class_ids.append(class_id)

    # Apply NMS (Non-maximum suppression)
    print("before nms, boxes num:", len(boxes))
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    print("after nms, indices num:", len(indices))
    if len(indices) > 0:
        indices = np.array(indices).flatten()
    else:
        indices = []
    for i in indices:
        box = boxes[i]
        left = int(box[0])
        top = int(box[1])
        width = int(box[2])
        height = int(box[3])
        img = drawPred(img, class_ids[i], scores[i], left, top, left + width, top + height)
        print("class:{}({})".format(classes[class_ids[i]], scores[i]))
        print("left:{}, top:{}, width:{}, height:{}".format(left, top, width, height))
    return img
    

def postprocess_yolov10_det(img, outs, newh, neww, padh, padw):
    output = outs['output0']
    N = len(output) // 6
    output = np.array(output).reshape(N, 6)

    conf_threshold = 0.25
    nms_threshold = 0.45
    img_height, img_width = img.shape[:2]
    ratioh = img_height / newh
    ratiow = img_width / neww
    rows = output.shape[0]

    boxes = []
    scores = []
    class_ids = []
    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = output[i][4]
        class_id = int(output[i][5])
        if classes_scores >= conf_threshold:
            box = [
                int((output[i][0] - padw) * ratiow),
                int((output[i][1] - padh) * ratioh),
                (output[i][2] - padw) * ratiow - (output[i][0] - padw) * ratiow,
                (output[i][3] - padh) * ratioh - (output[i][1] - padh) * ratioh,
            ]
            boxes.append(box)
            scores.append(classes_scores)
            class_ids.append(class_id)

    # Apply NMS (Non-maximum suppression)
    print("before nms, boxes num:", len(boxes))
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    print("after nms, indices num:", len(indices))
    if len(indices) > 0:
        indices = np.array(indices).flatten()
    else:
        indices = []
    for i in indices:
        box = boxes[i]
        left = int(box[0])
        top = int(box[1])
        width = int(box[2])
        height = int(box[3])
        img = drawPred(img, class_ids[i], scores[i], left, top, left + width, top + height)
        print("class:{}({})".format(classes[class_ids[i]], scores[i]))
        print("left:{}, top:{}, width:{}, height:{}".format(left, top, width, height))
    return img


def postprocess_yolov10_modified_det(img, outs, newh, neww, padh, padw):
    # out_boxes = outs['box']
    out_boxes = outs['/model.23/Mul_output_0']
    N = len(out_boxes) // 8400
    out_boxes = np.array(out_boxes).reshape(N, 8400)
    # out_scores = outs['score']
    out_scores = outs['/model.23/Sigmoid_output_0']
    N = len(out_scores) // 8400
    out_scores = np.array(out_scores).reshape(N, 8400)

    conf_threshold = 0.25
    nms_threshold = 0.45
    img_height, img_width = img.shape[:2]
    ratioh = img_height / newh
    ratiow = img_width / neww
    rows = out_boxes.shape[1]

    boxes = []
    scores = []
    class_ids = []
    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = out_scores[:, i]
        score = np.max(classes_scores)
        class_id = np.argmax(classes_scores)
        if score >= conf_threshold:
            box = out_boxes[:, i]  # shape: (4,)
            box_xywh = [
                int((box[0] - padw) * ratiow),
                int((box[1] - padh) * ratioh),
                (box[2] - padw) * ratiow - (box[0] - padw) * ratiow,
                (box[3] - padh) * ratioh - (box[1] - padh) * ratioh,
            ]
            boxes.append(box_xywh)
            scores.append(score)
            class_ids.append(class_id)

    # Apply NMS (Non-maximum suppression)
    print("before nms, boxes num:", len(boxes))
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    print("after nms, boxes num:", len(indices))
    if len(indices) > 0:
        indices = np.array(indices).flatten()
    else:
        indices = []
    for i in indices:
        box = boxes[i]
        left = int(box[0])
        top = int(box[1])
        width = int(box[2])
        height = int(box[3])
        img = drawPred(img, class_ids[i], scores[i], left, top, left + width, top + height)
        print("class:{}({})".format(classes[class_ids[i]], scores[i]))
        print("left:{}, top:{}, width:{}, height:{}".format(left, top, width, height))
    return img
    


def load_dlc_model(dlc_path, runtime):
    """加载DLC模型"""
    snpe_ort = SnpeContext(dlc_path, [], runtime, PerfProfile.BALANCED, LogLevel.INFO)
    assert snpe_ort.Initialize() == 0
    return snpe_ort


def yolov5_demo():
    image_path = r"test.jpg"
    dlc_path = r"models/yolov5n_sim_quant.dlc"

    runtime = [Runtime.CPU]
    anchor_decode = True

    session = load_dlc_model(dlc_path, runtime)

    # 读取并预处理图像
    img = cv2.imread(image_path)
    input_img, newh, neww, padh, padw = preprocess_yolo_det(img)
    input_img = np.expand_dims(input_img, axis=0)  # 添加batch维度
    input_feed = {"images": input_img}
    output_names = ["output"]
    outputs = session.Execute(output_names, input_feed)
    session.Release()

    # dick
    # print("outputs type:", type(outputs))
    # print("outputs length:", len(outputs))
    # print("outputs keys:", outputs.keys())
    # for key, value in outputs.items():
    #     print("key:{}, length:{}".format(key, len(value)))
    
    # 后处理：解析网络输出，绘制预测结果
    postprocess_yolov5_det(img, outputs, newh, neww, padh, padw, anchor_decode)

    # # 保存结果图像
    cv2.imwrite("result_dlc.jpg", img)


def yolov8_demo():
    image_path = r"test.jpg"
    dlc_path = r"models/yolov8n_sim_quant.dlc"
    runtime = [Runtime.DSP]

    session = load_dlc_model(dlc_path, runtime)

    # 读取并预处理图像
    img = cv2.imread(image_path)
    input_img, newh, neww, padh, padw = preprocess_yolo_det(img)
    input_img = np.expand_dims(input_img, axis=0)  # 添加batch维度

    input_feed = {"image": input_img}
    output_names = ["boxes", "scores", "class_idx"]
    outputs = session.Execute(output_names, input_feed)
    session.Release()

    # dick
    # print("outputs type:", type(outputs))
    # print("outputs length:", len(outputs))
    # print("outputs keys:", outputs.keys())
    # for key, value in outputs.items():
    #     print("key:{}, length:{}".format(key, len(value)))

    postprocess_yolov8_det(img, outputs, newh, neww, padh, padw)

    cv2.imwrite("result_dlc.jpg", img)


def yolov10_demo():
    image_path = r"test.jpg"
    dlc_path = r"models/yolov10n_sim.dlc"
    runtime = [Runtime.CPU]

    session = load_dlc_model(dlc_path, runtime)

    # 读取并预处理图像
    img = cv2.imread(image_path)
    input_img, newh, neww, padh, padw = preprocess_yolo_det(img)
    input_img = np.expand_dims(input_img, axis=0)  # 添加batch维度

    input_feed = {"images": input_img}
    output_names = ["output0"]
    outputs = session.Execute(output_names, input_feed)
    session.Release()

    # dick
    # print("outputs type:", type(outputs))
    # print("outputs length:", len(outputs))
    # print("outputs keys:", outputs.keys())
    # for key, value in outputs.items():
    #     print("key:{}, length:{}".format(key, len(value)))

    postprocess_yolov10_det(img, outputs, newh, neww, padh, padw)

    cv2.imwrite("result_dlc.jpg", img)


def yolov10_modified_demo():
    image_path = r"test.jpg"
    # dlc_path = r"models/modified_yolov10n_quant.dlc"
    dlc_path = r"modified_yolov10n_quantized.dlc"
    runtime = [Runtime.CPU]

    session = load_dlc_model(dlc_path, runtime)

    # 读取并预处理图像
    img = cv2.imread(image_path)
    input_img, newh, neww, padh, padw = preprocess_yolo_det(img)
    input_img = np.expand_dims(input_img, axis=0)  # 添加batch维度

    input_feed = {"images": input_img}
    # output_names = ["score", "box"]
    output_names = ["/model.23/Sigmoid_output_0", "/model.23/Mul_output_0"]
    outputs = session.Execute(output_names, input_feed)
    session.Release()

    postprocess_yolov10_modified_det(img, outputs, newh, neww, padh, padw)

    cv2.imwrite("result_dlc.jpg", img)


if __name__ == "__main__":
    # yolov5_demo()
    # yolov8_demo()
    # yolov10_demo()
    yolov10_modified_demo()


