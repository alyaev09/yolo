import cv2
import numpy as np


def draw_object_bounding_box(image_to_process, index, box):

    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    if classes[index]  == "sandwich": 
    	color = (255, 0, 0)
    elif classes[index]  == "pizza":
    	color = (0, 255, 0)
    else:
    	color = (0, 0, 255)	
    width = 1
    final_image = cv2.rectangle(image_to_process, start, end, color, width)
    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    final_image = cv2.putText(final_image, text, start, font, font_size, color, width, cv2.LINE_AA)

    return final_image

def yolo_object_detection(image_to_process):

    height, width, depth = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)

    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0

    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0.3:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)

                box = [center_x - obj_width // 2, center_y - obj_height // 2, obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.3, 0.4)

    for box_index in chosen_boxes:
        box = boxes[box_index]
        class_index = class_indexes[box_index]

        if classes[class_index] in classes_to_look_for:
            final_image = draw_object_bounding_box(image_to_process, class_index, box)

    return final_image


def start_image_object_detection():

    try:
        image = cv2.imread("assets/sandwich.jpg")
        image1 = cv2.imread("assets/pizza.jpeg")
        image2 = cv2.imread("assets/cakes.webp")
        image = yolo_object_detection(image)

        cv2.imshow("Image", image)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    cfgModel = "yolov4-tiny.cfg";
    weightsModel = "yolov4-tiny.weights";
    net = cv2.dnn.readNetFromDarknet(cfgModel, weightsModel)
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    with open("coco.names.txt") as file:
        classes = file.read().split("\n")

    classes_to_look_for = ["sandwich", "pizza", "cake"]

    start_image_object_detection()

