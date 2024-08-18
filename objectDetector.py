import cv2
import numpy as np
import sys

def load_yolo():
    net = cv2.dnn.readNet('model/yolov3.weights', 'model/yolov3.cfg')
    with open("model/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers


def detect_objects(img, net, output_layers):
    height, width, channels = img.shape

    # Preparar la imagen para el modelo YOLO
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Procesar las salidas de detección
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return class_ids, confidences, boxes

def get_most_confident_class(class_ids, confidences, classes):
    if len(class_ids) > 0:
        # Seleccionar la clase con la confianza más alta
        max_conf_idx = np.argmax(confidences)
        return classes[class_ids[max_conf_idx]]
    else:
        return None

def main(image_path):
    net, classes, output_layers = load_yolo()
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: The image could not be loaded.")
        sys.exit(1)

    class_ids, confidences, boxes = detect_objects(img, net, output_layers)
    most_confident_class = get_most_confident_class(class_ids, confidences, classes)
    
    if most_confident_class:
        print(f"The detected object is a {most_confident_class}")
    else:
        print("No object was detected with sufficient confidence.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python objectDetector.py <image path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    main(image_path)
