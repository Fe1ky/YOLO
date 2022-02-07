from turtle import width
import cv2
import numpy as np
# cv2.dnn_Net.readNet

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
calsses = []
with open("coco.names", "r") as f:


    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0,255, size = (len(classes) ,3))


layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


#load image
img = cv2.imread('test4.jpg')
# img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape


#detect image
blob= cv2.dnn.blobFromImage(img, 0.00392, (416,416),(0,0,0),True, crop = False)


net.setInput(blob)
outs = net.forward(output_layers)

#show info on screen 

class_ids=[]
confidences=[]
boxes=[]

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5 :
            #Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2]* width)
            h = int(detection[3]* height)

            #prints a circle in the center of the detected image 
            # cv2.circle(img, (center_x,center_y), 10,(0,255,0), 2)

            #rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)



            

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# number_objects_detected = len(boxes)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]
        label = classes[class_ids[i]]
        color = colors[i]
        print(label)
        cv2.rectangle(img, (x, y), (x + w, y + h),color, 2)
        cv2.putText(img, label, (x+10,y+60),font,5,color,5)

print(len(boxes))


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()