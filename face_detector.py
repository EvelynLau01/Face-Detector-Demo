import numpy as np
import cv2

min_confidence = 0.2 #threshold for face detection

#Load the model from disk, we're using pretrained resnet neutral network
prototxt_path = ".\Face_detection_model\deploy.prototxt"
model_path = ".\Face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    #pass the blob through the network and obtain the output classifications
    net.setInput(blob)
    detections = net.forward()#pass the blob through the whole net

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > min_confidence:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == ord("q"):  # accepts a string of length 1 as an argument and
        # returns the unicode code point representation of the passed argument.
        break

cap.release()
cv2.destroyAllWindows()