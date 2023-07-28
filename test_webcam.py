from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import cv2




model = load_model("./rock_paper_scissors_cnn_inception.h5")

from tensorflow.keras.utils import load_img, img_to_array
import numpy as np


def label(output):
    labels = ['paper', 'rock', 'scissors']
    result = np.argmax(output[0])
    
    return labels[result]


# img = cv2.resize(cv2.imread("./test_3.jpg"), (300, 200))
# # img = load_img("./rps-cv-images/train/rock/00nKV8oHuTGi20gq.png")
# x = img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x /= 255
# result = model.predict(x)
# print(result)
# print(label(result))

# plt.imshow(img)
# plt.show()
## define a video capture object
vid = cv2.VideoCapture(0)
idx=0
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frame = cv2.resize(frame, (150,100))
    x = img_to_array(frame)
    x = np.expand_dims(x, axis=0)
    x /= 255
    # print(f"frame: {frame.shape}, x: {x.shape}")
    result = model.predict(x)
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (40,60)
    fontScale              = 1
    fontColor              = (255,0,0)
    thickness              = 1
    lineType               = 2


    cv2.putText(frame,f"{label(result)}", 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    cv2.imwrite(f"./test/{idx}.png", frame)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    idx +=1
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the window