import pyocr
import numpy as np
import cv2

def convert(pyocr_output):
    codepoint = pyocr_output[1:]
    codepoint_value = int(codepoint, 16)
    return chr(codepoint_value)

def display_img(inputs):
    shape = (32, len(inputs)//32)
    shape = tuple(reversed(shape))
    img = 255*np.array(inputs, dtype=np.uint8).reshape(shape).T
    cv2.imshow("img", img)
    cv2.waitKey(0)


inputs = list(map(float, open("../etc/input.txt").read().strip().split()))

#print("Inputs:", inputs)
input_fv = pyocr.FloatVector(len(inputs))
for i in range(len(inputs)):
    input_fv[i] = inputs[i]

# OCR API
net = pyocr.NetAPI(
        "../etc/cvit_ocr_weights.xml",  # Weights file
        "../etc/lookup.txt") # Lookup file
recognized = net.recognize(input_fv)
print("Recognized:", ''.join(map(convert, recognized)))

