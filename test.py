import ocr
import numpy as np
import cv2

def convert(ocr_output):
    codepoint = ocr_output[1:]
    codepoint_value = int(codepoint, 16)
    return chr(codepoint_value)

def display_img(inputs):
    shape = (32, len(inputs)//32)
    shape = tuple(reversed(shape))
    img = 255*np.array(inputs, dtype=np.uint8).reshape(shape).T
    cv2.imshow("img", img)
    cv2.waitKey(0)


inputs = list(map(float, open("build/input.txt").read().strip().split()))

#print("Inputs:", inputs)
input_fv = ocr.FloatVector(len(inputs))
for i in range(len(inputs)):
    input_fv[i] = inputs[i]

# OCR API
net = ocr.NetAPI(
        "build/cvit_ocr_weights.xml",  # Weights file
        "build/lookup.txt") # Lookup file
recognized = net.recognize(input_fv, int(len(inputs)/32), 32)
print("Recognized:", ''.join(map(convert, recognized)))


