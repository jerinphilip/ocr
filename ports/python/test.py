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


sequences = pyocr.FloatVVector(1)
for i in range(1):
    sequences[i] = input_fv;

ls = [84, 38, 12]
truths = pyocr.IntVVector(3)
for i in range(1):
    labels = pyocr.IntVector(3)
    for j in range(3):
        labels[i] = ls[i]
    truths[i] = labels

# OCR API
net = pyocr.NetAPI(
        "../etc/cvit_ocr_weights.xml",  # Weights file
        "../etc/lookup.txt") # Lookup file

errors = net.train(sequences, truths)
for i in range(len(errors)):
    print("Error %d: %lf"%(i+1, errors[i]))
exit()
recognized = net.recognize(input_fv)
print("Recognized:", ''.join(map(convert, recognized)))

