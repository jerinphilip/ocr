import ocr

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
print("Recognized:", recognized)
