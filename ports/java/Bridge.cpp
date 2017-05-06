#include "Bridge.h"
#include "api.h"
#include <string>
using namespace std;

JNIEXPORT jstring JNICALL Java_Bridge_recognize
  (JNIEnv *env, jobject obj, jfloatArray imageW, jint length){
    // Assuming 32 input size: CVIT Standard.
    NetAPI api("cvit_ocr_weights.xml", "lookup.txt");
    jfloat* image = env->GetFloatArrayElements(imageW, NULL);
    string output = api.recognize(image, length/32, 32);
    jstring joutput = env->NewStringUTF(output.c_str());
    return joutput;
}
