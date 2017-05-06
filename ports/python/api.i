%module pyocr

%{
#include "ocr/api.h"
using namespace std;

%}

%include "stl.i"
namespace std
{
    %template(FloatVector) vector<float>;
    %template(StringVector) vector<string>;
}

%include "ocr/api.h"
