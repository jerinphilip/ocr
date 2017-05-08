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
    %template(FloatVVector) vector<vector<float> >;
    %template(IntVector) vector<int>;
    %template(IntVVector) vector<vector<int> >;
}

%include "ocr/api.h"
