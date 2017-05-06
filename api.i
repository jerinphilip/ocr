%module ocr

%{
#include <iostream>
#include <string> 
#include <list> 
#include <fstream> 
#include <map>
#include <time.h>
#include <vector>
#include <limits>
#include <stdio.h>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/framework/LocalFileInputSource.hpp>
#include <xercesc/dom/DOMException.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/util/NameIdPool.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/XMLValidator.hpp>
#include <xercesc/validators/schema/SchemaValidator.hpp>
#include <xercesc/validators/common/ContentSpecNode.hpp>
#include <xercesc/validators/schema/SchemaSymbols.hpp>
#include <xercesc/util/OutOfMemoryException.hpp>
#include "nnl/NetMaker.h"
#include "nnl/NDimNet.h"
#include "nnl/GradientFollowerMaker.h"
#include "nnl/DataExportHandler.h"
#include "nnl/GradientTest.h"
#include "nnl/Random.h"
#include "nnl/Typedefs.h"
#include "nnl/TranscriptionOutputLayer.h"
#include <stdlib.h>
#include "api.h"
using namespace std;

%}

/*
%include "std_vector.i"
%include "std_string.i"
*/
%include "stl.i"
namespace std
{
    %template(FloatVector) vector<float>;
    %template(StringVector) vector<string>;
}

%include "api.h"

