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
using namespace std;
//using namespace xercesc;


typedef vector<int> VecInt;
typedef vector<float> VecFloat;
typedef vector<VecInt> VecVecInt;
typedef vector<VecFloat> VecVecFloat;
typedef map<const string, pair<int, double> > errorMapType;


class NetAPI {
    private:
        DataDimensions dimensions;
        DataSequence sequence;
        NDimNet *net;
        TranscriptionOutputLayer *output;
        GradientFollower* gradientFollower;

        const char *weights_file;
        const char *lookup_file;

        DOMElement *root, *netData, *exportData; 
        DOMDocument *document;

        string netExport;

    public:
        NetAPI (const char *weights_f, const char *lookup_f);
        void load_weights_file();
        void initialize_dimensions();
        vector<DataSequence> generateSequences(VecVecFloat &,
                VecVecInt &);
        vector<string> test(VecFloat &);
        VecFloat train(VecVecFloat &, VecVecInt &);
	VecFloat train2(VecVecFloat &, VecVecInt &, int );
        string exportModel();
};
