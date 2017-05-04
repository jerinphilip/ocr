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
using namespace xercesc;

bool sequenceDebugOutput = false;


class NetAPI {
    private:
        DataDimensions dimensions;
        DataSequence sequence;
        NDimNet *net;
        TranscriptionOutputLayer *output;

        const char *weights_file;
        const char *lookup_file;

        DOMElement *root, *netData, *exportData; 
        DOMDocument *document;

    public:
        NetAPI (const char *weights_f, const char *lookup_f) {
            weights_file = weights_f;
            lookup_file = lookup_f;
            initialize_dimensions();
            load_weights_file();
            XMLPlatformUtils::Initialize();
        }

        void load_weights_file(){
            try {
                XMLPlatformUtils::Initialize();
            } catch (const XMLException& e) {
                char *message = XMLString::transcode(e.getMessage());
                cerr<<"error: xerces: "<<endl;
                cerr<< message << endl;
                XMLString::release(&message);
            }


            XercesDOMParser parser;
            parser.setDoNamespaces(true);
            parser.setValidationConstraintFatal(true);
            parser.setExitOnFirstFatalError(false);
            parser.setIncludeIgnorableWhitespace(false);


            /* Try reading from file. */
            try {
                ifstream metadata(weights_file);
                LocalFileInputSource source(XMLString::transcode(weights_file));
                parser.parse(source);
            } catch (const DOMException& e){
                cerr << "Error loading XML\n";
            }


            document = parser.getDocument();
            root = document->getDocumentElement();

            netData = (DOMElement*)(root->
                    getElementsByTagName(XMLString::transcode("Net"))
                    ->item(0));

            exportData = (DOMElement*)(root->
                    getElementsByTagName(XMLString::transcode("ExportedData"))
                    ->item(0));

            vector<string> criteria;
            net = (NDimNet*)NetMaker::makeNet(netData, dimensions, 
                    criteria);
            net->build();
            output = (TranscriptionOutputLayer*)(net->outputLayer);
            DataExportHandler::instance().load(exportData);

        }


        void initialize_dimensions(){
            dimensions.numDims = 1;
            dimensions.inputPattSize = 32;
            dimensions.targetPattSize = 0;
            vector<string> labels;
            string label;
            ifstream lookup(lookup_file);
            while (lookup >> label){
                labels.push_back(label);
            }
            dimensions.labels = labels;
        }

        string recognize(float *inputs, int width, int height){
            sequence.dimensions = {width};
            sequence.size = width;
            sequence.inputs = inputs;

            net->feedForward(sequence);
            vector<int> recognisedIndices;
            string recognized;
            output->getMostProbableString(recognisedIndices);
            for(vector<int>::iterator p=recognisedIndices.begin(); 
                    p!=recognisedIndices.end(); p++){

                recognized += dimensions.labels[*p];
            }
            return recognized;
        }
};


