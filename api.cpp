#include "api.h"
using namespace std;


NetAPI::NetAPI (const char *weights_f, const char *lookup_f) {
    weights_file = weights_f;
    lookup_file = lookup_f;
    initialize_dimensions();
    load_weights_file();
    XMLPlatformUtils::Initialize();
}

void NetAPI::load_weights_file(){
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


void NetAPI::initialize_dimensions(){
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

vector<string> NetAPI::recognize(vector<float> &inputs, int width, int height){
    sequence.dimensions = {width};
    sequence.size = width;
    sequence.inputs = &inputs[0];

    net->feedForward(sequence);
    vector<int> recognisedIndices;
    output->getMostProbableString(recognisedIndices);
    vector<string> recognized;
    for(vector<int>::iterator p=recognisedIndices.begin(); 
            p!=recognisedIndices.end(); p++){
        recognized.push_back(dimensions.labels[*p]);
    }
    return recognized;
}
