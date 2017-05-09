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

    stringstream ss;
    net->save("\t", ss);
    netExport = ss.str();

    gradientFollower = GradientFollowerMaker::makeStdGradientFollower();

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


vector<string> NetAPI::test(vector<float> &inputs){
    int height = 32;
    int width = (int)inputs.size()/height;
    sequence.dimensions.clear();
    sequence.dimensions.push_back(width);
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

vector<DataSequence> NetAPI::generateSequences(VecVecFloat &inputs, VecVecInt &labels){

    vector<DataSequence> result;
    int height = 32, width;

    for(int i=0; i <  (int)inputs.size(); i++){
        vector<float> &input = inputs[i];
        DataSequence seq;
        width = (int)input.size()/height;
        seq.dimensions.push_back(width);
        seq.size = width;
        seq.inputs = &input[0];
        seq.targetSeq = labels[i];
        result.push_back(seq);
    }
    return result;
}

VecFloat NetAPI::train(VecVecFloat &inputs, VecVecInt &labels){
    vector<DataSequence> sequences;
    sequences = generateSequences(inputs, labels);
    VecFloat errors;

    int maxEpochs = 200, epoch=0;
    bool satisfactory = false;
    float satisfactoryError = 1e-2;

    errorMapType errorMap;

    while (!satisfactory && epoch < maxEpochs){
        //for(auto &sequences: sequences){
        /*
        cout << "Satisfactory: "<<satisfactory << endl;
        cout << "Epoch: "<<epoch <<endl;
        */
        for(vector<DataSequence>::iterator sequence = sequences.begin();
                sequence != sequences.end();
                sequence ++ ){
            //cout << "Sequence gradient computing: "<<endl;
            net->calculateGradient(errorMap, *sequence);
            //sequence->print();
        }

        /*
        for(errorMapType::iterator p = errorMap.begin();
                p != errorMap.end(); p++){
            cout << p->first << ": (";
            cout << (p->second).first << ",";
            cout << (p->second).second << ")"<<endl;
        }
        */

        satisfactory = errorMap["ctcMlError"].second < satisfactoryError;
        /*
        cout << "updating, backpropogation"<<endl;
        cout << "----" <<endl;
        */
        gradientFollower->updateWeights();
        gradientFollower->resetDerivs();
        errorMap.clear();
        epoch += 1;
    }

    for(vector<DataSequence>::iterator sequence = sequences.begin();
            sequence != sequences.end();
            sequence ++ ){
        errorMap.clear();
        net->calculateGradient(errorMap, *sequence);
        errors.push_back(errorMap["ctcMlError"].second);
    }

    return errors;
}

string NetAPI::exportModel(){
    stringstream ss;
    ss << "<NeuralNet>" <<endl;
    ss << netExport << endl;
    DataExportHandler::instance().save("\t", ss);
    ss << "</NeuralNet>"<<endl;
    return ss.str();
}
