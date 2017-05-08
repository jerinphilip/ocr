#include "api.h"
#include <vector>
#include <string>
#include <iostream>
using namespace std;

int main(){
    VecFloat input;
    float x;
    ifstream inputfs("../etc/input.txt");
    while(inputfs >> x ){
        input.push_back(x);
    }

    VecVecFloat sequences;
    sequences.push_back(input);
    //cout<<weights_f<<" "<<lookup_f<<endl;
    NetAPI api("../etc/cvit_ocr_weights.xml", "../etc/lookup.txt");
    vector<string> S = api.recognize(input);
    vector<int> expected;
    expected.push_back(84);
    expected.push_back(38);
    expected.push_back(12);
    VecVecInt labels;
    labels.push_back(expected);
    api.train(sequences, labels);
    for (vector<string>::iterator s = S.begin();
            s != S.end(); s++)
        cout << *s << endl;
    return 0;
}

