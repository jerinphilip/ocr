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
    vector<int> expected;
    expected.push_back(84);
    expected.push_back(38);
    expected.push_back(12);
    VecVecInt labels;
    labels.push_back(expected);
    VecFloat errors = api.train(sequences, labels);
    for(int i=0; i<(int)sequences.size(); i++){
        cout << i+1 <<" error = " << errors[i];
        cout << endl;
    }
    
    vector<string> S = api.test(input);
    for (vector<string>::iterator s = S.begin();
            s != S.end(); s++)
        cout << *s << endl;
    return 0;
}

