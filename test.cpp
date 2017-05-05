#include "api.h"
#include <vector>
#include <string>
#include <iostream>
using namespace std;

int main(){
    vector<float> input;
    float x;
    ifstream inputfs("input.txt");
    while(inputfs >> x ){
        input.push_back(x);
    }
    cout<<"Loaded input"<<endl;

    NetAPI api("cvit_ocr_weights.xml", "lookup.txt");
    string S = api.recognize(&input[0], input.size()/32, 32);
    cout << S << endl;
    return 0;
}

