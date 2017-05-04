/*Copyright 2007,2008 Alex Graves

This file is part of nnl.

nnl is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

nnl is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with nnl.  If not, see <http://www.gnu.org/licenses/>.*/

#define NET_EXPORTS
#include <xercesc/dom/DOMLSSerializer.hpp>
#include <xercesc/dom/DOMImplementationRegistry.hpp>
#include <xercesc/dom/DOMImplementationLS.hpp>
#include <math.h>
#include <iostream>
#include "NDimNet.h"
#include "RegressionOutputLayer.h"
#include "ClassificationOutputLayer.h"
#include "TranscriptionOutputLayer.h"
#include "LabellingOutputLayer.h"
#include "SequenceMIAOutputLayer.h"
#include "SequenceClassificationOutputLayer.h"
#include "Connection.h"
#include "NDimBackpropLayer.h"
#include "NDimLstmLayer.h"
#include "Helpers.h"
#include "Typedefs.h"


NDimNet::NDimNet(const DOMElement* element, const DataDimensions& dims, vector<string>& criteria):
		numDims(dims.numDims),
		netNode((DOMElement*)element->cloneNode(true)),
		inputBlockDimString(""),
		inputOverlap(0),
		multidirectional(true),
		outputLayer(0)
{
	//load in net attributes
	//"task" required, others optional
	DOMNamedNodeMap* netAttributes = netNode->getAttributes();
	task = XMLString::transcode(netAttributes->getNamedItem(XMLString::transcode("task"))->getNodeValue());
	DOMNode* multidirectionalNode = netAttributes->getNamedItem(XMLString::transcode("multidirectional"));
	if (multidirectionalNode)
	{
		multidirectional = (bool)atoi(XMLString::transcode(multidirectionalNode->getNodeValue()));
	}
	DOMNode* blockDimsNode = netAttributes->getNamedItem(XMLString::transcode("inputBlockDims"));
	if (blockDimsNode)
	{
		inputBlockDimString = XMLString::transcode(blockDimsNode->getNodeValue());
	}
	DOMNode* inputOverlapNode = netAttributes->getNamedItem(XMLString::transcode("inputOverlap"));
	if (inputOverlapNode)
	{
		inputOverlap = atoi(XMLString::transcode(inputOverlapNode->getNodeValue()));
	}

	//create output layer
	if (task == "regression")
	{
		outputLayer = new RegressionOutputLayer("output", dims.targetPattSize, criteria);	
	}
	else if (task == "classification")
	{
		outputLayer = new ClassificationOutputLayer("output", dims.labels.size(), dims.labels, criteria);	
	}
	else if (task == "transcription")
	{	
		outputLayer = new TranscriptionOutputLayer("output", dims.labels, numDims, criteria, 
				dims.labelNumbers, dims.wordDelimiter, element);//dictFilename, bigramFilename);
	}
	else if (task == "labelling")
	{
		outputLayer = new LabellingOutputLayer("output", dims.labels, numDims, criteria);
	}
	else if (task == "sequence MIA")
	{
		outputLayer = new SequenceMIAOutputLayer("output", dims.labels, criteria);
	}
	else if (task == "sequence classification")
	{
		outputLayer = new SequenceClassificationOutputLayer("output", dims.labels.size(), dims.labels, criteria);
	}
	else if (task != "unsupervised")
	{
		cerr << "ERROR: NDimNet::NDimNet task " << task << " unknown, exiting" << endl;
		exit(0);
	}

	//create input layers
	inputLayer = new InputLayer("input", dims.inputPattSize, inputBlockDimString);

	//load hidden levels
	const DOMElement* hiddenLayerElement = (DOMElement*)netNode->getElementsByTagName(XMLString::transcode("Levels"))->item(0);
	if (hiddenLayerElement)
	{
		string oldLevelName = "";
		int subsampleDepth;
		bool prediction = false;
		for (const DOMNode* levelNode = hiddenLayerElement->getFirstChild(); levelNode; levelNode = levelNode->getNextSibling())
		{
			if (levelNode->getNodeType() == DOMNode::ELEMENT_NODE)
			{
				stringstream ss;
				ss << "level_" << (int)levels.size();
				NDimLevel* level = new NDimLevel(ss.str(), numDims, multidirectional, levelNode, inputOverlap);
				level->connectFrom(&bias);
				if (levels.empty())
				{
					level->connectFrom(inputLayer);
				}
				else
				{
#ifdef DELAY_IN_CONN
					if (prediction)
					{
						level->connectFrom(levels.back()->predLayer);
					}
					else
#endif
					{
						NDimBackpropLayer<Tanh>* sl = new NDimBackpropLayer<Tanh>(oldLevelName + "_subsample", "tanh", subsampleDepth);
						level->connectFrom(sl);
						levels.back()->subsampleLayer = sl;
						levels.back()->connectTo(sl);
					}
				}
				DOMNode* subsampleDepthNode = levelNode->getAttributes()->getNamedItem(XMLString::transcode("subsampleDepth"));
				if (subsampleDepthNode)
				{
					subsampleDepth = atoi(XMLString::transcode(subsampleDepthNode->getNodeValue()));
				}
				else
				{
					subsampleDepth = level->getOutputDepth() / level->size();
					if (!subsampleDepth)
					{
						subsampleDepth = 1;
					}
				}
#ifdef DELAY_IN_CONN
				DOMNode* PredictionNode = levelNode->getAttributes()->getNamedItem(XMLString::transcode("predictionStep"));
				if (PredictionNode)
				{
					prediction = true;
					int predStep = atoi(XMLString::transcode(PredictionNode->getNodeValue()));
#ifndef DELAY_IN_CONN
					cerr << "NDimNet::NDimNet macro DELAY_IN_CONN must be defined for prediction layers, exiting" << endl;
					exit(0);
#endif
					Layer* predInputLayer;
					PredictionOutputLayer* firstPredOutputLayer;
					bool predictErrors;
					if (levels.empty())
					{
						predInputLayer = (Layer*)inputLayer;
						firstPredOutputLayer = 0;
					}
					else
					{
						predInputLayer = (Layer*)levels.back()->predLayer;
						firstPredOutputLayer = levels.front()->predLayer;
					}					
					level->predLayer = new PredictionOutputLayer(ss.str()+"_prediction", criteria, predInputLayer, levels.size() & 1, firstPredOutputLayer);
					if (multidirectional && numDims == 2)
					{
						vector<int> delayCoords(numDims);
						delayCoords[0] = 0;
						delayCoords[1] = -predStep;
						new Connection(level->subnets[0]->layer, level->predLayer, -1, &delayCoords);
						delayCoords[0] = predStep;
						delayCoords[1] = 0;
						new Connection(level->subnets[1]->layer, level->predLayer, -1, &delayCoords);
						delayCoords[0] = -predStep;
						delayCoords[1] = 0;
						new Connection(level->subnets[2]->layer, level->predLayer, -1, &delayCoords);
						delayCoords[0] = 0;
						delayCoords[1] = predStep;
						new Connection(level->subnets[3]->layer, level->predLayer, -1, &delayCoords);
					}
					else
					{
						cerr << "NDimNet::NDimNet prediction layers only implemented for 2d multidirectional nets, exiting" << endl;
						exit(0);
					}
					
				}
				else
				{
					prediction = false;
				}
#endif
				oldLevelName = ss.str();
				levels.push_back(level);
			}
		}
	}
	if (outputLayer)
	{
		levels.back()->connectTo(outputLayer);
		new Connection(&bias, outputLayer);
	}
}

NDimNet::~NDimNet(void)
{
	delete inputLayer;
	delete outputLayer;
}

void NDimNet::printOutputs(ostream& out)
{
	if (outputLayer)
	{
		outputLayer->printOutputs(out);
	}
}

void NDimNet::outputInternalVariables(const string& path) const
{
	inputLayer->outputInternalVariables(path);
	for (VPNDLCI it = levels.begin(); it != levels.end(); ++it)
	{
		(*it)->outputInternalVariables(path);
	}
	if (outputLayer)
	{
		outputLayer->outputInternalVariables(path);
	}
}
// #define PREDICTION_JACOBIAN_HACK
//TODO store output layers in <string, outputLayer*> map and pass in name as parameter
void NDimNet::outputJacobian(const DataSequence& seq, const string& path, int timestep, int output)
{
#ifndef PREDICTION_JACOBIAN_HACK
	if (!outputLayer)
	{
		cout << "ERROR NDimNet::outputJacobian - no output layer" << endl;
		return;
	}
#endif
	feedForward(seq);
#ifdef PREDICTION_JACOBIAN_HACK
	int numOutputs = levels.front()->predLayer->outputSize();
#else
	int numOutputs = outputLayer->outputSize();
#endif
	int initOutput;
	int finalOutput;
	if (output < 0)
	{
		initOutput = 0;
		finalOutput = numOutputs;
	}
	else if (output > numOutputs)
	{
		cerr << "ERROR NDimNet::outputJacobian: output " << output << " > numOutputs " << numOutputs << ", exiting" << endl;
		exit(0);
	}
	else
	{
		initOutput = output;
		finalOutput = output + 1;
	}
	for (int i = initOutput; i < finalOutput; ++i)
	{
#ifdef PREDICTION_JACOBIAN_HACK
		levels.front()->predLayer->setErrToOutputDeriv(timestep, i);
#else
		outputLayer->setErrToOutputDeriv(timestep, i);
#endif
		//feed back through hidden levels
		for (VPNDLRI it = levels.rbegin(); it != levels.rend(); ++it)
		{
			(*it)->feedBack();
		}

		//feed back errors to input layer
		//TODO put whole back pass in input layer
		int seqLength = levels.front()->getOutputSeqLength();
// 		for (int n = 0; n < seqLength; ++n)
// 		{
// 			inputLayer->feedBack(n);
// 		}
		inputLayer->feedBack();

		//print out jacobians
		stringstream jacPath;
		jacPath << path << "jacobian_timestep_" << timestep << "_output_" << i << "_";
		inputLayer->outputInternalVariables(jacPath.str());
		for (VPNDLCI it = levels.begin(); it != levels.end(); ++it)
		{
			(*it)->outputInternalVariables(jacPath.str());
		}
#ifndef PREDICTION_JACOBIAN_HACK
		outputLayer->outputInternalVariables(jacPath.str());
#endif
	}
}

double NDimNet::calculateGradient(map<const string, pair<int,double> >& errorMap, const DataSequence& seq)
{
	feedForward(seq);
	double err = injectSequenceErrors(errorMap, seq);
	feedBack(seq);
	return err;
}

double NDimNet::calculateError(map<const string, pair<int,double> >& errorMap, const DataSequence& seq)
{
	feedForward(seq);
	return injectSequenceErrors(errorMap, seq);
}

double NDimNet::injectSequenceErrors(map<const string, pair<int,double> >& errorMap, const DataSequence& seq)
{
// 	double error = 0;
// 	bool storeTotalError = false;
// 	for (VPNDLCI it = levels.begin(); it != levels.end(); ++it)
// 	{
// 		error += (*it)->injectSequenceErrors(errorMap, seq);
// 	}
// 	if (error > 0)
// 	{
// 		storeTotalError = true;
// 	}
// 	if (outputLayer)
// 	{
// 		error += outputLayer->injectSequenceErrors(errorMap, seq);
// 	}
// 	if (storeTotalError)
// 	{
// 		errorMap["totalError"] += make_pair<int, double>(1, error);
// 	}
// 	return error;
	return outputLayer->injectSequenceErrors(errorMap, seq);
}

void NDimNet::save(const string& indent, ostream& out) const
{
	// construct the DOMWriter
	DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation(XMLString::transcode("XML 1.0 Traversal"));
	DOMLSSerializer* myWriter = ((DOMImplementationLS*)impl)->createLSSerializer();


	// serialize a DOMNode to a UTF-16 string
	XMLCh* nodeString = myWriter->writeToString(netNode);

	//write the string to file
	out << indent << XMLString::transcode(nodeString) << endl;

	// release the memory
	//TODO is this slow?
	XMLString::release(&nodeString); 
	myWriter->release(); 
}

void NDimNet::print(ostream& out) const
{
	save("", out);
	//out << numDims << " dimensional net, task " << task << " multidirectional " << multidirectional << endl;
	out << "input layer:" << endl;
	inputLayer->print(out);
	out << "hidden levels:" << endl;
	for (VPNDLCI it = levels.begin(); it != levels.end(); ++it)
	{
		out << "level " << (int)distance(levels.begin(), it) << endl;
		(*it)->print(out);
	}
	if (outputLayer)
	{
		out << "output layer:" << endl;
		outputLayer->print(out);
	}
	out << endl;
	out << "connections:" << endl;
	for (VPNDLCI it = levels.begin(); it != levels.end(); ++it)
	{
		(*it)->printInputConns(out);
	}
	if (outputLayer)
	{
		outputLayer->printInputConns(out);
	}
}

void NDimNet::feedForward(const DataSequence& seq)
{
	if (seq.dimensions.size() != numDims)
	{
		cerr << "ERROR: NDimNet::feedForward seq " << seq.tag  << endl;
		cerr << "seq.dimensions.size() " << (int)seq.dimensions.size();
		cerr << " != numDims " << numDims;
		cerr << ", exiting" << endl;
		exit(0);
	}

	//copy the input activations
	inputLayer->copyInputs(seq);
	const vector<int>* seqDims = &inputLayer->getSeqDims();

	//feed forward through the hidden levels
	for (VPNDLI it = levels.begin(); it != levels.end(); ++it)
	{
		(*it)->feedForward(*seqDims);
		seqDims = &(*it)->getOutputSeqDims();
	}
	
	if (outputLayer)
	{
		//feed forward to the output layer
		int seqLength = levels.back()->getOutputSeqLength();
		outputLayer->feedForward(*seqDims, seqLength, seq);
	}
}


void NDimNet::feedBack(const DataSequence& seq)
{
	if (outputLayer)
	{
		//update output layer derivs
		//TODO put whole back pass in output layer
		int outputSeqLength = levels.back()->getOutputSeqLength();
		for (int n = 0; n < outputSeqLength; ++n)
		{
			outputLayer->updateDerivs(n);
		}
	}

	//feed back through hidden levels
	for (VPNDLRI it = levels.rbegin(); it != levels.rend(); ++it)
	{
		(*it)->feedBack();
	}

	//feed back errors to input layer
	inputLayer->feedBack();
}

//needs to be done after every Net topology change
void NDimNet::build()
{
	for (VPNDLI it = levels.begin(); it != levels.end(); ++it)
	{
		(*it)->build();
	}
	if (outputLayer)
	{
		outputLayer->build();
	}
}
