#ifndef __NEURAL_NETWORK__
#define __NEURAL_NETWORK__

#include <iostream>
#include <vector>
#include "LargeArray.hpp"

struct NeuralLayer
{
	size_t size;
	unsigned int typeActivation;
};

struct NeuralGlobalData
{
	unsigned int numberNN;//nombre de reseaux de neuronne
	unsigned int sizeNN;//la taille d'un reseau de neuronne
	unsigned int sizePPold;//la taille de perceptrons dans la couche precedente precedente
	unsigned int sizePPrevious;//la taille de perceptrons dans la couche precedente
	unsigned int sizePCurrent;//la taille de perceptrons dans la couche actuel	
	unsigned int lop;//local previous offset
	unsigned int loc;//local offset current
	unsigned int activationType;//activation Type
};

struct NeuralInitData
{
	unsigned int numberNeuralNetwork;
	std::vector<NeuralLayer> neuralLayer;
};

struct TrainingData
{
	float* inputData;
	float* expetedResult;
};

namespace Ge
{
	class ComputeShader;
	class NeuralNetwork
	{
	public:
		NeuralNetwork(NeuralInitData nid);		
		void propagate();
		void fitnesse(TrainingData * tds);
		~NeuralNetwork();
	private:
		unsigned int m_ssboGlobalData;
		unsigned int m_ssboNNData;
		std::vector<unsigned int> m_layerOffset;
		NeuralGlobalData m_ngd;		
		NeuralInitData m_nid;
		unsigned int m_globalSize;
		LargeArray<float>* m_nndata;
		ComputeShader* m_computeLayer;
		ComputeShader* m_initLayer;
	};
}

#endif //!__NEURAL_NETWORK__