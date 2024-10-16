#include "NeuralNetwork.hpp"
#include "Debug.hpp"
#include "glcore.hpp"
#include "ComputeShader.hpp"
#include "ShaderUtil.hpp"
#include <random>
#include "BitonicMergeSort.hpp"

unsigned int generateRandomUint(unsigned int min, unsigned int max) 
{
	std::random_device rd;  // Pour obtenir une graine aléatoire
	std::mt19937 gen(rd()); // Générateur basé sur la graine
	std::uniform_int_distribution<unsigned int> dis(min, max); // Distribution uniforme
	return dis(gen);
}

namespace Ge
{
	NeuralNetwork::NeuralNetwork(NeuralInitData nid)
	{
		if (nid.neuralLayer.size() < 2)
		{
			Debug::Error("Not enough layer min 2");
			return;
		}
		m_nid = nid;
		size_t lastSize = 0;
		m_globalSize = 0;
		for (int i = 0; i < nid.neuralLayer.size(); i++)
		{
			m_layerOffset.push_back(m_globalSize);
			m_globalSize += lastSize * nid.neuralLayer[i].size + (nid.neuralLayer[i].size*2);
			lastSize = nid.neuralLayer[i].size;
		}
		m_ngd.sizeNN = m_globalSize;
		m_ngd.lop = m_layerOffset[0];
		m_ngd.loc = m_layerOffset[1];
		m_ngd.sizePPold = 0;
		m_ngd.sizePPrevious = nid.neuralLayer[0].size;
		m_ngd.sizePCurrent = nid.neuralLayer[nid.neuralLayer.size()-1].size;
		m_ngd.activationType = nid.neuralLayer[nid.neuralLayer.size() - 1].typeActivation;
		m_ngd.numberNN = nid.numberNeuralNetwork;
		m_ngd.negativeInputSize = nid.neuralLayer[0].size * 2;
		m_globalSize *= nid.numberNeuralNetwork;
		Debug::Log("NeuralNetwork float number %d", m_globalSize);
		m_nndata = new LargeArray<float>(m_globalSize);
		m_p2Size = ShaderUtil::PowTwoUp(m_ngd.numberNN);

		glGenBuffers(1, &m_ssboGlobalData);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboGlobalData);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(NeuralGlobalData), &m_ngd, GL_STREAM_DRAW);

		glGenBuffers(1, &m_ssboInputNNData);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboInputNNData);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * nid.neuralLayer[0].size, nullptr, GL_DYNAMIC_COPY);

		glGenBuffers(1, &m_ssboNNData);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboNNData);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * (m_globalSize - (nid.neuralLayer[0].size*2* nid.numberNeuralNetwork)), nullptr, GL_DYNAMIC_COPY);

		glGenBuffers(1, &m_ssboError);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboError);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * m_ngd.numberNN, nullptr, GL_DYNAMIC_COPY);		

		glGenBuffers(1, &m_ssboIndexError);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboIndexError);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * m_p2Size, nullptr, GL_DYNAMIC_COPY);

		glGenBuffers(1, &m_ssboExpectedResult);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboExpectedResult);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * nid.neuralLayer[nid.neuralLayer.size() - 1].size, nullptr, GL_STREAM_DRAW);

		NeuralMutate nm;
		nm.numberOfBest = nid.numberOfBest;
		nm.weight = nid.mutationWeight;
		nm.activation = nid.mutationActivation;
		nm.weightScale = nid.mutationWeightScale;
		
		glGenBuffers(1, &m_ssboMutate);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboMutate);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(NeuralMutate), &nm, GL_DYNAMIC_COPY);

		m_computeLayer = new ComputeShader("../Asset/Shader/NeuralNetwork.C.glsl");		
		m_initLayer = new ComputeShader("../Asset/Shader/InitNeuralNetwork.C.glsl");
		m_computeError = new ComputeShader("../Asset/Shader/FitnessNeuralNetwork.C.glsl");
		m_computeCopyElitims = new ComputeShader("../Asset/Shader/ElitimsNeuralNetwork.C.glsl");
		m_computeMutate = new ComputeShader("../Asset/Shader/MutateNeuralNetwork.C.glsl");
		m_randomlocation = glGetUniformLocation(m_computeMutate->getProgram(), "random_uniform");
		m_nbBestlocation = glGetUniformLocation(m_computeCopyElitims->getProgram(), "numberOfBest");

		m_bms = new ComputeShader("../Asset/Shader/BitonicMergeSortFloat.C.glsl");
		m_fill = new ComputeShader("../Asset/Shader/Fill.C.glsl");
		Debug::Log("Randomize Neural Network Weight");
		m_initLayer->use();
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboGlobalData);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssboNNData);
		m_ngd.sizePPold = 0;
		GLuint randomUniformLocation = glGetUniformLocation(m_initLayer->getProgram(), "random_uniform");
		int x, y, z;		
		for (int i = 1; i < m_nid.neuralLayer.size(); i++)
		{
			m_ngd.sizePCurrent = m_nid.neuralLayer[i].size;
			m_ngd.activationType = m_nid.neuralLayer[i].typeActivation;
			m_ngd.sizePPrevious = m_nid.neuralLayer[i-1].size;
			m_ngd.lop = m_layerOffset[i-1];
			m_ngd.loc = m_layerOffset[i];

			/*std::cout << "--------------NeuralGlobalData----------------" << std::endl;
			std::cout << "Number of Neural Networks: " << m_ngd.numberNN << std::endl;
			std::cout << "Size of Neural Network: " << m_ngd.sizeNN << std::endl;
			std::cout << "Size of Perceptrons (Previous-Previous Layer): " << m_ngd.sizePPold << std::endl;
			std::cout << "Size of Perceptrons (Previous Layer): " << m_ngd.sizePPrevious << std::endl;
			std::cout << "Size of Perceptrons (Current Layer): " << m_ngd.sizePCurrent << std::endl;
			std::cout << "Local Previous Offset (lop): " << m_ngd.lop << std::endl;
			std::cout << "Local Current Offset (loc): " << m_ngd.loc << std::endl;
			std::cout << "Activation Type: " << m_ngd.activationType << std::endl;*/

			glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboGlobalData);
			glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(NeuralGlobalData), &m_ngd);				
			ShaderUtil::CalcWorkSize(m_ngd.sizePCurrent* m_ngd.numberNN, &x, &y, &z);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboGlobalData);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssboNNData);
			glUniform1ui(randomUniformLocation, generateRandomUint(0, 100000));
			m_initLayer->dispatch(x, y, z);
			glFinish();
			m_ngd.sizePPold = m_ngd.sizePPrevious;
		}
	}

	void NeuralNetwork::propagate(std::vector<TrainingData> tds)
	{		
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboError);
		float* ssboError = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE);
		if (ssboError != nullptr)
		{
			memset(ssboError, 0.0f, m_ngd.numberNN * sizeof(float));
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		}
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);		

		int x, y, z;			
		for (int i = 0; i < tds.size(); i++)
		{
			m_computeLayer->use();
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboInputNNData);
			float* ssboData = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE);
			if (ssboData != nullptr)
			{
				memcpy(ssboData, tds[i].inputData.data(), tds[i].inputData.size()*sizeof(float));
				glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
			}
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

			m_ngd.sizePPold = 0;
			for (int j = 1; j < m_nid.neuralLayer.size(); j++)
			{
				m_ngd.sizePCurrent = m_nid.neuralLayer[j].size;
				m_ngd.activationType = m_nid.neuralLayer[j].typeActivation;
				m_ngd.sizePPrevious = m_nid.neuralLayer[j - 1].size;
				m_ngd.lop = m_layerOffset[j - 1];
				m_ngd.loc = m_layerOffset[j];

				glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboGlobalData);
				glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(NeuralGlobalData), &m_ngd);
				ShaderUtil::CalcWorkSize(m_ngd.sizePCurrent * m_ngd.numberNN, &x, &y, &z);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboGlobalData);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssboNNData);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_ssboInputNNData);				
				m_computeLayer->dispatch(x, y, z);
				glFinish();
				m_ngd.sizePPold = m_ngd.sizePPrevious;
			}

			glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboExpectedResult);
			glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, tds[i].expetedResult.size()*sizeof(float), tds[i].expetedResult.data());
			m_computeError->use();
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboGlobalData);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssboNNData);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_ssboError);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_ssboExpectedResult);
			ShaderUtil::CalcWorkSize(m_ngd.numberNN, &x, &y, &z);
			m_computeError->dispatch(x, y, z);
			glFinish();
		}
		BitonicMergeSort::Sort(m_bms, m_fill, m_ssboIndexError, m_ssboError, m_ngd.numberNN, m_p2Size);
		m_computeCopyElitims->use();
		ShaderUtil::CalcWorkSize(m_nid.numberOfBest* m_ngd.sizeNN, &x, &y, &z);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboGlobalData);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssboNNData);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_ssboIndexError);
		glUniform1ui(m_nbBestlocation, m_nid.numberOfBest);
		m_computeCopyElitims->dispatch(x, y, z);
		glFinish();
		m_computeMutate->use();
		m_ngd.sizePPold = 0;
		for (int j = 1; j < m_nid.neuralLayer.size(); j++)
		{
			m_ngd.sizePCurrent = m_nid.neuralLayer[j].size;
			m_ngd.activationType = m_nid.neuralLayer[j].typeActivation;
			m_ngd.sizePPrevious = m_nid.neuralLayer[j - 1].size;
			m_ngd.lop = m_layerOffset[j - 1];
			m_ngd.loc = m_layerOffset[j];

			glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboGlobalData);
			glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(NeuralGlobalData), &m_ngd);
			ShaderUtil::CalcWorkSize(m_ngd.sizePCurrent * m_ngd.numberNN, &x, &y, &z);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboGlobalData);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssboNNData);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_ssboIndexError);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_ssboMutate);
			glUniform1ui(m_randomlocation, generateRandomUint(0, 100000));
			m_computeMutate->dispatch(x, y, z);
			glFinish();
			m_ngd.sizePPold = m_ngd.sizePPrevious;//m_randomlocation
		}		

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboIndexError);
		int* ssboIndexError = (int*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
		if (ssboIndexError != nullptr)
		{
			memcpy(&m_index_best, ssboIndexError, sizeof(int));
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		}

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboError);
		ssboError = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
		if (ssboError != nullptr)
		{
			std::cout << ssboError[m_index_best] << std::endl;
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		}
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	void NeuralNetwork::computeNetwork(std::vector<float> &data, std::vector<float> &result)
	{				
		m_computeLayer->use();
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboInputNNData);
		float* ssboData = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE);
		if (ssboData != nullptr)
		{
			memcpy(ssboData, data.data(), data.size() * sizeof(float));
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		}
		int x, y, z;
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		m_ngd.sizePPold = 0;
		for (int j = 1; j < m_nid.neuralLayer.size(); j++)
		{
			m_ngd.sizePCurrent = m_nid.neuralLayer[j].size;
			m_ngd.activationType = m_nid.neuralLayer[j].typeActivation;
			m_ngd.sizePPrevious = m_nid.neuralLayer[j - 1].size;
			m_ngd.lop = m_layerOffset[j - 1];
			m_ngd.loc = m_layerOffset[j];

			glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboGlobalData);
			glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(NeuralGlobalData), &m_ngd);
			ShaderUtil::CalcWorkSize(m_ngd.sizePCurrent * m_ngd.numberNN, &x, &y, &z);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboGlobalData);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssboNNData);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_ssboInputNNData);
			m_computeLayer->dispatch(x, y, z);
			glFinish();
			m_ngd.sizePPold = m_ngd.sizePPrevious;
		}
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboNNData);
		ssboData = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE);
		if (ssboData != nullptr)
		{
			int off_inn = m_index_best * m_ngd.sizeNN;
			int off_link;
			for (int i = 0; i < m_ngd.sizePCurrent; i++)
			{
				off_link = ((i + 1) * m_ngd.sizePPrevious) + (i * 2);
				result[i] = ssboData[(off_inn + m_ngd.loc + off_link) - (m_ngd.negativeInputSize * (m_index_best + 1))];
			}
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		}
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	void NeuralNetwork::drawBestNeuralNetwork()
	{
		int sub = (m_nid.neuralLayer[0].size * 2 * (m_index_best +1));
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboNNData);
		float* ssboData = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE);
		if (ssboData != nullptr)
		{
			for (int i = m_nid.neuralLayer[0].size * 2; i < m_ngd.sizeNN; i++)
			{
				std::cout << i << " : " << ssboData[m_index_best * m_ngd.sizeNN + i - sub] << std::endl;
			}
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		}
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	NeuralNetwork::~NeuralNetwork()
	{		
		delete m_initLayer;
		delete m_computeLayer;
		delete m_computeError;
		delete m_computeCopyElitims;
		delete m_computeMutate;

		delete m_bms;
		delete m_fill;

		glDeleteBuffers(1, &m_ssboNNData);
		glDeleteBuffers(1, &m_ssboInputNNData);
		glDeleteBuffers(1, &m_ssboGlobalData);
		glDeleteBuffers(1, &m_ssboError);
		glDeleteBuffers(1, &m_ssboIndexError);
		glDeleteBuffers(1, &m_ssboExpectedResult);
		glDeleteBuffers(1, &m_ssboMutate);		
		delete m_nndata;
	}
}