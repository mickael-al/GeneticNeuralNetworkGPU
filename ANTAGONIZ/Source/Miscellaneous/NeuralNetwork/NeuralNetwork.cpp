#include "NeuralNetwork.hpp"
#include "Debug.hpp"
#include "glcore.hpp"
#include "ComputeShader.hpp"
#include "ShaderUtil.hpp"
#include <random>

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
		m_globalSize *= nid.numberNeuralNetwork;
		Debug::Log("NeuralNetwork float number %d", m_globalSize);
		m_nndata = new LargeArray<float>(m_globalSize);

		glGenBuffers(1, &m_ssboGlobalData);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboGlobalData);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(NeuralGlobalData), &m_ngd, GL_STREAM_DRAW);

		glGenBuffers(1, &m_ssboNNData);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboNNData);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * m_globalSize, nullptr, GL_DYNAMIC_COPY);
		m_computeLayer = new ComputeShader("../Asset/Shader/NeuralNetwork.C.glsl");		
		m_initLayer = new ComputeShader("../Asset/Shader/InitNeuralNetwork.C.glsl");
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

			std::cout << "--------------NeuralGlobalData----------------" << std::endl;
			std::cout << "Number of Neural Networks: " << m_ngd.numberNN << std::endl;
			std::cout << "Size of Neural Network: " << m_ngd.sizeNN << std::endl;
			std::cout << "Size of Perceptrons (Previous-Previous Layer): " << m_ngd.sizePPold << std::endl;
			std::cout << "Size of Perceptrons (Previous Layer): " << m_ngd.sizePPrevious << std::endl;
			std::cout << "Size of Perceptrons (Current Layer): " << m_ngd.sizePCurrent << std::endl;
			std::cout << "Local Previous Offset (lop): " << m_ngd.lop << std::endl;
			std::cout << "Local Current Offset (loc): " << m_ngd.loc << std::endl;
			std::cout << "Activation Type: " << m_ngd.activationType << std::endl;

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

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboNNData);
		float* ssboData = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
		if (ssboData != nullptr) 
		{
			for (int i = 0; i < m_globalSize; ++i) 
			{
				std::cout << "ssboData[" << i << "] = " << ssboData[i] << std::endl;
			}
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		}
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	}

	void NeuralNetwork::propagate()
	{
	
	}

	void NeuralNetwork::fitnesse(TrainingData* tds)
	{
		
	}

	NeuralNetwork::~NeuralNetwork()
	{		
		delete m_initLayer;
		delete m_computeLayer;
		glDeleteBuffers(1, &m_ssboNNData);
		glDeleteBuffers(1, &m_ssboGlobalData);
		delete m_nndata;
	}
}