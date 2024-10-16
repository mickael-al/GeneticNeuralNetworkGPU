#include "NeuralNetworkTest.hpp"
#include "PhysicsWraper.hpp"
#include "CollisionShape.hpp"
#include "Model.hpp"
#include "Time.hpp"
#include "NeuralNetwork.hpp"

void NeuralNetworkTest::start()
{
	m_pc = Engine::getPtrClassAddr();
	NeuralInitData nid;
	nid.neuralLayer.push_back({ 2, 0 });
	nid.neuralLayer.push_back({ 3, 0 });
	nid.neuralLayer.push_back({ 1, 0 });
	nid.numberNeuralNetwork = numberNeuralNetwork;
	nid.numberOfBest = numberOfBest;
	nid.mutationWeight = mutationWeight;
	nid.mutationWeightScale = mutationWeightScale;
	nid.mutationActivation = mutationActivation;
	m_neuralN = new NeuralNetwork(nid);
	std::vector<TrainingData> training;
	training.push_back({ {-1,-1}, {-1} });
	training.push_back({ {-1,1}, {1} });
	training.push_back({ {1,-1}, {1} });
	training.push_back({ {1,1}, {-1} });
	m_neuralN->propagate(training);
}

void NeuralNetworkTest::fixedUpdate()
{
	
}

void NeuralNetworkTest::update()
{
	if (m_pc->inputManager->getKey(GLFW_KEY_P))
	{
		std::vector<TrainingData> training;
		training.push_back({ {-1,-1}, {-1} });
		training.push_back({ {-1,1}, {1} });
		training.push_back({ {1,-1}, {1} });
		training.push_back({ {1,1}, {-1} });
		m_neuralN->propagate(training);
	}
	if (m_pc->inputManager->getKeyDown(GLFW_KEY_O))
	{
		std::vector<TrainingData> training;
		training.push_back({ {-1,-1}, {-1} });
		training.push_back({ {-1,1}, {1} });
		training.push_back({ {1,-1}, {1} });
		training.push_back({ {1,1}, {-1} });
		m_neuralN->propagate(training);
	}
	if (m_pc->inputManager->getKeyDown(GLFW_KEY_L))
	{
		std::cout << "-----------------------" << std::endl;
		m_neuralN->drawBestNeuralNetwork();
		std::cout << "-----------------------" << std::endl;
	}
	if (m_pc->inputManager->getKeyDown(GLFW_KEY_K))
	{
		std::vector<TrainingData> training;
		training.push_back({ {-1,-1}, {-1} });
		training.push_back({ {-1,1}, {1} });
		training.push_back({ {1,-1}, {1} });
		training.push_back({ {1,1}, {-1} });
		std::vector<float> result;
		result.resize(1);
		for (int i = 0; i < 4; i++)
		{
			m_neuralN->computeNetwork(training[i].inputData, result);
			std::cout << training[i].inputData[0] << " " << training[i].inputData[1] << " " << result[0] << std::endl;
		}
	}
}

void NeuralNetworkTest::stop()
{
	delete m_neuralN;
}

void NeuralNetworkTest::onGUI()
{

}