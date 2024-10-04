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
	nid.neuralLayer.push_back({ 2, 0 });
	nid.neuralLayer.push_back({ 1, 0 });
	nid.numberNeuralNetwork = numberNeuralNetwork;
	m_neuralN = new NeuralNetwork(nid);
}

void NeuralNetworkTest::fixedUpdate()
{
	
}

void NeuralNetworkTest::update()
{

}

void NeuralNetworkTest::stop()
{
	delete m_neuralN;
}

void NeuralNetworkTest::onGUI()
{

}