#include <iostream>
#include <cmath>
#include <armadillo>
#include <vector>
#include <sys/types.h>
#include <unistd.h>
#include <ctime>
#include <random>


using namespace std;
using namespace arma;

using ActivationFn = double (*)(double);
using ActivationDerivativeFn = double (*)(double);

double sigmoid(const double x){
	return 1 / (1 + exp(-x));
}

double sigmoidPrime(const double x){
	return x * ( 1 - x);
}

double sech2(const double x) {
    double th = tanh(x);
    return 1.0 - pow(th, 2); 
}

double relu(const double x){
	if(x <= 0)
		return 0;
	return x;
}

double relu_deriv(const double x){
	if(x <= 0)
		return 0;
	return 1;
}
class Layer{

private:
	mat m_inputMatrix;
	mat m_outputMatrix;
	mat m_weightMatrix;
	mat m_biasMatrix;
	double (*m_activation)(double);


public:
	
	Layer(const int &rows, const int &outputNodes, ActivationFn activation);
	const mat output(const mat &inputMatrix);
	void adjustWeights(const mat &deltaMatrix);
	void adjustBiases(const mat &deltaMatrix);
	const mat& getWeights();
	const mat& getBiases();
	mat getInput();
	mat getOutput();
	
};

Layer::Layer(const int &rows, const int &outputNodes, ActivationFn activation){

	m_weightMatrix 	= randn<mat>(outputNodes, rows);
	m_biasMatrix 	= randn<mat>(outputNodes, 1);
	m_activation 	= activation;
	// cout << m_weightMatrix << endl << m_biasMatrix  << endl;
}

const mat Layer::output(const mat &inputMatrix){
	m_inputMatrix 	= vectorise(inputMatrix);
	// m_outputMatrix 	= (m_weightMatrix * m_inputMatrix);
	m_outputMatrix 	= (m_weightMatrix * m_inputMatrix) + m_biasMatrix;
	m_outputMatrix 	= m_outputMatrix.transform(m_activation);
	return m_outputMatrix;
}

void Layer::adjustBiases(const mat &deltaMatrix){
	m_biasMatrix += deltaMatrix;
}
void Layer::adjustWeights(const mat &deltaMatrix){
	m_weightMatrix += deltaMatrix;
}

mat Layer::getOutput(){
	return m_outputMatrix;
}

mat Layer::getInput(){
	return m_inputMatrix;
}

const mat& Layer::getWeights(){
	return m_weightMatrix;
}
const mat& Layer::getBiases(){
	return m_biasMatrix;
}

class NeuralNetwork{

private:
	float m_learningRate;
	int m_hiddenLayers;
	int m_hiddenNodes;
	int m_outputNodes;
	double (*m_activation)(double);
	double (*m_activationDerivative)(double);

	vector<Layer> m_Layers;

	mat feedforward(mat inputMatrix);

public:
	NeuralNetwork() = default;
	NeuralNetwork(int inputNodes, int hiddenLayers, int hiddenNodes, float learningRate, int outputNodes, ActivationFn activation, ActivationDerivativeFn activationDerivative);
	mat backpropagate(mat errorMatrix);
	mat backpropagate(mat output, mat expectedOutput);
	mat output(mat inputMatrix){ return feedforward(inputMatrix); };

};

NeuralNetwork::NeuralNetwork(int inputNodes, int hiddenLayers, int hiddenNodes, float learningRate, int outputNodes, ActivationFn activation, ActivationDerivativeFn activationDerivative){
	arma_rng::set_seed_random();

	m_activation = activation;
	m_activationDerivative = activationDerivative;

	m_hiddenLayers 	= hiddenLayers;
	m_hiddenNodes	= hiddenNodes;
	m_outputNodes	= outputNodes;
	m_learningRate	= learningRate;

	m_Layers.push_back({inputNodes, hiddenNodes, m_activation} );

	for(int i = 0; i < hiddenLayers; ++i){
		m_Layers.push_back({hiddenNodes, hiddenNodes, m_activation});
	}

	m_Layers.push_back({hiddenNodes, outputNodes, m_activation});
}
mat NeuralNetwork::feedforward(mat inputMatrix){

	// add check to see if the input matrix is the correct shape

	for(Layer &l : m_Layers){
		inputMatrix = l.output(inputMatrix);
	}
	return inputMatrix;
}

mat NeuralNetwork::backpropagate(mat error){
	mat delta;

	for (auto &&layer = m_Layers.rbegin(); layer != m_Layers.rend(); ++layer) {

		delta = error % layer->getOutput().transform(m_activationDerivative);
		delta = m_learningRate * ( delta * layer->getInput().t() );
		layer->adjustWeights(delta);
		layer->adjustBiases( m_learningRate * error  );
		error = layer->getWeights().t() * error;

	}
	return error;

}

mat NeuralNetwork::backpropagate( mat output, mat expectedOutput ){
	// cout << "Error " << expectedOutput - output;
	return backpropagate(expectedOutput - output);
}


class AutoEncoder{

private:
	NeuralNetwork m_encoder;
	NeuralNetwork m_decoder;
	bool m_trained = false;
public:
	AutoEncoder(int inputNodes, int outputNodes, float learningRate, int dimensions, ActivationFn activation, ActivationDerivativeFn activationDerivative);
	void train(const vector<mat> &inputs, const vector<mat> &outputs);
	const NeuralNetwork & getEncoder() const;
	const NeuralNetwork & getDecoder() const;
};


AutoEncoder::AutoEncoder(int inputNodes, int outputNodes, float learningRate, int dimensions, ActivationFn activation, ActivationDerivativeFn activationDerivative){
	m_encoder = NeuralNetwork(inputNodes, 1, inputNodes, learningRate, dimensions, activation, activationDerivative);
	m_decoder = NeuralNetwork(dimensions, 1, inputNodes, learningRate, outputNodes, activation, activationDerivative);
}

void AutoEncoder::train(const vector<mat> &inputs, const vector<mat> &outputs){
	if (inputs.size() != outputs.size()) throw "The number of inputs is not equal to the number of outputs";

	mat encoderOutputMatrix;
	mat decoderOutputMatrix;
	mat decoderErrorMatrix;

	for(int j = 0; j < 10000; ++j){
		for(int i = 0; i < inputs.size(); ++i){
			encoderOutputMatrix = m_encoder.output(inputs[i]);
			decoderOutputMatrix = m_decoder.output(encoderOutputMatrix);

			// cout << "backpropagate on decoder" << endl;
			// cout << "Decoder output matrix" << endl << decoderOutputMatrix << endl;
			// cout << outputs[i] << endl;
			// cout << decoderOutputMatrix - outputs[i] << endl;

			// exit(0);

			decoderErrorMatrix = m_decoder.backpropagate(decoderOutputMatrix, inputs[i]);

			// cout << "backpropagate on encoder" << endl;

			m_encoder.backpropagate(decoderErrorMatrix);
		}
	}

	m_trained = true;
}

const NeuralNetwork & AutoEncoder::getEncoder() const{
	if(m_trained) return m_encoder;
	throw "Cannot get encoder object when training has not taken place";
}

const NeuralNetwork & AutoEncoder::getDecoder() const{
	if(m_trained) return m_decoder;
	throw "Cannot get decoder object when training has not taken place";
}

class MinkowskiDistance{

// (the sum | Xi - Yi |^p)^(1/p)
private:
	int m_power;
public:
	MinkowskiDistance(int power);
	double distance(mat x1, mat x2);

};

MinkowskiDistance::MinkowskiDistance(int power){
	m_power = power;
}

double MinkowskiDistance::distance(mat x1, mat x2){
	if(x1.size() != x2.size()) throw "The vector sizes are not the same";
	return 1.0;
};

class Eugenicist
{
public:
	Eugenicist();
	~Eugenicist();
	
};
int main(int argc, char const *argv[]){

    random_device dev;
    mt19937 rng(dev());

	vector<mat> inputs;
	vector<mat> outputs;

	mat inputMatrix;
	mat outputMatrix;
	// for(int i = 0; i < 100; i++){

	// 	inputMatrix = mat(1, 1);
	// 	outputMatrix = mat(1, 1);

	// 	inputMatrix.ones(1,1);
	// 	outputMatrix.ones(1,1);

	// 	inputs.push_back(inputMatrix);
	// 	outputs.push_back(outputMatrix);
	// }

	mat outputXD;

	inputs.push_back({ {0 , 0} });
	outputs.push_back({ 0 });
	inputs.push_back({ {0 , 1} });
	outputs.push_back({ 1 });
	// inputs.push_back({ {1 , 0} });
	// outputs.push_back({ 1 });

	inputs.push_back({ {1 , 1} });
	outputs.push_back({ 0 });

	inputMatrix = { {0 , 1} };
	uniform_int_distribution<mt19937::result_type> distr(0, inputs.size() -1);

	int num;
	// NeuralNetwork test = {2, 5, 10, 0.5, 1, sigmoid, sigmoidPrime};
	NeuralNetwork test = {2, 1, 100, 0.5, 1, sigmoid, sigmoidPrime};

	// int inputNodes, int hiddenLayers, int hiddenNodes, float learningRate, int outputNodes
	cout << "[" << endl;
	for(int j = 0; j < 1000; ++j){
		num = distr(rng);
		outputXD = test.output(inputs[ num ]);
		// cout << "Input " <<  num << " Expected Output " << outputs[ num ];
		// cout << "Actual " << outputXD << endl;
		cout << "(" << num << "," << abs(outputXD - outputs[ num ])  << ")," << endl;
		test.backpropagate(outputXD, outputs[ num ]);

	}
	cout << "]" << endl;
	// cout << test.output(inputMatrix) << endl;
	// inputMatrix = { {0 , 0} };
	// cout << test.output(inputMatrix) << endl;
	// AutoEncoder AE(2, 1, 0.01, 2, tanh, sech2);
	// AE.train(inputs, outputs);

	// NeuralNetwork encoder = AE.getEncoder();
	// NeuralNetwork decoder = AE.getDecoder();
	// cout << encoder.output(inputMatrix) << endl;
	cout << decoder.output(encoder.output(inputMatrix)) << endl;
	return 0;
}

