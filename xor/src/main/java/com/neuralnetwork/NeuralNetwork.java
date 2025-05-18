package com.neuralnetwork;

public class NeuralNetwork {
    private final Layer hiddenLayer;
    private final Layer outputLayer;
    private final double learningRate;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate) {
        this.hiddenLayer = new Layer(hiddenSize, inputSize);
        this.outputLayer = new Layer(outputSize, hiddenSize);
        this.learningRate = learningRate;
    }

    public double[] feedForward(double[] inputs) {
        double[] hiddenOutputs = hiddenLayer.feedForward(inputs);
        return outputLayer.feedForward(hiddenOutputs);
    }

    public void train(double[] inputs, double[] targetOutputs) {
        // Forward pass
        double[] hiddenOutputs = hiddenLayer.feedForward(inputs);
        double[] outputs = outputLayer.feedForward(hiddenOutputs);

        // Calculate output layer errors
        double[] outputErrors = new double[outputs.length];
        for (int i = 0; i < outputs.length; i++) {
            outputErrors[i] = targetOutputs[i] - outputs[i];
        }

        // Calculate hidden layer errors
        double[] hiddenErrors = new double[hiddenLayer.getNeurons().length];
        for (int i = 0; i < hiddenLayer.getNeurons().length; i++) {
            double error = 0;
            for (int j = 0; j < outputLayer.getNeurons().length; j++) {
                error += outputErrors[j] * outputLayer.getNeurons()[j].getWeights()[i];
            }
            hiddenErrors[i] = error;
        }

        // Update weights
        outputLayer.updateWeights(learningRate, outputErrors);
        hiddenLayer.updateWeights(learningRate, hiddenErrors);
    }

    public double getWeightSum() {
        double sum = 0;
        for (Neuron neuron : hiddenLayer.getNeurons()) {
            for (double weight : neuron.getWeights()) {
                sum += weight;
            }
            sum += neuron.getBias();
        }
        for (Neuron neuron : outputLayer.getNeurons()) {
            for (double weight : neuron.getWeights()) {
                sum += weight;
            }
            sum += neuron.getBias();
        }
        return sum;
    }
}
