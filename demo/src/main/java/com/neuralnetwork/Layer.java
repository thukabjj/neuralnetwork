package com.neuralnetwork;

public class Layer {
    private final Neuron[] neurons;
    private final int numberOfInputs;

    public Layer(int numberOfNeurons, int numberOfInputs) {
        this.neurons = new Neuron[numberOfNeurons];
        this.numberOfInputs = numberOfInputs;

        for (int i = 0; i < numberOfNeurons; i++) {
            neurons[i] = new Neuron(numberOfInputs);
        }
    }

    public double[] feedForward(double[] inputs) {
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].feedForward(inputs);
        }
        return outputs;
    }

    public void updateWeights(double learningRate, double[] errors) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].updateWeights(learningRate, errors[i]);
        }
    }

    public Neuron[] getNeurons() {
        return neurons;
    }

    public int getNumberOfInputs() {
        return numberOfInputs;
    }
}
