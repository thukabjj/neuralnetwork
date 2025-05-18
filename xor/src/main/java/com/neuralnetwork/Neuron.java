package com.neuralnetwork;

import java.util.Arrays;
import java.util.Random;

public class Neuron {
    private final double[] weights;
    private final double bias;
    private double output;
    private double delta;
    private final Random random = new Random();

    // Store the last inputs for use in backpropagation
    private double[] lastInputs;

    public Neuron(int numberOfInputs) {
        // Initialize weights with random values between -1 and 1
        this.weights = new double[numberOfInputs];
        for (int i = 0; i < numberOfInputs; i++) {
            this.weights[i] = random.nextDouble() * 2 - 1;
        }
        // Initialize bias with a random value between -1 and 1
        this.bias = random.nextDouble() * 2 - 1;
    }

    public double feedForward(double[] inputs) {
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException(
                "Number of inputs must match number of weights. " +
                "Expected: " + weights.length + ", Got: " + inputs.length
            );
        }

        // Store inputs for backpropagation
        this.lastInputs = Arrays.copyOf(inputs, inputs.length);

        // Calculate weighted sum
        double sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        sum += bias;

        // Apply activation function (sigmoid)
        this.output = sigmoid(sum);
        return this.output;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    public void updateWeights(double learningRate, double error) {
        // Calculate delta for this neuron
        this.delta = error * sigmoidDerivative(output);

        // Update weights
        for (int i = 0; i < weights.length; i++) {
            weights[i] += learningRate * delta * lastInputs[i];
        }
    }

    public double getOutput() {
        return output;
    }

    public double[] getWeights() {
        return Arrays.copyOf(weights, weights.length);
    }

    public double getDelta() {
        return delta;
    }

    public double getBias() {
        return bias;
    }

    @Override
    public String toString() {
        return "Neuron{" +
               "weights=" + Arrays.toString(weights) +
               ", bias=" + bias +
               ", output=" + output +
               '}';
    }
}
