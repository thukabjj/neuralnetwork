package com.neuralnetwork;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.neuralnetwork.utils.TrainingMetrics;

public class Main {
    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) {
        // Create a neural network for XOR problem
        NeuralNetwork nn = new NeuralNetwork(2, 4, 1, 0.1);
        TrainingMetrics metrics = new TrainingMetrics();

        // Training data for XOR operation
        double[][] trainingInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

        double[][] trainingOutputs = {{0}, {1}, {1}, {0}};

        logger.info("Starting neural network training...");

        int totalEpochs = 20000; // Increased for longer training
        int updateFrequency = 5; // More frequent updates

        for (int epoch = 0; epoch < totalEpochs; epoch++) {
            double totalError = 0;
            int correctPredictions = 0;
            double initialWeightSum = nn.getWeightSum();

            for (int i = 0; i < trainingInputs.length; i++) {
                double[] outputs = nn.feedForward(trainingInputs[i]);
                double error = Math.abs(trainingOutputs[i][0] - outputs[0]);
                totalError += error;

                // Count correct predictions (threshold at 0.5)
                if ((outputs[0] >= 0.5 && trainingOutputs[i][0] == 1)
                        || (outputs[0] < 0.5 && trainingOutputs[i][0] == 0)) {
                    correctPredictions++;
                }

                nn.train(trainingInputs[i], trainingOutputs[i]);
            }

            double finalWeightSum = nn.getWeightSum();
            double weightChange = Math.abs(finalWeightSum - initialWeightSum);
            double averageError = totalError / trainingInputs.length;
            double accuracy = (double) correctPredictions / trainingInputs.length;
            double accuracyPercentage = accuracy * 100;

            // Record metrics every 5 epochs for smoother animation
            if (epoch % updateFrequency == 0) {
                metrics.recordMetrics(averageError, accuracy, weightChange);

                // Log the accuracy percentage
                logger.info(
                        String.format("Epoch %d: Accuracy = %.2f%%", epoch, accuracyPercentage));

                // Add a sleep to slow down training and allow plot updates
                try {
                    Thread.sleep(100); // 100 ms pause per update, adjust as needed
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }

        logger.info("Training completed!");

        // Test the network
        logger.info("\nTesting the neural network:");
        for (int i = 0; i < trainingInputs.length; i++) {
            double[] output = nn.feedForward(trainingInputs[i]);
            logger.info(String.format("Input: %.1f, %.1f | Output: %.3f | Expected: %.0f",
                    trainingInputs[i][0], trainingInputs[i][1], output[0], trainingOutputs[i][0]));
        }
    }
}
