package com.neuralnetwork.utils;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.util.ArrayList;
import java.util.List;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import org.knowm.xchart.XChartPanel;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.XYSeries.XYSeriesRenderStyle;
import org.knowm.xchart.style.Styler;
import org.knowm.xchart.style.markers.SeriesMarkers;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TrainingMetrics {
    private static final Logger logger = LoggerFactory.getLogger(TrainingMetrics.class);

    private final List<Double> errors;
    private final List<Integer> epochs;
    private final List<Double> accuracies;
    private final List<Double> weightChanges;
    private int currentEpoch;

    private final XYChart chart;
    private final XChartPanel<XYChart> chartPanel;
    private final JFrame frame;

    public TrainingMetrics() {
        // Initialize lists with initial values
        this.errors = new ArrayList<>();
        this.epochs = new ArrayList<>();
        this.accuracies = new ArrayList<>();
        this.weightChanges = new ArrayList<>();
        this.currentEpoch = 0;

        // Add initial point to avoid empty series
        this.epochs.add(0);
        this.errors.add(0.0);
        this.accuracies.add(0.0);
        this.weightChanges.add(0.0);

        // Initialize chart
        chart = new XYChartBuilder()
            .width(800)
            .height(600)
            .title("Neural Network Training Progress")
            .xAxisTitle("Epoch")
            .yAxisTitle("Value")
            .build();

        // Customize chart
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeriesRenderStyle.Line);
        chart.getStyler().setChartTitleVisible(true);
        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNE);
        chart.getStyler().setXAxisTicksVisible(true);
        chart.getStyler().setYAxisTicksVisible(true);
        chart.getStyler().setPlotGridLinesVisible(true);
        chart.getStyler().setPlotBackgroundColor(Color.WHITE);
        chart.getStyler().setChartBackgroundColor(Color.WHITE);
        chart.getStyler().setPlotBorderVisible(false);
        chart.getStyler().setYAxisMax(1.0);
        chart.getStyler().setYAxisMin(0.0);

        // Add series with different colors and styles
        XYSeries errorSeries = chart.addSeries("Error", epochs, errors);
        errorSeries.setMarker(SeriesMarkers.NONE);
        errorSeries.setLineColor(Color.RED);
        errorSeries.setLineStyle(new BasicStroke(2.0f));

        XYSeries accuracySeries = chart.addSeries("Accuracy", epochs, accuracies);
        accuracySeries.setMarker(SeriesMarkers.NONE);
        accuracySeries.setLineColor(Color.BLUE);
        accuracySeries.setLineStyle(new BasicStroke(2.0f));

        XYSeries weightSeries = chart.addSeries("Weight Change", epochs, weightChanges);
        weightSeries.setMarker(SeriesMarkers.NONE);
        weightSeries.setLineColor(Color.GREEN);
        weightSeries.setLineStyle(new BasicStroke(2.0f));

        // Add reference line for 50% threshold
        List<Double> refLine = new ArrayList<>();
        refLine.add(0.5);
        refLine.add(0.5);
        List<Integer> refEpochs = new ArrayList<>();
        refEpochs.add(0);
        refEpochs.add(1);
        XYSeries refSeries = chart.addSeries("Threshold (50%)", refEpochs, refLine);
        refSeries.setMarker(SeriesMarkers.NONE);
        refSeries.setLineColor(Color.GRAY);
        refSeries.setLineStyle(new BasicStroke(1.5f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{5}, 0));

        // Create chart panel
        chartPanel = new XChartPanel<>(chart);

        // Create frame
        frame = new JFrame("Neural Network Training Progress");
        frame.setLayout(new BorderLayout());
        frame.add(chartPanel, BorderLayout.CENTER);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(1000, 600);
        frame.setLocationRelativeTo(null); // Center on screen
        frame.setVisible(true);
    }

    public void recordMetrics(double error, double accuracy, double weightChange) {
        currentEpoch++;

        epochs.add(currentEpoch);
        errors.add(error);
        accuracies.add(accuracy);
        weightChanges.add(weightChange);

        logger.info(String.format("Epoch %d: Error = %.4f, Accuracy = %.2f, Weight Change = %.6f",
            currentEpoch, error, accuracy, weightChange));

        updateChart();
    }

    private void updateChart() {
        SwingUtilities.invokeLater(() -> {
            try {
                // Update main series
                chart.updateXYSeries("Error", epochs, errors, null);
                chart.updateXYSeries("Accuracy", epochs, accuracies, null);
                chart.updateXYSeries("Weight Change", epochs, weightChanges, null);

                // Update reference line to span the current epoch range
                List<Integer> refEpochs = List.of(0, currentEpoch);
                List<Double> refLine = List.of(0.5, 0.5);
                chart.updateXYSeries("Threshold (50%)", refEpochs, refLine, null);

                chartPanel.revalidate();
                chartPanel.repaint();

                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
    }

    public double getLatestError() {
        return errors.isEmpty() ? 0.0 : errors.get(errors.size() - 1);
    }

    public double getLatestAccuracy() {
        return accuracies.isEmpty() ? 0.0 : accuracies.get(accuracies.size() - 1);
    }
}
