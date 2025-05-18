# Java Neural Network XOR Example

This project demonstrates a simple feedforward neural network implemented in pure Java (Java 24), capable of learning the XOR function. It features real-time training visualization using XChart and detailed logging with SLF4J and Logback.

---

## Features

- **Pure Java implementation**: No deep learning frameworks required.
- **Customizable architecture**: Easily change the number of layers and neurons.
- **Real-time training visualization**: See error, accuracy, and weight changes as the network learns.
- **Detailed logging**: Training progress and results are logged to both console and file.
- **Educational**: Well-commented code for learning and experimentation.

---

## Requirements

- Java 24 or later
- Maven 3.6+
- A modern IDE (IntelliJ IDEA, Eclipse, VS Code, etc.)

---

## Setup

1. **Clone the repository** (or copy the project files):

    ```sh
    git clone <your-repo-url>
    cd <your-project-directory>
    ```

2. **Install dependencies**
   The project uses Maven for dependency management.
   The key dependencies are:
   - [XChart](https://knowm.org/open-source/xchart/) (for plotting)
   - [SLF4J](http://www.slf4j.org/) and [Logback](https://logback.qos.ch/) (for logging)
   - JUnit (for testing)

   Your `pom.xml` should include:

    ```xml
    <dependency>
        <groupId>org.knowm.xchart</groupId>
        <artifactId>xchart</artifactId>
        <version>3.8.6</version>
    </dependency>
    <dependency>
        <groupId>org.slf4j</groupId>
        <artifactId>slf4j-api</artifactId>
        <version>2.0.13</version>
    </dependency>
    <dependency>
        <groupId>ch.qos.logback</groupId>
        <artifactId>logback-classic</artifactId>
        <version>1.4.14</version>
    </dependency>
    ```

3. **Build the project**:

    ```sh
    mvn clean install
    ```

4. **Run the application**:

    ```sh
    mvn exec:java -Dexec.mainClass="com.neuralnetwork.Main"
    ```

---

## How It Works

### Neural Network Structure

- **Input Layer**: 2 neurons (for the two XOR inputs)
- **Hidden Layer**: Configurable (default: 4 neurons)
- **Output Layer**: 1 neuron (for the XOR result)

### Training

- The network is trained on all possible XOR input pairs: (0,0), (0,1), (1,0), (1,1).
- Uses the sigmoid activation function and backpropagation for learning.
- Training progress is visualized in real time.

### Visualization

- **Error (red line)**: Average error per epoch.
- **Accuracy (blue line)**: Fraction of correct predictions per epoch.
- **Weight Change (green line)**: How much the weights change per epoch.
- **Threshold Line (gray, dashed)**: 50% accuracy reference.

### Logging

- Training progress, error, accuracy, and weight changes are logged to both the console and a log file (`logs/neural-network-training.log`).

---

## Customization

### Change Hidden Layer Size

In `Main.java`:

```java
NeuralNetwork nn = new NeuralNetwork(2, 4, 1, 0.1);
//                ^ 4 is the number of hidden neurons. Try 2, 3, 5, etc.
```

### Change Learning Rate

```java
NeuralNetwork nn = new NeuralNetwork(2, 4, 1, 0.1);
// Last parameter is the learning rate. Try 0.05, 0.01, etc.
```

### Change Training Duration

```java
int totalEpochs = 20000; // Increase or decrease as needed
```

### Change Visualization Update Frequency

```java
int updateFrequency = 5; // Lower = more frequent plot updates
```

### Change Sleep Time (Visualization Speed)

```java
Thread.sleep(100); // Increase for slower, more visible updates
```

---

## Example Output

- A window will open showing the training progress in real time.
- The log will show lines like:

    ```
    Epoch 100: Error = 0.3566, Accuracy = 1.00, Weight Change = 0.006316
    Input: 0.0, 1.0 | Output: 0.998 | Expected: 1
    ```

---

## How to Experiment

- **Try different hidden layer sizes** and see how the network learns.
- **Change the learning rate** to see how it affects convergence.
- **Modify the training data** to try other logic gates (AND, OR, etc.).
- **Add more layers** for more complex problems.

---

## How the Hidden Layer Size is Chosen

- For XOR, **2 neurons** in the hidden layer is the minimum required.
- More neurons can make learning easier, but too many can cause overfitting.
- For more complex problems, experiment with different sizes and use validation data to choose the best.

---

## File Structure

```
src/
  main/
    java/
      com/
        neuralnetwork/
          Main.java
          NeuralNetwork.java
          Layer.java
          Neuron.java
        neuralnetwork/
          utils/
            TrainingMetrics.java
    resources/
      logback.xml
pom.xml
```

---

## References

- [XChart Documentation](https://knowm.org/open-source/xchart/)
- [SLF4J Documentation](http://www.slf4j.org/manual.html)
- [Logback Documentation](https://logback.qos.ch/manual/index.html)
- [Neural Networks and Deep Learning (Book)](http://neuralnetworksanddeeplearning.com/)

---

## License

This project is for educational purposes.
Feel free to use, modify, and share!

---

**Enjoy experimenting with neural networks in Java!**
If you have questions or want to extend the project, open an issue or contribute.

---

Let me know if you want a shorter version or more details on any section!
