package neural;

import data.DataGrid;
import neural.activation.ActivationFunction;
import neural.initializers.WeightInitializer;
import neural.loss.LossFunction;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork implements Serializable
{
    private final List<Layer> layers;
    private final int inputSize;
    private final LossFunction lossFunction;


    public NeuralNetwork(int inputSize, LossFunction lossFunction)
    {
        this.layers = new ArrayList<>();
        this.inputSize = inputSize;
        this.lossFunction = lossFunction;
    }

    public void addLayer(int numberOfNeurons, WeightInitializer initializer, ActivationFunction activationFunction)
    {
        // determina la dimensione dell'input del nuovo layer in base al numero di neuroni dell'ultimo layer
        final int currentInputSize = (layers.isEmpty()) ? inputSize : layers.get(layers.size() - 1).getNeurons().size();
        final Layer layer = new Layer(currentInputSize, numberOfNeurons, initializer, activationFunction);
        layers.add(layer);
    }

    public double[] predict(double[] input)
    {
        return layers
                .stream()
                .reduce(input,
                        (currentInput, layer) -> layer.computeOutput(currentInput),
                        (a, b) -> b); // questo combinatore non serve ma è richiesto dalla reduce
    }

    public void train(List<DataGrid> trainingSet, int epochs, double learningRate)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalLoss = 0.0;

            for (DataGrid sample : trainingSet)
            {
                double[] input = sample.getGrid();
                double[] target = toOneHot(sample.getLabel());

                // Forward pass
                List<double[]> activations = new ArrayList<>(layers.size() + 1);
                activations.add(input);
                double[] current = input;
                for (Layer layer : layers)
                {
                    current = layer.computeOutput(current);  // salva lastInput e lastZ nel layer
                    activations.add(current);
                }

                // Loss
                totalLoss += lossFunction.compute(current, target);

                // Backward pass
                double[] delta = lossFunction.derivative(current, target); // iniziale dL/dA

                for (int i = layers.size() - 1; i >= 0; i--)
                {
                    Layer layer = layers.get(i);
                    double[] prevActivation = activations.get(i); // A_{i-1}

                    delta = layer.backward(delta, prevActivation, learningRate);
                    // delta è ora il gradiente rispetto all'input di questo layer,
                    // cioè serve come errore per il layer precedente
                }
            }

            System.out.printf("Epoch %d - Loss: %.6f%n", epoch + 1, totalLoss / trainingSet.size());
        }
    }

    private double[] toOneHot(int label)
    {
        double[] oneHot = new double[10];
        oneHot[label] = 1.0;
        return oneHot;
    }

}
