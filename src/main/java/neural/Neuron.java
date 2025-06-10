package neural;

import neural.activation.ActivationFunction;
import neural.initializers.WeightInitializer;

import java.io.Serializable;
import java.util.Random;

public class Neuron implements Serializable
{
    private static final Random random = new Random();

    private double bias;
    private final double[] weights;
    private ActivationFunction activationFunction;

    // Per la backpropagation
    private double[] lastInput;
    private double lastZ;


    public Neuron(int inputSize, int outputSize, WeightInitializer initializer, ActivationFunction activationFunction)
    {
        this.bias = random.nextDouble() - 0.5; // Inizializza bias casuale tra -0.5 e 0.5.
        this.weights = initializer.apply(inputSize, outputSize);
        this.activationFunction = activationFunction;
    }

    public double computeOutput(double[] inputs)
    {
        checkInputs(inputs);

        this.lastInput = inputs;
        this.lastZ = bias;
        for (int i = 0; i < inputs.length; i++)
            lastZ += inputs[i] * weights[i];

        return activationFunction.apply(lastZ);
    }

    public void updateWeights(double gradient, double learningRate)
    {
        for (int i = 0; i < weights.length; i++)
            weights[i] -= learningRate * gradient * lastInput[i];

        bias -= learningRate * gradient;
    }

    public double[] getWeights()
    {
        return weights;
    }

    public double getLastZ()
    {
        return lastZ;
    }

    private void checkInputs(double[] inputs)
    {
        if ( inputs.length != weights.length )
            throw new IllegalArgumentException("La dimensione degli input non corrisponde a quella dei pesi");
    }

    public ActivationFunction getActivationFunction()
    {
        return activationFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction)
    {
        this.activationFunction = activationFunction;
    }
}
