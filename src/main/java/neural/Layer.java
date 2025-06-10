package neural;

import neural.activation.ActivationFunction;
import neural.initializers.WeightInitializer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Layer implements Serializable
{
    private final List<Neuron> neurons;

    public Layer(int inputSize, int neurons, WeightInitializer initializer, ActivationFunction activationFunction)
    {
        this.neurons = new ArrayList<>(neurons);
        for (int i = 0; i < neurons; i++)
            this.neurons.add(new Neuron(inputSize, neurons, initializer, activationFunction));
    }

    /**
     * Calcola l'output di questo layer dato l'input.
     */
    public double[] computeOutput(double[] input)
    {
        double[] output = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++)
            output[i] = neurons.get(i).computeOutput(input);

        return output;
    }

    public double[] backward(double[] dLoss_dA, double[] prevActivations, double learningRate)
    {
        int numInputs = prevActivations.length;
        int numNeurons = neurons.size();
        double[] dLoss_dA_prev = new double[numInputs]; // da restituire

        for (int j = 0; j < numNeurons; j++)
        {
            Neuron neuron = neurons.get(j);

            double z = neuron.getLastZ(); // z_j = somma pesata prima di attivazione
            double da_dz = neuron.getActivationFunction().derivative(z); // ∂A/∂Z

            double dLoss_dZ = dLoss_dA[j] * da_dz; // ∂L/∂Z = ∂L/∂A * ∂A/∂Z

            neuron.updateWeights(dLoss_dZ, learningRate);

            // Calcola contributo all’errore del layer precedente: dL/dA_prev = somma( w_jk * dL/dZ_j )
            double[] weights = neuron.getWeights();
            for (int k = 0; k < weights.length; k++)
            {
                dLoss_dA_prev[k] += weights[k] * dLoss_dZ;
            }
        }

        return dLoss_dA_prev;
    }

    public List<Neuron> getNeurons()
    {
        return neurons;
    }
}