package neural.loss;

public interface LossFunction
{
    // calcola il valore del loss
    double compute(double[] predicted, double[] expected);

    // restituisce il gradiente del loss rispetto all’output della rete (necessario per la backpropagation)
    double[] derivative(double[] predicted, double[] expected);
}