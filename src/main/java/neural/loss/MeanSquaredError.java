package neural.loss;

import java.io.Serializable;

/**
 * La MSE è una funzione di loss che misura quanto è lontano l’output della rete da quello che dovrebbe essere (il "target").
 *
 * Per il metodo compute, si somma il quadrato della differenza per tutti i valori, poi si fa la media.
 * Si fa il quadrato (da cui il nome squared) per penalizzare più fortemente gli errori grandi ed evitare che errori positivi e negativi si annullino tra loro.
 *
 * Per il metodo derivative, si calcola la derivata della funzione MSE rispetto all’output della rete.
 * Serve per capire quanto e in che direzione correggere l’output della rete per avvicinarlo al target.
 */
public class MeanSquaredError implements LossFunction, Serializable
{
    @Override
    public double compute(double[] predicted, double[] expected)
    {
        checkInputs(predicted, expected);

        double sum = 0.0;
        for (int i = 0; i < predicted.length; i++)
        {
            double diff = predicted[i] - expected[i];
            sum += diff * diff;
        }
        return sum / predicted.length;
    }

    @Override
    public double[] derivative(double[] predicted, double[] expected)
    {
        checkInputs(predicted, expected);

        double[] grad = new double[predicted.length];
        for (int i = 0; i < predicted.length; i++)
            grad[i] = 2 * (predicted[i] - expected[i]) / predicted.length;

        return grad;
    }

    private void checkInputs(double[] predicted, double[] expected)
    {
        if ( predicted.length != expected.length )
            throw new IllegalArgumentException("Dimensioni diverse tra predicted e expected");
    }
}
