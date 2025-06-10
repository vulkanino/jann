package neural.loss;

import java.io.Serializable;

public class CrossEntropy implements LossFunction, Serializable
{
    @Override
    public double compute(double[] prediction, double[] target)
    {
        double epsilon = 1e-12; // per evitare log(0)
        double loss = 0.0;
        for (int i = 0; i < prediction.length; i++)
            loss -= target[i] * Math.log(prediction[i] + epsilon);
        return loss;
    }

    @Override
    public double[] derivative(double[] prediction, double[] target)
    {
        // Derivata della softmax + cross-entropy semplificata
        double[] grad = new double[prediction.length];
        for (int i = 0; i < prediction.length; i++)
            grad[i] = prediction[i] - target[i];
        return grad;
    }
}