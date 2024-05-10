public class Optimizer {
    
    private double learningRate = 0.01;
    private double momentum = 0.9;
    private double decay = 0;
    private boolean nesterov = false;

    public Optimizer(double learningRate, double momentum, double decay, boolean nesterov) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.decay = decay;
        this.nesterov = nesterov;
    }

    public double[] backward(double[] weigths, double gradient) {
        
        double lr = this.learningRate;
        if (this.decay > 0) {
            lr *= 1.0 / (1.0 + this.decay);
        }

        // Momentum
        double[] newWeigths = weigths.clone();
        double[] velocity = new double[weigths.length];
        for (int i = 0; i < weigths.length; i++) {
            velocity[i] = this.momentum * velocity[i] - lr * gradient;

            if (this.nesterov) {
                newWeigths[i] += this.momentum * velocity[i] - lr * gradient;
            } else {
                newWeigths[i] += velocity[i];
            }
        }

        return newWeigths;
    }

    public double getLearningRate() {
        return learningRate;
    }

}
