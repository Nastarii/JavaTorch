public class Neuron {
    
    private int numInputs;
    private double bias;
    private double[] weights;

    // Initialize the Neuron with random weights and bias
    public Neuron(int numInputs) {
        this.numInputs = numInputs;
        this.bias = Math.random();
        this.weights = new double[numInputs];

        for (int i = 0; i < numInputs; i++) {
            weights[i] = Math.random();
        }
    }

    // Transfer Learning
    public Neuron(double bias, double[] weights) {
        this.bias = bias;
        this.weights = weights;
        this.numInputs = weights.length;
    }

    // Get the weights of the Neuron
    public double[] getWeights() {
        return weights;
    }

    // Get the bias of the Neuron
    public double getBias() {
        return bias;
    }

    // Set the bias of the Neuron
    public void setBias(double bias) {
        this.bias = bias;
    }

    public void step(double[] weights, double gradient) {

        if (this.weights.length != weights.length) {
            throw new IllegalArgumentException("Invalid number of weights");
        }

        this.weights = weights;
        this.bias += gradient;
    }

    // Calculate the weighted sum of inputs
    private double calculateWeightedSum(double[] inputs) {
        if (inputs.length != numInputs) {
            throw new IllegalArgumentException("Invalid number of inputs");
        }

        double sum = bias;
        for (int i = 0; i < numInputs; i++) {
            sum += weights[i] * inputs[i];
        }
        sum += bias;
        return sum;
    }

    // Calculate the output of the Neuron
    public double forward(double[] inputs, Activation activationFunction) {
        double weightedSum = calculateWeightedSum(inputs);
        return activationFunction.calculate(weightedSum);
    }

}
