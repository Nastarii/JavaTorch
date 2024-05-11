import java.util.ArrayList;
import java.util.List;

public class Layer {
    
    private int numNeurons;
    private int input_shape;
    private int numWeights;
    private double[] outputs;
    private List<Neuron> neurons = new ArrayList<Neuron>();
    private Activation activation;

    public Layer(int numNeurons, int input_shape, Activation activation) {
        this.numNeurons = numNeurons;
        this.activation = activation;
        this.input_shape = input_shape;
        this.numWeights = numNeurons * input_shape;
        this.outputs = new double[numNeurons];
        for(int i = 0; i < numNeurons; i++) {
            this.neurons.add(new Neuron(input_shape));
        }
    }

    public double[] forward(double[] inputs) {
        for (int i = 0; i < numNeurons; i++) {
            Neuron neuron = neurons.get(i);
            double output = neuron.forward(inputs, this.activation);
            this.outputs[i] = output;
        }
        return this.outputs;
    }

    public double[] showOutputs() {
        return outputs;
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public double[] getWeights() {
        double[] weights = new double[this.numWeights]; // Initialize the weights array with the correct length
        int neuronIndex = 0;
        for (Neuron neuron : neurons) {
            double[] neuronWeights = neuron.getWeights();
            for (int j = 0; j < neuronWeights.length; j++) {
                weights[j + neuronIndex] += neuronWeights[j]; // Add the weights of each neuron to the weights array
            }
            neuronIndex += input_shape;
        }
        return weights;
    }

    public int getNumWeights() {
        return numWeights;
    }

    public void step(double[] weights, double gradient) {

        if (weights.length != this.numWeights) {
            throw new IllegalArgumentException("Invalid number of weights");
        }
        double[] neuronWeights = new double[this.input_shape]; // Calculate the number of weights per neuron
        int neuronIndex = 0;
        for (Neuron neuron : neurons) {
            for (int j = 0; j < input_shape; j++) {
                neuronWeights[j] = weights[j + neuronIndex]; // Set the weights of each neuron in the layer
            }
            neuronIndex += input_shape;
            neuron.step(neuronWeights, gradient);
        }
    }


}
