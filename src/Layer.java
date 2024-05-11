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
        for (int i = 0; i < neurons.size(); i++) {
            double[] neuronWeights = neurons.get(i).getWeights();
            for (int j = 0; j < neuronWeights.length; j++) {
                weights[j] += neuronWeights[j]; // Add the weights of each neuron to the weights array
            }
        }
        return weights;
    }

    public void step(double[] weights, double gradient) {
        double[] layerWeights = new double[this.numWeights]; // Calculate the number of weights per neuron
        for (int i = 0; i < numNeurons; i++) {
            for (int j = 0; j < input_shape; j++) {
                layerWeights[j] = weights[j + input_shape * i]; // Set the weights of each neuron in the layer
            }
            neurons.get(i).step(layerWeights, gradient);
        }
    }


}
