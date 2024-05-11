import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    
    private List<Layer> layers;
    private LossFunction lossFunction;
    private Optimizer optimizer;
    private int epochs;
    private int numWeights = 0;

    public NeuralNetwork(LossFunction lossFunction, Optimizer optimizer, int epochs) {
        this.layers = new ArrayList<Layer>();
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.epochs = epochs;
    }

    public void Sequential(Layer... layers) {
        for (Layer layer : layers) {
            this.numWeights += layer.getNumWeights();
            this.layers.add(layer);
        }
    }

    public double[] forward(double[] inputs) {
        double[] outputs = inputs;
        for (Layer layer : layers) {
            outputs = layer.forward(outputs);
        }
        return outputs;
    }

    public double[] getWeights() {
        int layerIndex = 0;
        double[] weights = new double[this.numWeights]; // Initialize the weights array with the correct length
        for (Layer layer : layers) {
            double[] neuronWeights = layer.getWeights();
            for (int j = 0; j < neuronWeights.length; j++) {
                weights[j + layerIndex] += neuronWeights[j]; // Add the weights of each neuron to the weights array
            }
            layerIndex += neuronWeights.length;
        }
        return weights;
    }

    public void step(double[] newWeights, double gradient) {
        int layerIndex = 0;
        for (Layer layer : layers) {
            int numWeights = layer.getNumWeights();
            double[] weights = new double[numWeights];
            for (int i = 0; i < numWeights; i++) {
                weights[i] = newWeights[i + layerIndex];
            }
            layerIndex += numWeights;
            layer.step(weights, gradient);
        }
    }

    public void train(double[] inputs, double target) {
        for (int i = 0; i < this.epochs; i++) {

            double[] outputs = forward(inputs);

            double loss = lossFunction.calculate(outputs[0], target);

            double[] currentWeights = getWeights();
            double[] newLayerWeights = this.optimizer.backward(currentWeights, loss);

            step(newLayerWeights, optimizer.getLearningRate() * loss); 

            System.out.println("Epoch: " + i + " Output: " + outputs[0]);
        }
    }
}
