public class App {
    public static void main(String[] args) throws Exception {
        
        Activation activation = new Activation("sigmoid");
        Optimizer optimizer = new Optimizer(0.32, 0.9, 0, false);
        LossFunction lossFunction = new LossFunction("MAE");
        Neuron neuron = new Neuron(4);  // Inicializa um neurônio com 3 entradas
        double[] inputs = {0.93, 0.95, 0.97, 0.98};  // Valores de entrada para o neurônio
        double target = 0.9;
        int epochs = 100;
        for (int i = 0; i < epochs; i++) {
            double output = neuron.forward(inputs, activation);  // Calcula a saída do neurônio
            double loss = lossFunction.calculate(output, target);  // Calcula o erro
            double[] newWeights = optimizer.backward(neuron.getWeights(), loss);
            neuron.step(newWeights, optimizer.getLearningRate() * loss);
            System.out.println("Epoca:" + i + " Saida do neuronio: " + output);
        }

    }
}
