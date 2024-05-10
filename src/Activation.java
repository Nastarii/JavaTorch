public class Activation {
    
    public String activationFunction;

    public Activation(String activationFunction) {
        this.activationFunction = activationFunction;
        if (!activationFunction.equals("sigmoid") && 
            !activationFunction.equals("tanh") && 
            !activationFunction.equals("relu") && 
            !activationFunction.equals("leakyRelu")
        ) {
            throw new IllegalArgumentException("Invalid activation function");
        }  
    }

    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private static double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    private static double tanh(double x) {
        return Math.tanh(x);
    }

    private static double tanhDerivative(double x) {
        return 1 - Math.pow(x, 2);
    }

    private static double relu(double x) {
        return Math.max(0, x);
    }

    private static double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    private static double leakyRelu(double x) {
        return Math.max(0.01 * x, x);
    }

    private static double leakyReluDerivative(double x) {
        return x > 0 ? 1 : 0.01;
    }

    public double calculate(double x) {
        switch (this.activationFunction) {
            case "sigmoid":
                return sigmoid(x);
            case "sigmoidDerivative":
                return sigmoidDerivative(x);
            case "tanh":
                return tanh(x);
            case "tanhDerivative":
                return tanhDerivative(x);
            case "relu":
                return relu(x);
            case "reluDerivative":
                return reluDerivative(x);
            case "leakyRelu":
                return leakyRelu(x);
            case "leakyReluDerivative":
                return leakyReluDerivative(x);
            default:
                throw new IllegalArgumentException("Invalid activation function");
        }
    }

}
