import numpy as np
from src.dnn_functions import initialize_parameters_deep, L_model_forward, compute_cost, L_model_backward, update_parameters

def main():
    print("--- Deep Neural Network Test Run ---")
    
    # 1. Giả lập dữ liệu (Input: 12288 features, 209 examples)
    np.random.seed(1)
    X = np.random.randn(12288, 209)
    Y = np.random.randint(0, 2, (1, 209)) # Nhãn nhị phân (0 hoặc 1)
    
    # 2. Cấu hình layers (Input -> Hidden -> ... -> Output)
    layers_dims = [12288, 20, 7, 5, 1] 
    
    # 3. Initialization
    print("Initializing parameters...")
    parameters = initialize_parameters_deep(layers_dims)
    
    # 4. Training Loop (Ví dụ chạy 100 vòng lặp)
    learning_rate = 0.0075
    num_iterations = 100
    
    print(f"Start training for {num_iterations} iterations...")
    for i in range(0, num_iterations):
        # Forward propagation
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost
        cost = compute_cost(AL, Y)
        
        # Backward propagation
        grads = L_model_backward(AL, Y, caches)
        
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if i % 10 == 0:
            print(f"Cost after iteration {i}: {cost}")

    print("Training finished successfully!")

if __name__ == "__main__":
    main()
