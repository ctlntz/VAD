#include "model_weights.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

// Simple activation functions
float relu(float x) {
    return (x > 0) ? x : 0;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Forward pass for a single layer
void forward_layer(const float* input, float* output, const layer_t* layer) {
    // Matrix multiplication: output = weights * input + bias
    for (int i = 0; i < layer->output_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < layer->input_size; j++) {
            sum += layer->weights[i * layer->input_size + j] * input[j];
        }
        output[i] = sum + layer->bias[i];
    }
}

// Full network inference
void model_inference(const float* input, float* output, int input_size) {
    static float layer_outputs[2][256]; // Adjust size based on your network
    int current_buffer = 0;
    
    // Copy input to first buffer
    memcpy(layer_outputs[current_buffer], input, input_size * sizeof(float));
    
    // Process each layer
    for (int layer_idx = 0; layer_idx < network_num_layers; layer_idx++) {
        const layer_t* layer = &network_layers[layer_idx];
        int next_buffer = 1 - current_buffer;
        
        // Forward pass
        forward_layer(layer_outputs[current_buffer], layer_outputs[next_buffer], layer);
        
        // Apply activation (ReLU for hidden layers, sigmoid for output)
        if (layer_idx < network_num_layers - 1) {
            // Hidden layer - apply ReLU
            for (int i = 0; i < layer->output_size; i++) {
                layer_outputs[next_buffer][i] = relu(layer_outputs[next_buffer][i]);
            }
        } else {
            // Output layer - apply sigmoid (or softmax for multi-class)
            for (int i = 0; i < layer->output_size; i++) {
                layer_outputs[next_buffer][i] = sigmoid(layer_outputs[next_buffer][i]);
            }
        }
        
        current_buffer = next_buffer;
    }
    
    // Copy final output
    const layer_t* final_layer = &network_layers[network_num_layers - 1];
    memcpy(output, layer_outputs[current_buffer], final_layer->output_size * sizeof(float));
}

// Example usage
int main() {
    // Example input (adjust size based on your model)
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
    float output[16]; // Adjust size based on your output layer
    
    // Run inference
    model_inference(input, output, 15); // Adjust input size
    
    // Print results
    printf("Model output:\n");
    for (int i = 0; i < 1; i++) { // Adjust based on output size
        printf("Output[%d]: %.6f\n", i, output[i]);
    }
    
    return 0;
}
