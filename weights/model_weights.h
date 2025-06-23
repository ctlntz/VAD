#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#ifdef __cplusplus
extern "C" {
#endif

// Model weights and biases

// layers.0.weight - Shape: [32, 15]
#define LAYERS_0_WEIGHT_SIZE 480
extern const float layers_0_weight[480];

// layers.0.bias - Shape: [32]
#define LAYERS_0_BIAS_SIZE 32
extern const float layers_0_bias[32];

// layers.3.weight - Shape: [8, 32]
#define LAYERS_3_WEIGHT_SIZE 256
extern const float layers_3_weight[256];

// layers.3.bias - Shape: [8]
#define LAYERS_3_BIAS_SIZE 8
extern const float layers_3_bias[8];

// output_layer.weight - Shape: [2, 8]
#define OUTPUT_LAYER_WEIGHT_SIZE 16
extern const float output_layer_weight[16];

// output_layer.bias - Shape: [2]
#define OUTPUT_LAYER_BIAS_SIZE 2
extern const float output_layer_bias[2];

// Network structure information
#define NUM_LAYERS 3

typedef struct {
    const float* weights;
    const float* bias;
    int input_size;
    int output_size;
} layer_t;

// Layer configuration
extern const layer_t network_layers[];
extern const int network_num_layers;


#ifdef __cplusplus
}
#endif

#endif // MODEL_WEIGHTS_H
