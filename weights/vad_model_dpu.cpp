#include <iostream>
#include <vector>
#include <memory>
#include <vart/runner.hpp>
#include <vart/tensor_buffer.hpp>
#include <xir/graph/graph.hpp>

class VAD_MODELRunner {
private:
    std::unique_ptr<vart::Runner> runner_;
    std::vector<const xir::Tensor*> input_tensors_;
    std::vector<const xir::Tensor*> output_tensors_;
    
public:
    VAD_MODELRunner(const std::string& model_path) {
        // Load the compiled DPU model
        auto graph = xir::Graph::deserialize(model_path);
        auto subgraph = get_dpu_subgraph(graph.get());
        runner_ = vart::Runner::create_runner(subgraph[0], "run");
        
        // Get input and output tensor specifications
        input_tensors_ = runner_->get_input_tensors();
        output_tensors_ = runner_->get_output_tensors();
        
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "Input tensors: " << input_tensors_.size() << std::endl;
        std::cout << "Output tensors: " << output_tensors_.size() << std::endl;
    }
    
    std::vector<float> predict(const std::vector<float>& input) {
        // Create input tensor buffers
        auto input_tensor_buffers = std::vector<std::unique_ptr<vart::TensorBuffer>>{};
        auto output_tensor_buffers = std::vector<std::unique_ptr<vart::TensorBuffer>>{};
        
        for (auto& tensor : input_tensors_) {
            input_tensor_buffers.push_back(
                std::make_unique<vart::TensorBuffer>(tensor, nullptr));
        }
        
        for (auto& tensor : output_tensors_) {
            output_tensor_buffers.push_back(
                std::make_unique<vart::TensorBuffer>(tensor, nullptr));
        }
        
        // Copy input data
        auto input_data = reinterpret_cast<float*>(
            input_tensor_buffers[0]->data({0, 0}).first);
        std::copy(input.begin(), input.end(), input_data);
        
        // Run inference
        auto v_input_tensor_buffers = std::vector<vart::TensorBuffer*>{};
        auto v_output_tensor_buffers = std::vector<vart::TensorBuffer*>{};
        
        for (auto& tb : input_tensor_buffers) {
            v_input_tensor_buffers.push_back(tb.get());
        }
        for (auto& tb : output_tensor_buffers) {
            v_output_tensor_buffers.push_back(tb.get());
        }
        
        auto job_id = runner_->execute_async(v_input_tensor_buffers, v_output_tensor_buffers);
        runner_->wait(job_id.first, -1);
        
        // Get output data
        auto output_data = reinterpret_cast<float*>(
            output_tensor_buffers[0]->data({0, 0}).first);
        auto output_size = output_tensors_[0]->get_element_num();
        
        return std::vector<float>(output_data, output_data + output_size);
    }
    
private:
    std::vector<const xir::Subgraph*> get_dpu_subgraph(const xir::Graph* graph) {
        auto root = graph->get_root_subgraph();
        auto children = root->children_topological_sort();
        std::vector<const xir::Subgraph*> dpu_subgraphs;
        for (auto& child : children) {
            if (child->get_attr<std::string>("device") == "DPU") {
                dpu_subgraphs.push_back(child);
            }
        }
        return dpu_subgraphs;
    }
};

// Example usage
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.xmodel>" << std::endl;
        return -1;
    }
    
    try {
        // Initialize the model
        VAD_MODELRunner model(argv[1]);
        
        // Example input (adjust based on your model)
        std::vector<float> input = {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        };
        
        // Run inference
        auto output = model.predict(input);
        
        // Print results
        std::cout << "Inference results:" << std::endl;
        for (size_t i = 0; i < output.size(); ++i) {
            std::cout << "Output[" << i << "]: " << output[i] << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
