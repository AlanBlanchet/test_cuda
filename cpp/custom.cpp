#include <iostream>
#include <chrono>

#define N 1024 // Ensure this matches the value in mm.cu

extern "C" void matrixMulCUDA(float *h_A, float *h_B, float *h_C);

void matrixMulCPU(float *A, float *B, float *C, int n)
{
    // Matrix multiply on the CPU
    for (int row = 0; row < n; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            float value = 0;
            for (int k = 0; k < n; ++k)
            {
                // Linear combination of A and B
                value += A[row * n + k] * B[k * n + col];
            }
            // Store result in C
            C[row * n + col] = value;
        }
    }
}

void initializeMatrix(float *matrix, int size)
{
    for (int i = 0; i < size; ++i)
    {
        matrix[i] = 1.0f; // Simple initialization for testing
    }
}

int main()
{
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C_CPU = new float[N * N];
    float *h_C_CUDA = new float[N * N];

    initializeMatrix(h_A, N * N);
    initializeMatrix(h_B, N * N);

    // CPU Matrix Multiplication
    auto start = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_C_CPU, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "CPU Matrix multiplication completed in " << duration.count() << " ms." << std::endl;

    // CUDA Matrix Multiplication
    start = std::chrono::high_resolution_clock::now();
    matrixMulCUDA(h_A, h_B, h_C_CUDA);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "CUDA Matrix multiplication completed in " << duration.count() << " ms." << std::endl;

    // Verify the result
    bool correct = true;
    for (int i = 0; i < N * N; ++i)
    {
        if (h_C_CPU[i] != h_C_CUDA[i])
        {
            std::cerr << "Mismatch at index " << i << ": CPU=" << h_C_CPU[i] << ", CUDA=" << h_C_CUDA[i] << std::endl;
            correct = false;
            break;
        }
    }

    if (correct)
    {
        std::cout << "Matrix multiplication results are correct!" << std::endl;
    }
    else
    {
        std::cerr << "Matrix multiplication results are incorrect!" << std::endl;
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_CPU;
    delete[] h_C_CUDA;

    return 0;
}
