#include <iostream>

__global__ void matmul(float* A, float* B, float* C, int m, int n, int k) {
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    if (x<m && y<k) {
        float sum = 0.0f;
        for (int i = 0; i<n; i++) {
            sum+=A[x*n + i]*B[i*k + y];
        }
        C[x*k + y] = sum;
    }

    
}

int main () {

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int m = 1000;
    int n = 1000;
    int k = 1000;

    float* cpuA = new float[m*n]{};
    float* cpuB = new float[n*k]{};
    float* cpuC = new float[m*k]{};

    cpuA[0]=5; cpuA[1] = -3; cpuA[2] = 1; cpuA[3] = 2;
    cpuB[0]=2; cpuB[1] = 3; cpuB[2] = 4; cpuB[3] = -1;


    float* gpuA = nullptr;
    float* gpuB = nullptr;
    float* gpuC = nullptr;

    cudaMalloc(&gpuA, m*n*sizeof(float));
    cudaMalloc(&gpuB, n*k*sizeof(float));
    cudaMalloc(&gpuC, m*k*sizeof(float));

    cudaMemcpy(gpuA,cpuA,m*n*sizeof(float),cudaMemcpyDefault);
    cudaMemcpy(gpuB,cpuB,n*k*sizeof(float),cudaMemcpyDefault);
    cudaMemcpy(gpuC,cpuC,m*k*sizeof(float),cudaMemcpyDefault);

    
    dim3 block(16,16);
    dim3 grid((m+16-1)/16,(k+16-1)/16);
    matmul<<<grid,block>>>(gpuA,gpuB,gpuC,m,n,k);
    cudaDeviceSynchronize(); //stops cpu thread until matmul finishes executing bc matmul is asynch function

    cudaEventRecord(start);
    matmul<<<grid,block>>>(gpuA,gpuB,gpuC,m,n,k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout<<"elapsed time: "<<ms<<" miliseconds"<<std::endl;

    cudaMemcpy(cpuC,gpuC,m*k*sizeof(float),cudaMemcpyDefault);


    // for (int i = 0; i<m*k; i++) {
    //     std::cout<<cpuC[i]<<std::endl;
    // }

    delete cpuA[];
    delete cpuB[];
    delete cpuC[];
    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);

    return 0;
}





