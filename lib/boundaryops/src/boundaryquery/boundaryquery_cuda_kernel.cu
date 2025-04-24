#include "../cuda_util.h"
#include "boundaryquery_cuda_kernel.h"

typedef struct
{
    int data[100];
    int front, rear;
}Queue;

__device__ bool Set_NULL(Queue &Q)
{
	Q.front = -1;
	Q.rear = -1;
	return true;
}

__device__ bool Is_NULL(Queue Q)
{
	if (Q.front == Q.rear)
	{
		return true;  
	}
	return false;
}

__device__ bool En_Queue(Queue &Q, int a)
{
	if ((Q.rear - Q.front) >= 100-1)
	{
		// cout<<"The queue is full~";
		return false;
	}
	Q.rear += 1;
	Q.data[Q.rear] = a;
	return true;
}

__device__ bool De_Queue(Queue &Q)
{
	if (Is_NULL(Q))
	{
		// cout<<"The queue is empty~";
		return false;
	}
	Q.front += 1;
	return true;
}


__device__ int front_element(Queue Q)
{
	if (Is_NULL(Q))
	{
		// cout<<"The queue is empty~";
		return NULL;
	}
	return Q.data[Q.rear];
}

__device__ void swap_float(float *x, float *y)
{
    float tmp = *x;
    *x = *y;
    *y = tmp;
}


__device__ void swap_int(int *x, int *y)
{
    int tmp = *x;
    *x = *y;
    *y = tmp;
}


__device__ void reheap(float *dist, int *idx, int k)
{
    int root = 0;
    int child = root * 2 + 1;
    while (child < k)
    {
        if(child + 1 < k && dist[child+1] > dist[child])
            child++;
        if(dist[root] > dist[child])
            return;
        swap_float(&dist[root], &dist[child]);
        swap_int(&idx[root], &idx[child]);
        root = child;
        child = root * 2 + 1;
    }
}


__device__ void heap_sort(float *dist, int *idx, int k)
{
    int i;
    for (i = k - 1; i > 0; i--)
    {
        swap_float(&dist[0], &dist[i]);
        swap_int(&idx[0], &idx[i]);
        reheap(dist, idx, i);
    }
}


__device__ int get_bt_idx(int idx, const int *offset)
{
    int i = 0;
    while (1)
    {
        if (idx < offset[i])
            break;
        else
            i++;
    }
    return i;
}


__global__ void boundaryquery_cuda_kernel(int m, int nsample, int b, const float *__restrict__ xyz, const float *__restrict__ new_xyz, const int *__restrict__ offset, const int *__restrict__ new_offset, int *__restrict__ idx, float *__restrict__ dist2, const int *__restrict__ edges, const int *__restrict__ boundary){
    // input: xyz (n, 3) new_xyz (m, 3)
    // output: idx (m, nsample) dist2 (m, nsample)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (pt_idx >= m) return;

    
    new_xyz += pt_idx * 3;
    idx += pt_idx * nsample;
    dist2 += pt_idx * nsample;
    // edges += pt_idx * 8 
    int bt_idx = get_bt_idx(pt_idx, new_offset);
    int start;
    if (bt_idx == 0)
        start = 0;
    else
        start = offset[bt_idx - 1];
    int end = offset[bt_idx];

    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    float best_dist[100];
    int best_idx[100];
    int search_idx[100];    // Boundary-guided sampling strategy results index
    for(int i = 0; i < nsample; i++){
        best_dist[i] = 1e10;
        best_idx[i] = start;
        search_idx[i] = start;
    }

    
    Queue idx_queue;    
    Set_NULL(idx_queue);
    En_Queue(idx_queue, pt_idx);
    search_idx[0] = pt_idx;
    // idx_queue.push(pt_idx);
    int s_idx = 1;      //(1~7)
    bool flag = false;  // Select knn to search inside small instance segments

    if (boundary[pt_idx] == 1)
        flag = true;     // Select knn to search for boundary points

    while (s_idx < nsample){
        int current_idx;
        if (flag)
            break;
        
        if (!Is_NULL(idx_queue)){   
            current_idx = front_element(idx_queue);
            De_Queue(idx_queue);
        }else{
            flag = true;
            break;
        }

        int count = edges[current_idx * b + 0]; 
        for(int j = 1; j <= count; j++){
            int p_idx = edges[current_idx * b + j];     
            bool flag1 = true;      
            if (p_idx == -1)
                break;      
            if (boundary[p_idx] == 0){
                for(int k = 0; k < nsample; k++){
                    if (search_idx[k] == p_idx){
                        flag1 = false;      // The point has been selected
                        break;
                    }
                }
                if(flag1){
                    // idx_queue.push(p_idx);
                    En_Queue(idx_queue, p_idx);     // Non-boundary points join seed point queue
                    search_idx[s_idx] = p_idx;
                    s_idx++;
                }
            }
                
        }
    }

    if (flag){      
        // Select knn in boundary point or small instance
        for(int i = start; i < end; i++){
            float x = xyz[i * 3 + 0];
            float y = xyz[i * 3 + 1];
            float z = xyz[i * 3 + 2];
            float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
            if (d2 < best_dist[0]){
                best_dist[0] = d2;
                best_idx[0] = i;
                reheap(best_dist, best_idx, nsample);
            }
        }
        heap_sort(best_dist, best_idx, nsample);
        for(int i = 0; i < nsample; i++){
            idx[i] = best_idx[i];
            dist2[i] = best_dist[i];
        }

    }else{
        for(int i = 0; i < nsample; i++){
            idx[i] = search_idx[i];
            dist2[i] = best_dist[i];
        }
    }

}


void boundaryquery_cuda_launcher(int m, int nsample, int b, const float *xyz, const float *new_xyz, const int *offset, const int *new_offset, int *idx, float *dist2, const int *edges, const int *boundary) {
    // input: new_xyz: (m, 3), xyz: (n, 3), idx: (m, nsample)
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    boundaryquery_cuda_kernel<<<blocks, threads, 0>>>(m, nsample, b, xyz, new_xyz, offset, new_offset, idx, dist2, edges, boundary);
}