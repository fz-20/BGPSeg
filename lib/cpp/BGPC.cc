// Adapted from https://github.com/hjwdzh/PrimitiveNet/blob/main/src/cpp/regiongrow.cc
// Modified by Zheng Fang

#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <map>
#include <vector>
#include <cstring>
#include <unordered_map>
#include <cmath>

extern "C" {

float distance(float* v1, float* v2, int numC) {
	
	float d = 0.0;
	for (int i = 0; i < numC; ++i) {
		d += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	}
	float result = std::sqrt(d);
	// std::cout << result << std::endl;
	return result;
}


void Boundary_guided_primitive_clustering(float* embedding, int numC, int* boundary, int* F, int numV, int numF, int* face_labels, int* output_mask, float score_thres, int* output_label) {
	/*
	embedding: embedding of vertices, numV * numC
	boundary: boundary predictions
    output_label: output clustering results
	*/
	std::vector<std::vector<float>> point_embed(numV,std::vector<float>(numC));
	for(int i=0;i<numV;i++){
		for(int j=0;j<numC;j++){
			point_embed[i][j]=embedding[i*numC+j];
		}
	}

	std::vector<std::unordered_map<int, int> > v_neighbors(numV);
	for (int i = 0; i < numF; ++i) {
		for (int j = 0; j < 3; ++j) {
			int b = boundary[i * 3 + j];	// 0 or 1
			int v0 = F[i * 3 + j];
			int v1 = F[i * 3 + (j + 1) % 3];	
			v_neighbors[v0][v1] = b;
			v_neighbors[v1][v0] = b;	
		}
	}

	// for (auto& nv : v_neighbors[0]) {
	// 	std::cout << nv.first << std::endl;
	// }

	std::vector<int> mask(numV, -2);	// Initialize the clustering label for each vertex to -2
	std::vector<int> mask_face(numV, -2);	
	std::vector<int> boundary_neighbors(numV, 0);	
	int num_boundary = 0;
	for (int i = 0; i < numV; ++i) {
		int is_boundary = 0;	
		for (auto& info : v_neighbors[i]) {
			if (info.second == 1) {
				is_boundary += 1;
				if (boundary_neighbors[i] == 0) {
					boundary_neighbors[i] = 1;
				}
			}
		}
		if (is_boundary >= v_neighbors[i].size() * score_thres) {
			mask[i] = -1;
			mask_face[i] = -1;
			num_boundary += 1;
		}	// Mark boundary vertex
	}

	if (output_mask) {	
		int b = 0;
		for (int i = 0; i < numV; ++i) {
			output_mask[i] = (mask[i] == -1) ? 1 : 0;	
			b += output_mask[i];
		}
	}

	int num_labels = 0;	// Number of clusters
	for (int i = 0; i < mask.size(); ++i) {
		if (mask[i] == -2) {
			std::queue<int> q;
			q.push(i);
			mask[i] = num_labels;
			mask_face[i] = num_labels;
			while (!q.empty()) {
				int v = q.front();
				q.pop();
				for (auto& nv : v_neighbors[v]) {
					if (nv.second == 0 && mask[nv.first] == -2) {
						if (boundary_neighbors[v] == 1) {
                            // pairwise similarity discrimination
							if (distance(point_embed[v].data(),point_embed[nv.first].data(),numC)<0.1){
								mask[nv.first] = num_labels;
								mask_face[nv.first] = num_labels;
								q.push(nv.first);
							}
						} else {
							mask[nv.first] = num_labels;
							mask_face[nv.first] = num_labels;
							q.push(nv.first);
						}
						
					}
					if (nv.second == 1 && mask[nv.first] == -1) {
						mask[nv.first] = num_labels;
					}
				}
			}
			num_labels += 1;	// Starting from the current vertex, breadth-first search
		}
	}

	for (int i = 0; i < numF; ++i) {
		int label = -1;
		for (int j = 0; j < 3; ++j) {
			if (mask_face[F[i * 3 + j]] >= 0) {
				label = mask_face[F[i * 3 + j]];
				break;
			}
		}
		face_labels[i] = label;
	}

	for (int i = 0; i < numV; ++i) {
		output_label[i] = mask[i];
	}
}

void primitive_clustering(int* boundary, int* F, int numV, int numF, int* face_labels, int* output_mask, float score_thres, int* output_label) {
	std::vector<std::unordered_map<int, int> > v_neighbors(numV);
	for (int i = 0; i < numF; ++i) {
		for (int j = 0; j < 3; ++j) {
			int b = boundary[i * 3 + j];	
			int v0 = F[i * 3 + j];
			int v1 = F[i * 3 + (j + 1) % 3];	
			v_neighbors[v0][v1] = b;
			v_neighbors[v1][v0] = b;	
		}
	}

	// for (auto& nv : v_neighbors[0]) {
	// 	std::cout << nv.first << std::endl;
	// }

	std::vector<int> mask(numV, -2);	
	int num_boundary = 0;
	for (int i = 0; i < numV; ++i) {
		int is_boundary = 0;	
		for (auto& info : v_neighbors[i]) {
			if (info.second == 1) {
				is_boundary += 1;
			}
		}
		if (is_boundary >= v_neighbors[i].size() * score_thres) {
			mask[i] = -1;
			num_boundary += 1;
		}	
	}

	if (output_mask) {	
		int b = 0;
		for (int i = 0; i < numV; ++i) {
			output_mask[i] = (mask[i] == -1) ? 1 : 0;	
			b += output_mask[i];
		}
	}

	int num_labels = 0;	
	for (int i = 0; i < mask.size(); ++i) {
		if (mask[i] == -2) {
			std::queue<int> q;
			q.push(i);
			mask[i] = num_labels;
			while (!q.empty()) {
				int v = q.front();
				q.pop();
				for (auto& nv : v_neighbors[v]) {
					if (nv.second == 0 && mask[nv.first] == -2) {
						mask[nv.first] = num_labels;
						q.push(nv.first);
					}
					if (nv.second == 1 && mask[nv.first] == -1) {
						mask[nv.first] = num_labels;
					}
				}
			}
			num_labels += 1;	
		}
	}

	for (int i = 0; i < numF; ++i) {
		int label = -1;
		for (int j = 0; j < 3; ++j) {
			if (mask[F[i * 3 + j]] >= 0) {
				label = mask[F[i * 3 + j]];
				break;
			}
		}
		face_labels[i] = label;
	}

	for (int i = 0; i < numV; ++i) {
		output_label[i] = mask[i];
	}
}

};