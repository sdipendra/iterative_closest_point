#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>

#define FLANN_USE_CUDA
#include <flann/flann.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

using namespace std;

bool file_read(string filename,vector<vector<float>>& vec)
{
	ifstream file(filename);
	if(file.is_open())
	{
		string line;
		getline(file,line);
		int n_line=stoi(line);
		vec=vector<vector<float>>(n_line, vector<float>(3));
		for(int i=0;i<n_line;i++)
		{
			getline(file,line);
			string::size_type sz_x,sz_y,sz_z;
			vec[i][0]=std::stod(line,&sz_x);
			vec[i][1]=std::stod(line.substr(sz_x),&sz_y);
			vec[i][2]=std::stod(line.substr(sz_x).substr(sz_y),&sz_z);
			std::stod(line.substr(sz_x).substr(sz_y).substr(sz_z));
		}
		file.close();
		return true;
	}
	else
	{
		return false;
	}
}

int main( int argc, char** argv)
{
	if(argc != 4)
	{
		cout<<"usage: ./icp file1 file2 no_of_iterations\n";
	}
	else
	{
		int nn = 1;	// No. of nearest neighbours to search
		vector<vector<float>> reading, reference;	// reading and reference point clouds
		if(!file_read(argv[1], reading))
		{
			exit(1);
		}
		if(!file_read(argv[2], reference))
		{
			exit(1);
		}
		
		thrust::host_vector<float4> query_host(reading.size());
		thrust::host_vector<float4> data_host(reference.size());	// cuda code
		
		for(int i=0; i<reading.size(); ++i)	// copy point cloud data
		{
			query_host[i]=make_float4(reading[i][0], reading[i][1], reading[i][2], 0);	// cuda code
		}
		for(int i=0; i<reference.size(); ++i)
		{
			data_host[i]=make_float4(reference[i][0], reference[i][1], reference[i][2], 0);	// cuda code
		}
		
		thrust::device_vector<float4> query_device = query_host;
		thrust::device_vector<float4> data_device = data_host;	// cuda code
		
		flann::Matrix<float> query_device_matrix( (float*)thrust::raw_pointer_cast(&query_device[0]),reading.size(),3,4*4);	// cuda code
		flann::Matrix<float> data_device_matrix( (float*)thrust::raw_pointer_cast(&data_device[0]),reference.size(),3,4*4);
				
		flann::KDTreeCuda3dIndexParams index_params;	// cuda code
		index_params["input_is_gpu_float4"]=true;
		flann::KDTreeCuda3dIndex<flann::L2_Simple<float> > index(data_device_matrix, index_params);
    index.buildIndex();

		thrust::device_vector<int> indices_device(reading.size()*nn);	// cuda code
		thrust::device_vector<float> dists_device(reading.size()*nn);
		flann::Matrix<int> indices_device_matrix( (int*)thrust::raw_pointer_cast(&indices_device[0]),reading.size(),nn);
	flann::Matrix<float> dists_device_matrix( (float*)thrust::raw_pointer_cast(&dists_device[0]),reading.size(),nn);
		
		flann::SearchParams sp;
		sp.matrices_in_gpu_ram=true;
				
		int iterations = stoi(argv[3]);	// No. of icp iterations
		
		flann::Matrix<int> indices_host( new int[ reading.size()*nn],reading.size(),nn );	// cuda code
    flann::Matrix<float> dists_host( new float[ reading.size()*nn],reading.size(),nn );
		
		gsl_matrix *TF = gsl_matrix_alloc(4, 4);
		gsl_vector *query_mean = gsl_vector_alloc(3);
		gsl_vector *dataset_mean = gsl_vector_alloc(3);
		gsl_matrix *U = gsl_matrix_alloc(3, 3);
		gsl_matrix *V = gsl_matrix_alloc(3, 3);
		gsl_vector *S = gsl_vector_alloc(3);
		gsl_vector *work = gsl_vector_alloc(3);
		gsl_matrix *R = gsl_matrix_alloc(3, 3);
		gsl_vector *t = gsl_vector_alloc(3);
		gsl_matrix *T = gsl_matrix_alloc(4, 4);
		gsl_vector *temp1 = gsl_vector_alloc(3);
		gsl_matrix *temp2 = gsl_matrix_alloc(4, 4);
		gsl_vector *temp3 = gsl_vector_alloc(4);
		gsl_vector *temp4 = gsl_vector_alloc(4);
		
		gsl_matrix_set_identity(TF);
		
		for(int i=0; i<iterations; ++i)
		{
	    index.knnSearch(query_device_matrix, indices_device_matrix, dists_device_matrix, nn, sp );	//cuda code
	    
    	thrust::copy( dists_device.begin(), dists_device.end(), dists_host.ptr() );	// cuda code
			thrust::copy( indices_device.begin(), indices_device.end(), indices_host.ptr() );
			
			gsl_vector_set_zero(query_mean);
			for(int j=0; j<reading.size(); ++j)
			{
				gsl_vector_set(query_mean, 0, gsl_vector_get(query_mean, 0)+query_host[j].x/reading.size());
				gsl_vector_set(query_mean, 1, gsl_vector_get(query_mean, 1)+query_host[j].y/reading.size());
				gsl_vector_set(query_mean, 2, gsl_vector_get(query_mean, 2)+query_host[j].z/reading.size());
			}
			
			gsl_vector_set_zero(dataset_mean);
			for(int j=0; j<reference.size(); ++j)
			{
				gsl_vector_set(dataset_mean, 0, gsl_vector_get(dataset_mean, 0)+data_host[j].x/reading.size());
				gsl_vector_set(dataset_mean, 1, gsl_vector_get(dataset_mean, 1)+data_host[j].y/reading.size());
				gsl_vector_set(dataset_mean, 2, gsl_vector_get(dataset_mean, 2)+data_host[j].z/reading.size());
			}
			
			gsl_matrix_set_zero(U);
			for(int j=0; j<reading.size(); ++j)
			{
				gsl_matrix_set(U, 0, 0, gsl_matrix_get(U, 0, 0)+(data_host[indices_host[j][0]].x-gsl_vector_get(dataset_mean, 0))*(query_host[j].x-gsl_vector_get(query_mean, 0)));
				gsl_matrix_set(U, 0, 1, gsl_matrix_get(U, 0, 1)+(data_host[indices_host[j][0]].x-gsl_vector_get(dataset_mean, 0))*(query_host[j].y-gsl_vector_get(query_mean, 1)));
				gsl_matrix_set(U, 0, 2, gsl_matrix_get(U, 0, 2)+(data_host[indices_host[j][0]].x-gsl_vector_get(dataset_mean, 0))*(query_host[j].z-gsl_vector_get(query_mean, 2)));
				gsl_matrix_set(U, 1, 0, gsl_matrix_get(U, 1, 0)+(data_host[indices_host[j][0]].y-gsl_vector_get(dataset_mean, 1))*(query_host[j].x-gsl_vector_get(query_mean, 0)));
				gsl_matrix_set(U, 1, 1, gsl_matrix_get(U, 1, 1)+(data_host[indices_host[j][0]].y-gsl_vector_get(dataset_mean, 1))*(query_host[j].y-gsl_vector_get(query_mean, 1)));
				gsl_matrix_set(U, 1, 2, gsl_matrix_get(U, 1, 2)+(data_host[indices_host[j][0]].y-gsl_vector_get(dataset_mean, 1))*(query_host[j].z-gsl_vector_get(query_mean, 2)));
				gsl_matrix_set(U, 2, 0, gsl_matrix_get(U, 2, 0)+(data_host[indices_host[j][0]].z-gsl_vector_get(dataset_mean, 2))*(query_host[j].x-gsl_vector_get(query_mean, 0)));
				gsl_matrix_set(U, 2, 1, gsl_matrix_get(U, 2, 1)+(data_host[indices_host[j][0]].z-gsl_vector_get(dataset_mean, 2))*(query_host[j].y-gsl_vector_get(query_mean, 1)));
				gsl_matrix_set(U, 2, 2, gsl_matrix_get(U, 2, 2)+(data_host[indices_host[j][0]].z-gsl_vector_get(dataset_mean, 2))*(query_host[j].z-gsl_vector_get(query_mean, 2)));
			}
				
			
			gsl_linalg_SV_decomp(U, V, S, work);
			gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, U, V, 0.0, R);

			gsl_blas_dgemv(CblasNoTrans, 1.0, R, query_mean, 0.0, temp1);
			gsl_vector_memcpy(t, dataset_mean);
			gsl_vector_sub(t, temp1);
			
			gsl_matrix_set_identity(T);
			for(int j=0; j<3; ++j)
			{
				for(int k=0; k<3; k++)
				{
					gsl_matrix_set(T, j, k, gsl_matrix_get(R, j, k));
				}
				gsl_matrix_set(T, j, 3, gsl_vector_get(t, j));
			}
			
			for(int j=0; j<reading.size(); ++j)
			{
				gsl_vector_set(temp3, 0, query_host[j].x);
				gsl_vector_set(temp3, 1, query_host[j].y);
				gsl_vector_set(temp3, 2, query_host[j].z);
				gsl_vector_set(temp3, 3, 1.0);

				gsl_blas_dgemv(CblasNoTrans, 1.0, T, temp3, 0.0, temp4);

				query_host[j].x = gsl_vector_get(temp4, 0);
				query_host[j].y = gsl_vector_get(temp4, 1);
				query_host[j].z = gsl_vector_get(temp4, 2);
			}
			
			gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, T, TF, 0.0, temp2);
			gsl_matrix_memcpy(TF, temp2);
			/*
			cout<<endl<<"iteration - "<<i+1<<endl;
			for(int j=0; j<4; ++j)
			{
				for(int k=0; k<4; ++k)
				{
					cout<<gsl_matrix_get(TF, j, k)<<"     ";
				}
				cout<<endl;
			}
			*/
			
/*			
			for(int j=0; j<reading.size(); ++j)
			{
				cout<<query_host[j].x<<"  "<<query_host[j].y<<"  "<<query_host[j].z<<"  "<<query_host[j].w<<"  -  "<<indices_host[j][0]<<"  -  "<<data_host[indices_host[j][0]].x<<"  "<<data_host[indices_host[j][0]].y<<"  "<<data_host[indices_host[j][0]].z<<"  "<<data_host[indices_host[j][0]].w<<"  -  "<<dists_host[j][0]<<endl;
			}
			*/
		}
	}
	return 0;
}
