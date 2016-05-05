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

		thrust::device_vector<int> indices_device(reading.size()*4);	// cuda code
		thrust::device_vector<float> dists_device(reading.size()*4);
		flann::Matrix<int> indices_device_matrix( (int*)thrust::raw_pointer_cast(&indices_device[0]),reading.size(),4);
	flann::Matrix<float> dists_device_matrix( (float*)thrust::raw_pointer_cast(&dists_device[0]),reading.size(),4);
		
		flann::SearchParams sp;
		sp.matrices_in_gpu_ram=true;
				
		int iterations = stoi(argv[3]);	// No. of icp iterations
		
		flann::Matrix<int> indices_host( new int[ reading.size()*4],reading.size(),4 );	// cuda code
    flann::Matrix<float> dists_host( new float[ reading.size()*4],reading.size(),4 );
				
		for(int i=0; i<iterations; ++i)
		{
	    index.knnSearch(query_device_matrix, indices_device_matrix, dists_device_matrix, nn, sp );	//cuda code
	    
    	thrust::copy( dists_device.begin(), dists_device.end(), dists_host.ptr() );	// cuda code
			thrust::copy( indices_device.begin(), indices_device.end(), indices_host.ptr() );
			
			for(int j=0; j<reading.size(); ++j)
			{
//				cout<<query_host[j].x<<"  "<<query_host[j].y<<"  "<<query_host[j].z<<"  -  "<<indices_host[j][0]<<"  -  "<<data_host[indices_host[j][0]].x<<"  "<<data_host[indices_host[j][0]].y<<"  "<<data_host[indices_host[j][0]].z<<"  -  "<<dists_host[j][0]<<endl;
			}
		}
	}
	return 0;
}
