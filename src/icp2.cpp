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

		flann::Matrix<float> query(new float[reading.size()*reading[0].size()], reading.size(), reading[0].size());	// flann equivalent of point clouds query->reading dataset->reference
		flann::Matrix<float> dataset(new float[reference.size()*reference[0].size()], reference.size(), reference[0].size());
		
		for(int i=0; i<query.rows; ++i)	// copy point cloud data
		{
			for(int j=0; j<query.cols; ++j)
			{
				query[i][j]=reading[i][j];
			}
		}
		for(int i=0; i<dataset.rows; ++i)
		{
			for(int j=0; j<dataset.cols; ++j)
			{
				dataset[i][j]=reference[i][j];
			}
		}
		reading.clear();
		reference.clear();

		flann::Matrix<long unsigned int> indices(new long unsigned int[query.rows*nn], query.rows, nn);
		flann::Matrix<float> dists(new float[query.rows*nn], query.rows, nn);
		
		// construct an randomized kd-tree index using kd-trees-cuda-3d
//		flann::Index<flann::L2<float> > index(dataset, flann::KDTreeIndexParams(1));	// Normal kdtree search
		flann::Index<flann::L2<float> > index(dataset, flann::KDTreeCuda3dIndexParams());
//	  flann::KDTreeCuda3dIndex<flann::L2<float>> index(dataset);	// bogus
		index.buildIndex();	// Build index
		
		int iterations = stoi(argv[3]);	// No. of icp iterations
		
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
			// do a knn search, using 128 checks
			index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));
			
			gsl_vector_set_zero(query_mean);
			for(int j=0; j<query.rows; ++j)
			{
				for(int k=0; k<query.cols; ++k)
				{
					gsl_vector_set(query_mean, k, gsl_vector_get(query_mean, k)+query[j][k]/query.rows);
				}
			}

			gsl_vector_set_zero(dataset_mean);
			for(int j=0; j<dataset.rows; ++j)
			{
				for(int k=0; k<dataset.cols; ++k)
				{
						gsl_vector_set(dataset_mean, k, gsl_vector_get(dataset_mean, k)+dataset[j][k]/dataset.rows);
				}
			}

			gsl_matrix_set_zero(U);
			for(int j=0; j<query.rows; ++j)
			{
				for(int k=0; k<3; k++)
				{
					for(int l=0; l<3; l++)
					{
						gsl_matrix_set(U, k, l, gsl_matrix_get(U, k, l)+(dataset[indices[j][0]][k]-gsl_vector_get(dataset_mean, k))*(query[j][l]-gsl_vector_get(query_mean, l)));
					}
				}
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
			
			for(int j=0; j<query.rows; ++j)
			{
				for(int k=0; k<3; ++k)
				{
					gsl_vector_set(temp3, k, query[j][k]);
				}
				gsl_vector_set(temp3, 3, 1.0);
				gsl_blas_dgemv(CblasNoTrans, 1.0, T, temp3, 0.0, temp4);
				for(int k=0; k<3; ++k)
				{
					query[j][k] = gsl_vector_get(temp4, k);
				}
			}
			
			gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, T, TF, 0.0, temp2);
			gsl_matrix_memcpy(TF, temp2);
			cout<<endl<<"iteration - "<<i+1<<endl;
			for(int j=0; j<4; ++j)
			{
				for(int k=0; k<4; ++k)
				{
					cout<<gsl_matrix_get(TF, j, k)<<"     ";
				}
				cout<<endl;
			}
		}
		gsl_matrix_free(TF);
		gsl_vector_free(temp1);
		gsl_matrix_free(temp2);
		gsl_vector_free(temp3);
		gsl_vector_free(temp4);
		gsl_matrix_free(U);
		gsl_matrix_free(V);
		gsl_vector_free(S);
		gsl_vector_free(work);
		gsl_matrix_free(R);
		gsl_vector_free(t);
		gsl_vector_free(query_mean);
		gsl_vector_free(dataset_mean);
		gsl_matrix_free(T);

		delete[] query.ptr();
		delete[] dataset.ptr();
		delete[] indices.ptr();
		delete[] dists.ptr();
	}
	return 0;
}
