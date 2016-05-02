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

using namespace std;

struct {
	bool operator()(vector<double> a, vector<double> b)
	{
		if (a[0] == b[0])
			if (a[1] == b[1])
				if (a[2] == b[2])
					return true;
				else
					return (a[2] < b[2]);
			else
				return (a[1] < b[1]);
		else
			return(a[0] < b[0]);
	}
}comparator_x;
struct {
	bool operator()(vector<double> a, vector<double> b)
	{
		if (a[1] == b[1])
			if (a[2] == b[2])
				if (a[0] == b[0])
					return true;
				else
					return (a[0] < b[0]);
			else
				return (a[2] < b[2]);
		else
			return(a[1] < b[1]);
	}
}comparator_y;
struct {
	bool operator()(vector<double> a, vector<double> b)
	{
		if (a[2] == b[2])
			if (a[0] == b[0])
				if (a[1] == b[1])
					return true;
				else
					return (a[1] < b[1]);
			else
				return (a[0] < b[0]);
		else
			return(a[2] < b[2]);
	}
}comparator_z;

struct node {
	vector<double> value;
	int kd;
	int level;
	node *left;
	node *right;
	node *parent;
	node()
	{
		value = vector<double>(4);
	}
};

class kd_tree {
private:
	node *root;
	double ds(vector<double>& a, vector<double>& b)
	{
		return ((a[0] - b[0])*(a[0] - b[0]) + (a[1] - b[1])*(a[1] - b[1]) + (a[2] - b[2])*(a[2] - b[2]));
	}
	double thresholder(node* current, vector<double>& point1, vector<double>& point2)
	{
		copy((*current).value.begin(), (*current).value.end(), point2.begin());
		if (point1[(*current).kd] < (*current).value[(*current).kd])
		{
			if ((*current).left != NULL)
				return thresholder((*current).left, point1, point2);
			else
				return ds((*current).value, point1);
		}
		else
		{
			if ((*current).right != NULL)
				return thresholder((*current).right, point1, point2);
			else
				return ds((*current).value, point1);
		}
	}
	double explorer(node* current, vector<double>& point1, vector<double>& point2, double minimum)
	{
		if (ds((*current).value, point1) < minimum)
		{
			minimum = ds((*current).value, point1);
			copy((*current).value.begin(), (*current).value.end(), point2.begin());
		}
		if ((*current).left != NULL)
		{
			if (point1[(*current).kd] < (*current).value[(*current).kd])
				minimum = explorer((*current).left, point1, point2, minimum);
			else if ((point1[(*current).kd] - (*current).value[(*current).kd])*(point1[(*current).kd] - (*current).value[(*current).kd]) <= minimum)
				minimum = explorer((*current).left, point1, point2, minimum);
		}
		if ((*current).right != NULL)
		{
			if (point1[(*current).kd] >= (*current).value[(*current).kd])
				minimum = explorer((*current).right, point1, point2, minimum);
			else if ((point1[(*current).kd] - (*current).value[(*current).kd])*(point1[(*current).kd] - (*current).value[(*current).kd]) <= minimum)
				minimum = explorer((*current).left, point1, point2, minimum);
		}
		return minimum;
	}
	void build_tree(node** current, vector<vector<vector<double>>>& sorted_point, int kd, int level, node* parent)
	{
		*current = new node;
		(*current)->left = NULL;
		(*current)->right = NULL;
		(*current)->kd = kd;
		(*current)->level = level;
		(*current)->parent = parent;
		vector<double> median(4);

		copy(sorted_point[kd][(int)sorted_point[kd].size() / 2].begin(), sorted_point[kd][(int)sorted_point[kd].size() / 2].end(), median.begin());
		copy(median.begin(), median.end(), (*current)->value.begin());

		vector<vector<vector<double>>> sorted_point1(3, vector<vector<double>>((int)sorted_point[kd].size() / 2, vector<double>(4)));
		vector<vector<vector<double>>> sorted_point2(3, vector<vector<double>>((int)sorted_point[kd].size() - (int)sorted_point[kd].size() / 2 - 1, vector<double>(4)));
		int n1, n2;
		for (int j = 0; j < 3; ++j)
		{
			n1 = 0, n2 = 0;
			for (int i = 0; i < (int)(sorted_point[kd].size()); ++i)
			{
				if (sorted_point[(kd + j) % 3][i][kd] < median[kd])
				{
					copy(sorted_point[(kd + j) % 3][i].begin(), sorted_point[(kd + j) % 3][i].end(), sorted_point1[(kd + j) % 3][n1].begin());
					n1++;
				}
				else if (sorted_point[(kd + j) % 3][i][kd]>median[kd])
				{
					copy(sorted_point[(kd + j) % 3][i].begin(), sorted_point[(kd + j) % 3][i].end(), sorted_point2[(kd + j) % 3][n2].begin());
					n2++;
				}
				else
				{
					if (sorted_point[(kd + j) % 3][i][(kd + 1) % 3] < median[(kd + 1) % 3])
					{
						copy(sorted_point[(kd + j) % 3][i].begin(), sorted_point[(kd + j) % 3][i].end(), sorted_point1[(kd + j) % 3][n1].begin());
						n1++;
					}
					else if (sorted_point[(kd + j) % 3][i][(kd + 1) % 3]>median[(kd + 1) % 3])
					{
						copy(sorted_point[(kd + j) % 3][i].begin(), sorted_point[(kd + j) % 3][i].end(), sorted_point2[(kd + j) % 3][n2].begin());
						n2++;
					}
					else
					{
						if (sorted_point[(kd + j) % 3][i][(kd + 2) % 3] < median[(kd + 2) % 3])
						{
							copy(sorted_point[(kd + j) % 3][i].begin(), sorted_point[(kd + j) % 3][i].end(), sorted_point1[(kd + j) % 3][n1].begin());
							n1++;
						}
						else if (sorted_point[(kd + j) % 3][i][(kd + 2) % 3]>median[(kd + 2) % 3])
						{
							copy(sorted_point[(kd + j) % 3][i].begin(), sorted_point[(kd + j) % 3][i].end(), sorted_point2[(kd + j) % 3][n2].begin());
							n2++;
						}
						else
						{
							continue;
						}
					}
				}
			}
		}
		if (n1 != 0)
			build_tree(&(*current)->left, sorted_point1, (kd + 1) % 3, level + 1, *current);
		if (n2 != 0)
			build_tree(&(*current)->right, sorted_point2, (kd + 1) % 3, level + 1, *current);
	}
public:
	kd_tree(vector<vector<double>>& point)
	{
		vector<vector<vector<double>>> sorted_point(3, vector<vector<double>>(point.size(), vector<double>(4)));
		for (int i = 0; i < 3; ++i)
			copy(point.begin(), point.end(), sorted_point[i].begin());

		sort(sorted_point[0].begin(), sorted_point[0].end(), comparator_x);
		sort(sorted_point[1].begin(), sorted_point[1].end(), comparator_y);
		sort(sorted_point[2].begin(), sorted_point[2].end(), comparator_z);

		build_tree(&root, sorted_point, 0, 0, NULL);
	}
	double nearest_neighbour(vector<double>& point, vector<double>& neighbour)
	{
		return sqrt(explorer(root, point, neighbour, thresholder(root, point, neighbour)));
	}
};

// class to store reading and reference pointcloud and icp functions
class icp{
  private:
    vector<vector<double>> reading, reference;
    kd_tree* reference_tree;

    // function to read the pointcloud data and store as vector
    bool file_read(string filename,vector<vector<double>>& vec)
    {
      ifstream file(filename);
      if(file.is_open())
      {
        string line;
        getline(file,line);
        int n_line=stoi(line);
        vec=vector<vector<double>>(n_line, vector<double>(4));
        for(int i=0;i<n_line;i++)
        {
          getline(file,line);
          string::size_type sz_x,sz_y,sz_z;
          vec[i][0]=std::stod(line,&sz_x);
          vec[i][1]=std::stod(line.substr(sz_x),&sz_y);
          vec[i][2]=std::stod(line.substr(sz_x).substr(sz_y),&sz_z);
          vec[i][3]=std::stod(line.substr(sz_x).substr(sz_y).substr(sz_z));
        }
        file.close();
        return true;
      }
      else
      {
        return false;
      }
    }
    // function to calculate distance between two points
    double distance(vector<double>& pt1, vector<double>& pt2)
    {
      return sqrt((pt1[0]-pt2[0])*(pt1[0]-pt2[0])+(pt1[1]-pt2[1])*(pt1[1]-pt2[1])+(pt1[2]-pt2[2])*(pt1[2]-pt2[2]));
    }

  public:
    // icp object constructor
    icp(string filename1, string filename2)
    {
      if(!file_read(filename1, reading))  // reads the file for reading cloud
      {
        exit(1);
      }
      if(!file_read(filename2, reference)) // reads the file for reference cloud
      {
        exit(1);
      }
      reference_tree = new kd_tree(reference);
      vector<double> point2(4);
      double minimum;
      for(int i=0; i<(int)reading.size(); ++i)
      {
        minimum=reference_tree->nearest_neighbour(reading[i], point2);
        /*
        cout<<i<<" - ";
        cout<<"("<<reading[i][0]<<", "<<reading[i][1]<<", "<<reading[i][2]<<") ";
        cout<<"("<<point2[0]<<", "<<point2[1]<<", "<<point2[2]<<") ";
        cout<<minimum<<"\n";
        */
      }
    }
};

int main( int argc, char** argv)
{
	if(argc != 3)
	{
	  cout<<"usage: ./icp file1 file2\n";
	}
	else
	{
  	icp my_icp(argv[1],argv[2]);
	}
	return 0;
}
