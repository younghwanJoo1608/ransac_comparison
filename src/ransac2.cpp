#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <ros/ros.h>
#include <random>
#include <fstream>
#include <unistd.h>

#include "ransac_comparison/pyransac.h"

int main(int argc, char **argv)
{
    float ransac_thresh = 0.01;
    int ransac_max_iteration = 1000;
    int min_points_of_cluster = 5;

    std::ifstream infile("/home/jyh/catkin_ws/src/ransac_comparison/data/points.txt");
    std::string line;
    std::vector<std::vector<double>> v;
    int line_num = 0;

    while (std::getline(infile, line))
    {
        double value;
        std::stringstream ss(line);

        v.push_back(std::vector<double>());

        while (ss >> value)
        {
            v[line_num].push_back(value);
        }
        ++line_num;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr planes(new pcl::PointCloud<pcl::PointXYZ>());
    planes->points.resize(v.size());
    ROS_INFO_STREAM(v.size());
    for (int i = 0; i < v.size(); i++)
    {
        pcl::PointXYZ point;

        planes->points[i].x = v[i][0];
        planes->points[i].y = v[i][1];
        planes->points[i].z = v[i][2];
    }

    std::ofstream outfile1("/home/jyh/catkin_ws/src/ransac_comparison/data/ransac2_1.txt");
    std::ofstream outfile2("/home/jyh/catkin_ws/src/ransac_comparison/data/ransac2_2.txt");
    std::ofstream outfile3("/home/jyh/catkin_ws/src/ransac_comparison/data/ransac2_3.txt");

    for (int i = 0; i < 1000; i++)
    {
        int n_front_inliers = 0;
        float thresh = ransac_thresh;
        int max_iteration = ransac_max_iteration;
        std::vector<Eigen::Vector4f> plane_eqs;
        CuboidRANSAC(planes, thresh, max_iteration, plane_eqs, n_front_inliers);

        if (n_front_inliers < min_points_of_cluster)
        {
            std::cout << "Failed to compute plane for yolact instance" << std::endl;
            //break;
        }

        for (int j = 0; j < 3; j++)
        {
            if (plane_eqs[j][3] < 0)
            {
                for (int k = 0; k < 4; k++)
                {
                    plane_eqs[j][k] = -plane_eqs[j][k];
                }
            }
        }
        std::sort(std::begin(plane_eqs),
                  std::end(plane_eqs),
                  [](const Eigen::Vector4f &a, const Eigen::Vector4f &b)
                  { return a[3] > b[3]; });

        outfile1 << plane_eqs[0][0] << "\t" << plane_eqs[0][1] << "\t" << plane_eqs[0][2] << "\t" << plane_eqs[0][3] << std::endl;
        outfile2 << plane_eqs[1][0] << "\t" << plane_eqs[1][1] << "\t" << plane_eqs[1][2] << "\t" << plane_eqs[1][3] << std::endl;
        outfile3 << plane_eqs[2][0] << "\t" << plane_eqs[2][1] << "\t" << plane_eqs[2][2] << "\t" << plane_eqs[2][3] << std::endl;

        // pcl::ModelCoefficients::Ptr plane_eq(new pcl::ModelCoefficients);
        // plane_eq->values = std::vector<float>(4);
       
        // for (int i = 0; i < 4; i++)
        // {
        //     plane_eq->values.at(i) = plane_eqs[2][i];
        // }
    }

    outfile1.close();
    outfile2.close();
    outfile3.close();

    // pcl::visualization::PCLVisualizer viewer1("Simple Cloud Viewer");
    // viewer1.addPointCloud<pcl::PointXYZ>(planes, "src_red");
    // viewer1.addPlane(*plane_eq, "plane");

    // while (!viewer1.wasStopped())
    // {
    //     viewer1.spinOnce();
    // }

    ROS_INFO("data saved");
}