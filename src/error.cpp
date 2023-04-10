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

    std::ifstream plane_eqs("/media/jyh/Extreme SSD/230321/3/ransac6_1.txt");
    std::string pline;
    std::vector<std::vector<double>> coefficients;
    ROS_INFO_STREAM("a");
    int pline_num = 0;
    while (std::getline(plane_eqs, pline))
    {
        double coeff;
        std::stringstream ssp(pline);

        coefficients.push_back(std::vector<double>());

        while (ssp >> coeff)
        {
            coefficients[pline_num].push_back(coeff);
        }
        ++pline_num;
    }

    double total_error = 0;
    for (int i = 0; i < 1000; i++)
    {
        double error = 0;
        int num_inlier = 0;
        for (const pcl::PointXYZ &pt : *planes)
        {
            double dist = std::fabs(
                        (coefficients[i][0] * pt.x + coefficients[i][1] * pt.y + coefficients[i][2] * pt.z + coefficients[i][3]) /
                    sqrt(coefficients[i][0] * coefficients[i][0] + coefficients[i][1] * coefficients[i][1] + coefficients[i][2] * coefficients[i][2] + coefficients[i][3] * coefficients[i][3])
                    );
            
            if (dist < 0.01)
            {
                num_inlier++;
                error += dist;
            }
        }
        total_error += (error / num_inlier);
    }
    ROS_INFO_STREAM(total_error / 1000);

    ROS_INFO("data saved");
}