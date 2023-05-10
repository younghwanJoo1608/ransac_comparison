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
#include <pcl/console/parse.h>

#include <ros/ros.h>
#include <random>
#include <fstream>
#include <unistd.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "test");
    ros::NodeHandle nh;
    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);

    std::ifstream infile("/home/jyh/catkin_ws/src/ransac_comparison/data/points_filtered.txt");
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

    std::ofstream outfile("/home/jyh/catkin_ws/src/ransac_comparison/data/ransac1_filtered.txt");

    for (int i = 0; i < 10; i++)
    {
        double start = ros::Time::now().toNSec();
        pcl::ModelCoefficients::Ptr plane_eq(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg(true);
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);
        seg.setInputCloud(planes);
        seg.segment(*inliers, *plane_eq);
        std::cout << seg.getModelType() << ", " << seg.getMethodType() << std::endl;

        if (plane_eq->values.at(3) > 0)
            outfile << plane_eq->values.at(0) << "\t" << plane_eq->values.at(1) << "\t" << plane_eq->values.at(2) << "\t" << plane_eq->values.at(3) << std::endl;
        else
            outfile << -plane_eq->values.at(0) << "\t" << -plane_eq->values.at(1) << "\t" << -plane_eq->values.at(2) << "\t" << -plane_eq->values.at(3) << std::endl;
        double end = ros::Time::now().toNSec();
        std::cout << "Total Duration : " << (end - start) / 1000000 << " ms" << std::endl;

        std::cout << plane_eq->values.at(0) << "\t" << plane_eq->values.at(1) << "\t" << plane_eq->values.at(2) << "\t" << plane_eq->values.at(3) << std::endl;

        sleep(1);
    }

    outfile.close();

    ROS_INFO("data saved");
}