#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include "ransac_comparison/sac_segmentation.h"

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <random>
#include <fstream>
#include <unistd.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "test");
    ros::NodeHandle nh;
    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG); 

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

    // std::ofstream outfile("/home/jyh/catkin_ws/src/ransac_comparison/data/pyransac_pcl.txt");

    std::ofstream outfile1("/home/jyh/catkin_ws/src/ransac_comparison/data/ransac_new.txt");
    std::ofstream outfile2("/home/jyh/catkin_ws/src/ransac_comparison/data/ransac_new2.txt");
    std::ofstream outfile3("/home/jyh/catkin_ws/src/ransac_comparison/data/ransac_new3.txt");

    double dur = 0;
    for (int i = 0; i < 10; i++)
    {
        std::cout << "Start" << std::endl;

        double start = ros::Time::now().toNSec();
        std::vector<pcl::ModelCoefficients> plane_eq_model;

        pcl::ModelCoefficients::Ptr plane_eq1(new pcl::ModelCoefficients);
        pcl::ModelCoefficients::Ptr plane_eq2(new pcl::ModelCoefficients);
        pcl::ModelCoefficients::Ptr plane_eq3(new pcl::ModelCoefficients);

        plane_eq_model.push_back(*plane_eq1);
        plane_eq_model.push_back(*plane_eq2);
        plane_eq_model.push_back(*plane_eq3);

        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg(true);
        seg.setOptimizeCoefficients(false);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMaxIterations(500);
        std::cout << "model_type_: " << seg.getModelType() << std::endl;

        seg.setIsCuboid(true);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);
        seg.setInputCloud(planes);
        std::cout << "isCuboid: " << seg.getIsCuboid() << std::endl;

        std::cout << "before start" << std::endl;
        seg.segment(*inliers, plane_eq_model);

        // std::vector<Eigen::Vector4f> plane_eqs;

        double end = ros::Time::now().toNSec();
        dur += (end - start);
        std::cout << "Total Duration : " << (end - start) / 1000000 << " ms" << std::endl;
        std::cout << plane_eq_model.size() << std::endl;

        outfile1 << plane_eq_model[0].values.at(0) << "\t" << plane_eq_model[0].values.at(1) << "\t" << plane_eq_model[0].values.at(2) << "\t" << plane_eq_model[0].values.at(3) << std::endl;
        outfile2 << plane_eq_model[1].values.at(0) << "\t" << plane_eq_model[1].values.at(1) << "\t" << plane_eq_model[1].values.at(2) << "\t" << plane_eq_model[1].values.at(3) << std::endl;
        outfile3 << plane_eq_model[2].values.at(0) << "\t" << plane_eq_model[2].values.at(1) << "\t" << plane_eq_model[2].values.at(2) << "\t" << plane_eq_model[2].values.at(3) << std::endl;

        // for (int j = 0; j < plane_eqs.size(); j++)
        // {
        //     if (plane_eqs[j][3] < 0)
        //     {
        //         for (int k = 0; k < 4; k++)
        //         {
        //             plane_eqs[j][k] = -plane_eqs[j][k];
        //         }
        //     }
        // }

        // std::sort(std::begin(plane_eqs),
        //           std::end(plane_eqs),
        //           [](const Eigen::Vector4f &a, const Eigen::Vector4f &b)
        //           { return a[3] > b[3]; });
        // outfile1 << plane_eqs[0][0] << "\t" << plane_eqs[0][1] << "\t" << plane_eqs[0][2] << "\t" << plane_eqs[0][3] << std::endl;
        // outfile2 << plane_eqs[1][0] << "\t" << plane_eqs[1][1] << "\t" << plane_eqs[1][2] << "\t" << plane_eqs[1][3] << std::endl;
        // outfile3 << plane_eqs[2][0] << "\t" << plane_eqs[2][1] << "\t" << plane_eqs[2][2] << "\t" << plane_eqs[2][3] << std::endl;

        // if (plane_eq->values.at(3) > 0)
        //     outfile
        //         << plane_eq->values.at(0) << "\t" << plane_eq->values.at(1) << "\t" << plane_eq->values.at(2) << "\t" << plane_eq->values.at(3) << std::endl;
        // else
        //     outfile << -plane_eq->values.at(0) << "\t" << -plane_eq->values.at(1) << "\t" << -plane_eq->values.at(2) << "\t" << -plane_eq->values.at(3) << std::endl;

        // if (plane_eqs[0](3) > 0)
        //     outfile
        //         << plane_eqs[0](0) << "\t" << plane_eqs[0](1) << "\t" << plane_eqs[0](2) << "\t" << plane_eqs[0](3) << std::endl;
        // else
        //     outfile << -plane_eqs[0](0) << "\t" << -plane_eqs[0](1) << "\t" << -plane_eqs[0](2) << "\t" << -plane_eqs[0](3) << std::endl;

        // outfile << plane_eq_model[0]->values[0] << "\t" << plane_eq_model[0]->values[1] << "\t" << plane_eq_model[0]->values[2] << "\t" << plane_eq_model[0]->values[3] << std::endl;
    }

    outfile1.close();
    outfile2.close();
    outfile3.close();

    // outfile3.close();

    // pcl::visualization::PCLVisualizer viewer1("Simple Cloud Viewer");
    // viewer1.addPointCloud<pcl::PointXYZ>(planes, "src_red");
    // viewer1.addPlane(*(plane_eq_model[0]), "plane1");
    // viewer1.addPlane(*(plane_eq_model[1]), "plane2");
    // viewer1.addPlane(*(plane_eq_model[2]), "plane3");

    // while (!viewer1.wasStopped())
    // {
    //     viewer1.spinOnce();
    // }
    std::cout << dur / 10 << std::endl;
    ROS_INFO("data saved");
}