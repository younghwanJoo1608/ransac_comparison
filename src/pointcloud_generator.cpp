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

int main(int argc, char **argv)
{
    double point_size = 0.01;
    double xmin = -0.3;
    double xmax = 0.3;
    double ymin = -0.4;
    double ymax = 0.4;
    double zmin = -0.2;
    double zmax = 0.2;

    // Plane1 : front plane
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane1(new pcl::PointCloud<pcl::PointXYZ>());
    for (double i = ymin; i <= ymax; i += point_size)
    {
        for (double j = zmin; j <= zmax; j += point_size)
        {
            pcl::PointXYZ point;

            point.x = xmin;
            point.y = i;
            point.z = j;

            plane1->push_back(point);
        }
    }

    pcl::ModelCoefficients::Ptr plane_eq1(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers1(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg1;
    seg1.setOptimizeCoefficients(true);
    seg1.setModelType(pcl::SACMODEL_PLANE);
    seg1.setMethodType(pcl::SAC_RANSAC);
    seg1.setDistanceThreshold(0.01);
    seg1.setInputCloud(plane1);
    seg1.segment(*inliers1, *plane_eq1);

    ROS_INFO("Plane 1 :%fx + %fy + %fz = %f", plane_eq1->values.at(0), plane_eq1->values.at(1), plane_eq1->values.at(2), plane_eq1->values.at(3));

    // Plane2 : right side plane
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane2(new pcl::PointCloud<pcl::PointXYZ>());
    for (double i = xmin; i < xmax; i += point_size)
    {
        for (double j = zmin; j < zmax; j += point_size)
        {
            pcl::PointXYZ point;

            point.x = i;
            point.y = ymin;
            point.z = j;

            plane2->push_back(point);
        }
    }

    pcl::ModelCoefficients::Ptr plane_eq2(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers2(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg2;
    seg2.setOptimizeCoefficients(true);
    seg2.setModelType(pcl::SACMODEL_PLANE);
    seg2.setMethodType(pcl::SAC_RANSAC);
    seg2.setDistanceThreshold(0.01);
    seg2.setInputCloud(plane2);
    seg2.segment(*inliers2, *plane_eq2);

    ROS_INFO("Plane 2 :%fx + %fy + %fz = %f", plane_eq2->values.at(0), plane_eq2->values.at(1), plane_eq2->values.at(2), plane_eq2->values.at(3));

    // Plane3 : upper plane
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane3(new pcl::PointCloud<pcl::PointXYZ>());
    for (double i = xmin; i < xmax; i += point_size)
    {
        for (double j = ymin; j < ymax; j += point_size)
        {
            pcl::PointXYZ point;

            point.x = i;
            point.y = j;
            point.z = zmax;

            plane3->push_back(point);
        }
    }

    pcl::ModelCoefficients::Ptr plane_eq3(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers3(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg3;
    seg3.setOptimizeCoefficients(true);
    seg3.setModelType(pcl::SACMODEL_PLANE);
    seg3.setMethodType(pcl::SAC_RANSAC);
    seg3.setDistanceThreshold(0.01);
    seg3.setInputCloud(plane3);
    seg3.segment(*inliers3, *plane_eq3);

    ROS_INFO("Plane 3 :%fx + %fy + %fz = %f", plane_eq3->values.at(0), plane_eq3->values.at(1), plane_eq3->values.at(2), plane_eq3->values.at(3));

    pcl::PointCloud<pcl::PointXYZ>::Ptr planes(new pcl::PointCloud<pcl::PointXYZ>());
    planes->points.resize(plane1->size() + plane2->size() + plane3->size());

    *planes = *plane1 + *plane2 + *plane3;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0, 0.005);

    for (int i = 0; i < planes->size(); i++)
    {
        planes->points[i].x += dist(gen);
        planes->points[i].y += dist(gen);
        planes->points[i].z += dist(gen);
    }

    // std::ofstream outfile("/home/jyh/catkin_ws/src/ransac_comparison/data/points.txt");

    // for (int i = 0; i < planes->size(); i++)
    // {
    //     outfile << planes->points[i].x << "\t" << planes->points[i].y << "\t" << planes->points[i].z << std::endl;
    // }

        
    pcl::ModelCoefficients::Ptr plane_eq(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg(true);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(planes);
    seg.segment(*inliers, *plane_eq);

    //outfile << plane_eq->values.at(0) << "\t" << plane_eq->values.at(1) << "\t" << plane_eq->values.at(2) << "\t" << plane_eq->values.at(3) << std::endl;
    ROS_INFO("Plane 3 :%fx + %fy + %fz = %f", plane_eq->values.at(0), plane_eq->values.at(1), plane_eq->values.at(2), plane_eq->values.at(3));

    sleep(1);
        
        //outfile.close();

    ROS_INFO("data saved");
    ROS_INFO("plane1 : %d points", plane1->size());
    ROS_INFO("plane2 : %d points", plane2->size());
    ROS_INFO("plane3 : %d points", plane3->size());
    ROS_INFO("total %d points", planes->size());

    pcl::visualization::PCLVisualizer viewer1("Simple Cloud Viewer");
    viewer1.addPointCloud<pcl::PointXYZ>(planes, "src_red");
    viewer1.addPlane(*plane_eq1, "plane");
    viewer1.addPlane(*plane_eq2, "plane");
    while (!viewer1.wasStopped())
    {
        viewer1.spinOnce();
    }
}