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
#include <opencv2/opencv.hpp>

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

    //std::ofstream outfile("/home/jyh/catkin_ws/src/ransac_comparison/data/ransac_lo.txt");

    for (int i = 0; i < 1000; i++)
    {
        ROS_INFO_STREAM("aa");
        pcl::ModelCoefficients::Ptr plane_eq(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg(true);
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);
        seg.setInputCloud(planes);
        seg.segment(*inliers, *plane_eq);

        std::vector<pcl::PointXYZ> normalizedPoints;
        normalizedPoints.reserve(inliers->indices.size());

        // Calculating the mass point of the points
        pcl::PointXYZ masspoint(0, 0, 0);

        for (const auto inlierIdx : inliers->indices)
        {
            pcl::PointXYZ p = planes->points[inlierIdx];
            masspoint.getArray3fMap() += p.getArray3fMap();
            normalizedPoints.emplace_back(p);
        }
        masspoint.getArray3fMap() = masspoint.getArray3fMap() * (1.0 / inliers->indices.size());

        // Move the point cloud to have the origin in their mass point
        for (auto &point : normalizedPoints)
            point.getArray3fMap() -= masspoint.getArray3fMap();

        // Calculating the average distance from the origin
        double averageDistance = 0.0;
        for (auto &point : normalizedPoints)
        {
            averageDistance += sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        }

        averageDistance /= normalizedPoints.size();
        const double ratio = sqrt(2) / averageDistance;

        // Making the average distance to be sqrt(2)
        for (auto &point : normalizedPoints)
            point.getArray3fMap() *= ratio;

        // Now, we should solve the equation.
        cv::Mat A(normalizedPoints.size(), 3, CV_64F);

        // Building the coefficient matrix
        for (size_t pointIdx = 0; pointIdx < normalizedPoints.size(); ++pointIdx)
        {
            const size_t &rowIdx = pointIdx;

            A.at<double>(rowIdx, 0) = normalizedPoints[pointIdx].x;
            A.at<double>(rowIdx, 1) = normalizedPoints[pointIdx].y;
            A.at<double>(rowIdx, 2) = normalizedPoints[pointIdx].z;
        }

        cv::Mat evals, evecs;
        cv::eigen(A.t() * A, evals, evecs);
        const cv::Mat &normal = evecs.row(2); // the normal of the line is the eigenvector corresponding to the smallest eigenvalue
        const double &a = normal.at<double>(0),
                     &b = normal.at<double>(1),
                     &c = normal.at<double>(2);
        Eigen::Vector3f normalP(a, b, c);
        Eigen::Vector3f massP(masspoint.x, masspoint.y, masspoint.z);
        const double d = -normalP.dot(massP);

        // if (d > 0)
        //     outfile << a << "\t" << b << "\t" << c << "\t" << d << std::endl;
        // else
        //     outfile << -a << "\t" << -b << "\t" << -c << "\t" << -d << std::endl;
        ROS_INFO_STREAM("aa");
        sleep(1);
    }

    //outfile.close();

    ROS_INFO("data saved");
}