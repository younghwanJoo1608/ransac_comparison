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
#include <pcl/surface/concave_hull.h>

#include <ros/ros.h>
#include <random>
#include <fstream>
#include <unistd.h>
#include <cmath>

int main(int argc, char **argv)
{
    std::ifstream infile("/home/jyh/catkin_ws/src/ransac_comparison/data/rot/points_5.txt");
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

    // Compute concavehull for (inner) cloud.
    pcl::ConcaveHull<pcl::PointXYZ> chull;
    chull.setAlpha(0.1);
    std::vector<pcl::Vertices> polygons;
    chull.setInputCloud(planes);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
    chull.reconstruct(*cloud_hull, polygons);
    if (cloud_hull->empty())
    {
        std::cout << "Failed to compute concavehull for instance" << std::endl;
    }

    // Each surface of concavehull are considered as candidate for front plane.
    Eigen::Vector3f u0;
    int best_polygon_idx = -1;
    pcl::PointIndices::Ptr surface(new pcl::PointIndices);

    for (int it = 0; it < polygons.size(); it++)
    {
        const pcl::Vertices &polygon = polygons.at(it);
        size_t n_vertices = polygon.vertices.size();

        const uint32_t &idx0 = polygon.vertices.at(0);
        const uint32_t &idx1 = polygon.vertices.at(1);
        const uint32_t &idx2 = polygon.vertices.at(2);
        const pcl::PointXYZ &pcl_pt0 = cloud_hull->at(idx0);
        const pcl::PointXYZ &pcl_pt1 = cloud_hull->at(idx1);
        const pcl::PointXYZ &pcl_pt2 = cloud_hull->at(idx2);
        Eigen::Vector3f p_candi(pcl_pt0.x, pcl_pt0.y, pcl_pt0.z);
        Eigen::Vector3f u0_candi(pcl_pt1.x - pcl_pt0.x, pcl_pt1.y - pcl_pt0.y, pcl_pt1.z - pcl_pt0.z);
        Eigen::Vector3f v0_candi(pcl_pt2.x - pcl_pt0.x, pcl_pt2.y - pcl_pt0.y, pcl_pt2.z - pcl_pt0.z);
        u0_candi.normalize();

        // If the offset between the normal vector and the depth direction is greater than the threshold,
        // exclude it from the candidate list.
        Eigen::Vector3f depth_dir;
        depth_dir(0) = 0.904;
        depth_dir(1) = 0.1223;
        depth_dir(2) = 0.409667;
        Eigen::Vector3f n0_candi = u0_candi.cross(v0_candi).normalized();
        if (n0_candi.dot(depth_dir) > 0.)
            n0_candi = -n0_candi;
        double cos_dir = -n0_candi.dot(depth_dir);
        ROS_INFO("cos_dir : %f", cos_dir);
        ROS_INFO("min_cos_dir : %f", 40 / 180 * M_PI);
        if (cos_dir < 40/180*M_PI)
            continue;

        // Count number of the inlier points which are close to plane.
        pcl::PointIndices::Ptr surface_candi(new pcl::PointIndices);
        surface_candi->indices.reserve(planes->size());
        int n_points_nearthan_frontplane = 0;
        int i = 0;
        for (const pcl::PointXYZ &pt : *planes)
        {
            double e = n0_candi.dot(Eigen::Vector3f(pt.x, pt.y, pt.z) - p_candi);
            if (std::abs(e) < 0.02)
                surface_candi->indices.push_back(i);
            if (e > 0.02)
                n_points_nearthan_frontplane++;
            i++;
        }

        // For under-segmented mask includes front box and backward box,
        // reported by Eugene's STC_dataset_2022-01-11-18-10-05.bag
        const int nth = 0.1 * (float)planes->size();
        if (n_points_nearthan_frontplane > nth)
            continue;

        // The candidate plane with maximum inliner points are chosen as an champion.
        if (surface_candi->indices.size() > surface->indices.size())
        {
            u0 = u0_candi;
            surface = surface_candi;
            best_polygon_idx = it;
        }
    }

    if(best_polygon_idx < 0){
        std::cout << "Failed to get front plane for instance" << std::endl;
    }

    pcl::ModelCoefficients::Ptr plane_eq(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg(true);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(planes);
    seg.segment(*inliers, *plane_eq);
    
    pcl::visualization::PCLVisualizer viewer1("Simple Cloud Viewer");
    viewer1.addPointCloud<pcl::PointXYZ>(planes, "src_red");
    viewer1.addPlane(*plane_eq, "plane");

    while (!viewer1.wasStopped())
    {
        viewer1.spinOnce();
    }
}