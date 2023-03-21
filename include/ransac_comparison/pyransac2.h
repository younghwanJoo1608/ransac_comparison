#include <pcl/filters/random_sample.h>
#include <Eigen/Dense>
#include <cmath>
#include <pcl/search/kdtree.h>

void CuboidRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                float thresh,
                int max_iteration,
                std::vector<Eigen::Vector4f>& best_eqs,
                int& best_num_inliers
                )
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud);
    for (int i = 0; i < max_iteration; i++)
    {
        std::vector<Eigen::Vector4f> plane_eqs;
        // Samples 6 random points
        pcl::PointCloud<pcl::PointXYZ>::Ptr pt_samples(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::RandomSample <pcl::PointXYZ> random;

        random.setInputCloud(cloud);
        random.setSeed (std::rand());
        random.setSample((unsigned int)(4));
        random.filter(*pt_samples);

        // We have to find the plane equation described by those 3 points
        // We find first 2 vectors that are part of this plane
        // A = pt2 - pt1
        // B = pt3 - pt1
        Eigen::Vector3f pt1((*pt_samples).at(0).x,
                            (*pt_samples).at(0).y,
                            (*pt_samples).at(0).z);

        std::vector<int> kd_idxes;
        std::vector<float> sqr_dists;
        tree->nearestKSearch((*pt_samples).at(0), 30, kd_idxes, sqr_dists);

        pcl::PointCloud<pcl::PointXYZ>::Ptr kd_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr kd_samples(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::RandomSample <pcl::PointXYZ> kd_random;
        
        for (const auto& idx: kd_idxes){
            kd_cloud->points.push_back(cloud->points[idx]);
        }

        kd_random.setInputCloud(kd_cloud);
        kd_random.setSeed (std::rand());
        kd_random.setSample((unsigned int)(2));
        kd_random.filter(*kd_samples);

        
        Eigen::Vector3f pt2((*kd_samples).at(0).x,
                            (*kd_samples).at(0).y,
                            (*kd_samples).at(0).z);
        Eigen::Vector3f pt3((*kd_samples).at(1).x,
                            (*kd_samples).at(1).y,
                            (*kd_samples).at(1).z);

        Eigen::Vector3f vecA = pt2 - pt1;
        Eigen::Vector3f vecB = pt3 - pt1;

        // Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
        Eigen::Vector3f vecC = vecA.cross(vecB);

        // The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
        // We have to use a point to find k
        vecC.normalize();

        float k = -vecC.dot(pt2);
        Eigen::Vector4f plane1_eq(vecC(0), vecC(1), vecC(2), k);
        plane_eqs.push_back(plane1_eq);

        // Now we use another point to find a orthogonal plane 2
        // Calculate distance from the point to the first plane
        Eigen::Vector3f pt4((*pt_samples).at(1).x,
                            (*pt_samples).at(1).y,
                            (*pt_samples).at(1).z);

        float dist_p4_plane = (
            plane_eqs[0](0) * pt4(0)
            + plane_eqs[0](1) * pt4(1)
            + plane_eqs[0](2) * pt4(2)
            + k ) / (plane_eqs[0].norm());

        // vecC is already normal (module 1) so we only have to discount from the point, the distance*unity = distance*normal
        // A simple way of understanding this is we move our point along the normal until it reaches the plane
        Eigen::Vector3f p4_proj_plane = pt4 - dist_p4_plane * vecC;

        // Now, with help of our point p5 we can find another plane P2 which contains p4, p4_proj, p5 and
        Eigen::Vector3f vecD = p4_proj_plane - pt4;

        Eigen::Vector3f pt5((*pt_samples).at(2).x,
                            (*pt_samples).at(2).y,
                            (*pt_samples).at(2).z);

        Eigen::Vector3f vecE = pt5 - pt4;
        Eigen::Vector3f vecF = vecD.cross(vecE);
        vecF.normalize();

        k = -vecF.dot(pt5);
        Eigen::Vector4f plane2_eq(vecF(0), vecF(1), vecF(2), k);
        plane_eqs.push_back(plane2_eq);

        // The last plane will be orthogonal to the first and sacond plane (and its normals will be orthogonal to first and second planes' normal)
        Eigen::Vector3f vecG = vecC.cross(vecF);

        Eigen::Vector3f pt6((*pt_samples).at(3).x,
                            (*pt_samples).at(3).y,
                            (*pt_samples).at(3).z);

        k = -vecG.dot(pt6);
        Eigen::Vector4f plane3_eq(vecG(0), vecG(1), vecG(2), k);
        plane_eqs.push_back(plane3_eq);

        // We have to find the value D for the last plane.

        // Distance from a point to a plane
        // https://mathworld.wolfram.com/Point-PlaneDistance.html
        int num_inliers = 0;

        for(const pcl::PointXYZ& pt : *cloud)
        {
            std::vector<float> dist;
            for(int j = 0; j < (plane_eqs.size()); j++)
            {
                dist.push_back(
                    std::fabs(
                        (
                            plane_eqs[j](0) * pt.x
                            + plane_eqs[j](1) * pt.y
                            + plane_eqs[j](2) * pt.z
                            + plane_eqs[j](3)
                        )
                        / (plane_eqs[j].norm())
                    ));
            }
            //float min_dist = std::min(std::min(dist[0], dist[1]), dist[2]);
            float min_dist = *std::min_element(dist.begin(), dist.end());
            if (std::fabs(min_dist) <= thresh){ num_inliers++; }
        }

        if (num_inliers > best_num_inliers)
        {
            best_eqs = plane_eqs;
            best_num_inliers = num_inliers;
        }

        if (num_inliers > (cloud->size())*0.9) {break;}
    }
    return;
}