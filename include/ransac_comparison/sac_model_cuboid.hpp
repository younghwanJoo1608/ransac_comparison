/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 a
 */

#ifndef SAC_MODEL_CUBOID_H_
#define SAC_MODEL_CUBOID_H_

#include "sac_model_cuboid.h"

//////////////////////////////////////////////////////////////////////////
template <typename PointT>
void pcl::SampleConsensusModelCuboid<PointT>::selectWithinDistance(
    const Eigen::VectorXf &model_coefficients, const double threshold, Indices &inliers)
{
    // Check if the model is valid given the user constraints   
    if (!isModelValid(model_coefficients))
    {
        inliers.clear();
        return;
    }

    SampleConsensusModelPlane<PointT>::selectWithinDistance(model_coefficients, threshold, inliers);
}

//////////////////////////////////////////////////////////////////////////
template <typename PointT>
std::size_t
pcl::SampleConsensusModelCuboid<PointT>::countWithinDistance(
    const Eigen::VectorXf &model_coefficients, const double threshold) const
{
    // Check if the model is valid given the user constraints
    if (!isModelValid(model_coefficients))
    {
        return (0);
    }

    return (SampleConsensusModelPlane<PointT>::countWithinDistance(model_coefficients, threshold));
}

//////////////////////////////////////////////////////////////////////////
template <typename PointT>
void pcl::SampleConsensusModelCuboid<PointT>::getDistancesToModel(
    const Eigen::VectorXf &model_coefficients, std::vector<double> &distances) const
{
    // Check if the model is valid given the user constraints
    if (!isModelValid(model_coefficients))
    {
        distances.clear();
        return;
    }

    SampleConsensusModelPlane<PointT>::getDistancesToModel(model_coefficients, distances);
}

//////////////////////////////////////////////////////////////////////////
template <typename PointT>
bool pcl::SampleConsensusModelCuboid<PointT>::isModelValid(const Eigen::VectorXf &model_coefficients) const
{
    return SampleConsensusModel<PointT>::isModelValid(model_coefficients);
}

template <typename PointT>
void pcl::SampleConsensusModelCuboid<PointT>::filterInliers(Indices &inliers, pcl::PointCloud<pcl::PointXYZ>::Ptr filtered, bool isfirst)
{
    pcl::PointIndices::Ptr inliers_ptr(new pcl::PointIndices());
    inliers_ptr->indices = inliers;

    pcl::ExtractIndices<PointT> extract;
    filtered_pcd_.reset(new PointCloud);

    if (isfirst)
    {
        extract.setInputCloud(input_);
        extract.setIndices(inliers_ptr); // ptr이 들어가야 함. indicesptr 멤버변수를 만들고 fitting 후 inlier ptr을 저장하게 하자.
        extract.setNegative(true);
        extract.filter(*filtered_pcd_);
        *filtered = *filtered_pcd_;
    }
    else
    {
        extract.setInputCloud(filtered);
        extract.setIndices(inliers_ptr); // ptr이 들어가야 함. indicesptr 멤버변수를 만들고 fitting 후 inlier ptr을 저장하게 하자.
        extract.setNegative(true);
        extract.filter(*filtered_pcd_);
        *filtered = *filtered_pcd_;
    }

    return;
}
template <typename PointT>
void pcl::SampleConsensusModelCuboid<PointT>::resetIndices(Indices &new_inliers, PointCloud &filtered)
{
    pcl::PointIndices::Ptr inliers_ptr(new pcl::PointIndices());
    PointCloudPtr temp(new PointCloud());
    *temp = filtered;
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(temp);
    extract.setIndices(inliers_ptr);
    extract.setNegative(false);
    extract.filter(*temp);
    new_inliers = inliers_ptr->indices;

    new_inliers.resize(filtered.size());
    std::iota(std::begin(new_inliers), std::end(new_inliers), 0);
    return;
}
template <typename PointT>
bool pcl::SampleConsensusModelCuboid<PointT>::computeModelCoefficientsSecond(const std::vector<int> &samples,
                                                                             Eigen::VectorXf &model_coefficients,
                                                                             PointCloud &cloud) const
{
    // Need 3 samples
    if (samples.size() != sample_size_)
    {
        PCL_ERROR("[pcl::SampleConsensusModelCuboid::computeModelCoefficients] Invalid set of samples given (%lu)!\n", samples.size());
        return (false);
    }

    Eigen::Array4f p3;
    p3 << cloud[samples[0]].x, cloud[samples[0]].y,
        cloud[samples[0]].z, 0;

    Eigen::Array4f p4;
    p4 << cloud[samples[1]].x, cloud[samples[1]].y,
        cloud[samples[1]].z, 0;
    // pcl::Array4fMapConst p3 = cloud[samples[0]].getArray4fMap();
    // pcl::Array4fMapConst p4 = cloud[samples[1]].getArray4fMap();

    Eigen::Array4f p4p3 = p4 - p3;
    Eigen::Array4f n1;
    n1 << model_coefficients[0], model_coefficients[1], model_coefficients[2], model_coefficients[3];

    model_coefficients[0] = p4p3[1] * n1[2] - p4p3[2] * n1[1];
    model_coefficients[1] = p4p3[2] * n1[0] - p4p3[0] * n1[2];
    model_coefficients[2] = p4p3[0] * n1[1] - p4p3[1] * n1[0];
    model_coefficients[3] = 0;

    // Normalize
    model_coefficients.normalize();

    // ... + d = 0
    model_coefficients[3] = -1 * (model_coefficients.template head<4>().dot(p3.matrix()));

    return (true);
}
template <typename PointT>
bool pcl::SampleConsensusModelCuboid<PointT>::computeModelCoefficientsThird(const std::vector<int> &samples,
                                                                           Eigen::VectorXf &model_coefficients,
                                                                           PointCloud &cloud) const
{
    return (true);
}
template <typename PointT>
std::size_t
pcl::SampleConsensusModelCuboid<PointT>::countWithinDistanceSecond(
    const Eigen::VectorXf &model_coefficients, const double threshold, Indices &new_indices, PointCloud &cloud) const
{
    // Needs a valid set of model coefficients
    if (model_coefficients.size() != model_size_)
    {
        PCL_ERROR("[pcl::SampleConsensusModelCuboid::countWithinDistance] Invalid number of model coefficients given (%lu)!\n", model_coefficients.size());
        return (0);
    }

    std::size_t nr_p = 0;

    // Iterate through the 3d points and calculate the distances from them to the plane
    for (std::size_t i = 0; i < new_indices.size(); ++i)
    {
        // Calculate the distance from the point to the plane normal as the dot product
        // D = (P-A).N/|N|
        Eigen::Vector4f pt(cloud[new_indices[i]].x,
                           cloud[new_indices[i]].y,
                           cloud[new_indices[i]].z,
                           1);
        if (std::abs(model_coefficients.dot(pt)) < threshold)
            nr_p++;
    }
    return (nr_p);
}
template <typename PointT>
void pcl::SampleConsensusModelCuboid<PointT>::selectWithinDistanceSecond(
    const Eigen::VectorXf &model_coefficients, const double threshold, std::vector<int> &inliers, Indices &new_indices, PointCloud &cloud)
{
    // Needs a valid set of model coefficients
    if (model_coefficients.size() != model_size_)
    {
        PCL_ERROR("[pcl::SampleConsensusModelCuboid::selectWithinDistance] Invalid number of model coefficients given (%lu)!\n", model_coefficients.size());
        return;
    }

    int nr_p = 0;
    inliers.clear();
    error_sqr_dists_.clear();

    inliers.resize(new_indices.size());
    error_sqr_dists_.resize(new_indices.size());

    // Iterate through the 3d points and calculate the distances from them to the plane
    for (std::size_t i = 0; i < new_indices.size(); ++i)
    {
        // Calculate the distance from the point to the plane normal as the dot product
        // D = (P-A).N/|N|
        Eigen::Vector4f pt(cloud[new_indices[i]].x,
                           cloud[new_indices[i]].y,
                           cloud[new_indices[i]].z,
                           1);

        float distance = std::abs(model_coefficients.dot(pt));

        if (distance < threshold)
        {
            // Returns the indices of the points whose distances are smaller than the threshold
            inliers[nr_p] = new_indices[i];
            error_sqr_dists_[nr_p] = static_cast<double>(distance);
            ++nr_p;
        }
    }
    inliers.resize(nr_p);
    error_sqr_dists_.resize(nr_p);
}
#define PCL_INSTANTIATE_SampleConsensusModelCuboid(T) template class PCL_EXPORTS pcl::SampleConsensusModelCuboid<T>;

#endif // PCL_SAMPLE_CONSENSUS_IMPL_SAC_MODEL_PARALLEL_PLANE_H_
