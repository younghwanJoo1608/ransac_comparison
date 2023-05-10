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
inline __m256 pcl::SampleConsensusModelCuboid<PointT>::dist8(const std::size_t i, const __m256 &a_vec, const __m256 &b_vec, const __m256 &c_vec, const __m256 &d_vec, const __m256 &abs_help, const PointCloudConstPtr &input) const
{
    // The andnot-function realizes an abs-operation: the sign bit is removed
    return _mm256_andnot_ps(abs_help,
                            _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(a_vec, _mm256_set_ps(input->points[i].x, input->points[i + 1].x, input->points[i + 2].x, input->points[i + 3].x, input->points[i + 4].x, input->points[i + 5].x, input->points[i + 6].x, input->points[i + 7].x)),
                                                        _mm256_mul_ps(b_vec, _mm256_set_ps(input->points[i].y, input->points[i + 1].y, input->points[i + 2].y, input->points[i + 3].y, input->points[i + 4].y, input->points[i + 5].y, input->points[i + 6].y, input->points[i + 7].y))),
                                          _mm256_add_ps(_mm256_mul_ps(c_vec, _mm256_set_ps(input->points[i].z, input->points[i + 1].z, input->points[i + 2].z, input->points[i + 3].z, input->points[i + 4].z, input->points[i + 5].z, input->points[i + 6].z, input->points[i + 7].z)),
                                                        d_vec))); // TODO this could be replaced by three fmadd-instructions (if available), but the speed gain would probably be minimal
}

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
pcl::SampleConsensusModelCuboid<PointT>::countWithinDistanceAVX(
    const Eigen::VectorXf &model_coefficients, const double threshold, const PointCloudConstPtr &input, const IndicesPtr &indices, std::size_t i) const
{
    std::size_t nr_p = 0;
    const __m256 a_vec = _mm256_set1_ps(model_coefficients[0]);
    const __m256 b_vec = _mm256_set1_ps(model_coefficients[1]);
    const __m256 c_vec = _mm256_set1_ps(model_coefficients[2]);
    const __m256 d_vec = _mm256_set1_ps(model_coefficients[3]);
    const __m256 threshold_vec = _mm256_set1_ps(threshold);
    const __m256 abs_help = _mm256_set1_ps(-0.0F); // -0.0F (negative zero) means that all bits are 0, only the sign bit is 1
    __m256i res = _mm256_set1_epi32(0);            // This corresponds to nr_p: 8 32bit integers that, summed together, hold the number of inliers
    for (; (i + 8) <= indices->size(); i += 8)
    {
        const __m256 mask = _mm256_cmp_ps(dist8(i, a_vec, b_vec, c_vec, d_vec, abs_help, input), threshold_vec, _CMP_LT_OQ); // The mask contains 1 bits if the corresponding points are inliers, else 0 bits
        res = _mm256_add_epi32(res, _mm256_and_si256(_mm256_set1_epi32(1), _mm256_castps_si256(mask)));               // The latter part creates a vector with ones (as 32bit integers) where the points are inliers
                                                                                                                      // const int res = _mm256_movemask_ps (mask);
        // if (res &   1) nr_p++;
        // if (res &   2) nr_p++;
        // if (res &   4) nr_p++;
        // if (res &   8) nr_p++;
        // if (res &  16) nr_p++;
        // if (res &  32) nr_p++;
        // if (res &  64) nr_p++;
        // if (res & 128) nr_p++;
    }
    nr_p += _mm256_extract_epi32(res, 0);
    nr_p += _mm256_extract_epi32(res, 1);
    nr_p += _mm256_extract_epi32(res, 2);
    nr_p += _mm256_extract_epi32(res, 3);
    nr_p += _mm256_extract_epi32(res, 4);
    nr_p += _mm256_extract_epi32(res, 5);
    nr_p += _mm256_extract_epi32(res, 6);
    nr_p += _mm256_extract_epi32(res, 7);

    // Process the remaining points (at most 7)
    nr_p += countWithinDistanceStandard(model_coefficients, threshold, input, indices, i);
    return (nr_p);
}

//////////////////////////////////////////////////////////////////////////
template <typename PointT>
std::size_t
pcl::SampleConsensusModelCuboid<PointT>::countWithinDistanceStandard(
    const Eigen::VectorXf &model_coefficients, const double threshold, const PointCloudConstPtr &input, const IndicesPtr &indices, std::size_t i) const
{

    std::size_t nr_p = 0;
    // Iterate through the 3d points and calculate the distances from them to the plane
    for (; i < indices->size(); ++i)
    {
        // Calculate the distance from the point to the plane normal as the dot product
        // D = (P-A).N/|N|
        Eigen::Vector4f pt((*input)[(*indices)[i]].x,
                           (*input)[(*indices)[i]].y,
                           (*input)[(*indices)[i]].z,
                           1.0f);
        if (std::abs(model_coefficients.dot(pt)) < threshold)
        {
            nr_p++;
        }
    }

    return (nr_p);
}

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

    return (countWithinDistanceAVX(model_coefficients, threshold, input_, indices_));
}

template <typename PointT>
std::size_t
pcl::SampleConsensusModelCuboid<PointT>::countWithinDistanceSecond(
    const Eigen::VectorXf &model_coefficients, const double threshold) const
{
    // Check if the model is valid given the user constraints
    if (!isModelValid(model_coefficients))
    {
        return (0);
    }

    return (countWithinDistanceAVX(model_coefficients, threshold, temp_, new_indices_));
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
void pcl::SampleConsensusModelCuboid<PointT>::filterInliers(Indices &inliers, PointCloudPtr filtered, bool isfirst)
{
    pcl::PointIndices::Ptr inliers_ptr(new pcl::PointIndices());
    inliers_ptr->indices = inliers;
    // std::cout << "temp : " << temp_->size() << std::endl;
    pcl::ExtractIndices<PointT> extract;
    //*filtered.reset(new PointCloud());

    if (isfirst)
    {
        extract.setInputCloud(input_);
        extract.setIndices(inliers_ptr); // ptr이 들어가야 함. indicesptr 멤버변수를 만들고 fitting 후 inlier ptr을 저장하게 하자.
        extract.setNegative(true);
        extract.filter(*filtered);
    }
    else
    {
        extract.setInputCloud(temp_);
        extract.setIndices(inliers_ptr); // ptr이 들어가야 함. indicesptr 멤버변수를 만들고 fitting 후 inlier ptr을 저장하게 하자.
        extract.setNegative(true);
        extract.filter(*filtered);
    }
    return;
}
template <typename PointT>
void pcl::SampleConsensusModelCuboid<PointT>::resetIndices(Indices &new_indices, PointCloud &filtered)
{

    pcl::PointIndices::Ptr inliers_ptr(new pcl::PointIndices());
    PointCloudPtr temp(new PointCloud());
    *temp = filtered;
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(temp);
    extract.setIndices(inliers_ptr);
    extract.setNegative(false);
    extract.filter(*temp);
    new_indices = inliers_ptr->indices;

    new_indices.resize(filtered.size());
    std::iota(std::begin(new_indices), std::end(new_indices), 0);

    *new_indices_ = new_indices;

    return;
}
template <typename PointT>
bool pcl::SampleConsensusModelCuboid<PointT>::computeModelCoefficientsSecond(const std::vector<int> &samples,
                                                                             Eigen::VectorXf &model_coefficients) const
{
    // Need 3 samples
    if (samples.size() != sample_size_)
    {
        PCL_ERROR("[pcl::SampleConsensusModelCuboid::computeModelCoefficients] Invalid set of samples given (%lu)!\n", samples.size());
        return (false);
    }
    
    Eigen::Array4f p3;
    p3 << temp_->points[samples[0]].x, temp_->points[samples[0]].y,
        temp_->points[samples[0]].z, 0;

    Eigen::Array4f p4;
    p4 << temp_->points[samples[1]].x, temp_->points[samples[1]].y,
        temp_->points[samples[1]].z, 0;
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
                                                                            std::vector<Eigen::VectorXf> &model_coefficients_array, 
                                                                                Eigen::VectorXf &model_coefficients) const
{
    // Need 3 samples
    if (samples.size() != sample_size_)
    {
        PCL_ERROR("[pcl::SampleConsensusModelCuboid::computeModelCoefficients] Invalid set of samples given (%lu)!\n", samples.size());
        return (false);
    }

    Eigen::Array4f p5;
    p5 << temp_->points[samples[0]].x, temp_->points[samples[0]].y,
        temp_->points[samples[0]].z, 0;

    Eigen::Array4f n1;
    n1 << model_coefficients_array[0][0], model_coefficients_array[0][1], model_coefficients_array[0][2], model_coefficients_array[0][3];
    Eigen::Array4f n2;
    n2 << model_coefficients_array[1][0], model_coefficients_array[1][1], model_coefficients_array[1][2], model_coefficients_array[1][3];

    model_coefficients[0] = n2[1] * n1[2] - n2[2] * n1[1];
    model_coefficients[1] = n2[2] * n1[0] - n2[0] * n1[2];
    model_coefficients[2] = n2[0] * n1[1] - n2[1] * n1[0];
    model_coefficients[3] = 0;

    // Normalize
    model_coefficients.normalize();

    // ... + d = 0
    model_coefficients[3] = -1 * (model_coefficients.template head<4>().dot(p5.matrix()));

    return (true);
}

template <typename PointT>
void pcl::SampleConsensusModelCuboid<PointT>::selectWithinDistanceSecond(
    const Eigen::VectorXf &model_coefficients, const double threshold, std::vector<int> &inliers)
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

    inliers.resize(new_indices_->size());
    error_sqr_dists_.resize(new_indices_->size());

    // Iterate through the 3d points and calculate the distances from them to the plane
    for (std::size_t i = 0; i < new_indices_->size(); ++i)
    {
        // Calculate the distance from the point to the plane normal as the dot product
        // D = (P-A).N/|N|
        Eigen::Vector4f pt(temp_->points[(*new_indices_)[i]].x,
                           temp_->points[(*new_indices_)[i]].y,
                           temp_->points[(*new_indices_)[i]].z,
                           1);

        float distance = std::abs(model_coefficients.dot(pt));

        if (distance < threshold)
        {
            // Returns the indices of the points whose distances are smaller than the threshold
            inliers[nr_p] = (*new_indices_)[i];
            error_sqr_dists_[nr_p] = static_cast<double>(distance);
            ++nr_p;
        }
    }
    inliers.resize(nr_p);
    error_sqr_dists_.resize(nr_p);
}
#define PCL_INSTANTIATE_SampleConsensusModelCuboid(T) template class PCL_EXPORTS pcl::SampleConsensusModelCuboid<T>;

#endif // PCL_SAMPLE_CONSENSUS_IMPL_SAC_MODEL_PARALLEL_PLANE_H_
