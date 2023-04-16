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
    std::cout << "selectwithinDistance" << std::endl;
   
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
    std::cout << "countWithinDistance" << std::endl;

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

#define PCL_INSTANTIATE_SampleConsensusModelCuboid(T) template class PCL_EXPORTS pcl::SampleConsensusModelCuboid<T>;

#endif // PCL_SAMPLE_CONSENSUS_IMPL_SAC_MODEL_PARALLEL_PLANE_H_
