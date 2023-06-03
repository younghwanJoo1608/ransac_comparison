/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
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
 *
 */

#pragma once

// #include <pcl/sample_consensus/sac_model_plane.h>
#include "sac_model_plane.h"
#include <pcl/filters/extract_indices.h>
#include <pmmintrin.h>
#include <immintrin.h>

namespace pcl
{
    /** \brief @b SampleConsensusModelCuboid defines a model for 3D plane segmentation using additional
     * angular constraints. The plane must be parallel to a user-specified axis
     * (\ref setAxis) within a user-specified angle threshold (\ref setEpsAngle).
     * In other words, the plane <b>normal</b> must be (nearly) <b>perpendicular</b> to the specified axis.
     *
     * Code example for a plane model, parallel (within a 15 degrees tolerance) with the Z axis:
     * \code
     * SampleConsensusModelCuboid<pcl::PointXYZ> model (cloud);
     * model.setAxis (Eigen::Vector3f (0.0, 0.0, 1.0));
     * model.setEpsAngle (pcl::deg2rad (15));
     * \endcode
     *
     * \note Please remember that you need to specify an angle > 0 in order to activate the axis-angle constraint!
     *
     * \author Radu B. Rusu, Nico Blodow
     * \ingroup sample_consensus
     */
    template <typename PointT>
    class SampleConsensusModelCuboid : public SampleConsensusModelPlane<PointT>
    {
    public:
        using SampleConsensusModel<PointT>::model_name_;
        using SampleConsensusModel<PointT>::input_;
        using SampleConsensusModel<PointT>::temp_;
        using SampleConsensusModel<PointT>::indices_;
        using SampleConsensusModel<PointT>::new_indices_;
        using SampleConsensusModel<PointT>::error_sqr_dists_;

        using PointCloud = typename SampleConsensusModelCuboid<PointT>::PointCloud;
        using PointCloudPtr = typename SampleConsensusModelCuboid<PointT>::PointCloudPtr;
        using PointCloudConstPtr = typename SampleConsensusModelCuboid<PointT>::PointCloudConstPtr;

        using Ptr = shared_ptr<SampleConsensusModelCuboid<PointT>>;
        using ConstPtr = shared_ptr<const SampleConsensusModelCuboid<PointT>>;

        /** \brief Constructor for base SampleConsensusModelCuboid.
         * \param[in] cloud the input point cloud dataset
         * \param[in] random if true set the random seed to the current time, else set to 12345 (default: false)
         */
        SampleConsensusModelCuboid(const PointCloudConstPtr &cloud,
                                   bool random = false)
            : SampleConsensusModelPlane<PointT>(cloud, random), axis_(Eigen::Vector3f::Zero()), eps_angle_(0.0), sin_angle_(-1.0)
        {
            model_name_ = "SampleConsensusModelCuboid";
            sample_size_ = 3;
            model_size_ = 4;
        }

        /** \brief Constructor for base SampleConsensusModelCuboid.
         * \param[in] cloud the input point cloud dataset
         * \param[in] indices a vector of point indices to be used from \a cloud
         * \param[in] random if true set the random seed to the current time, else set to 12345 (default: false)
         */
        SampleConsensusModelCuboid(const PointCloudConstPtr &cloud,
                                   const Indices &indices,
                                   bool random = false)
            : SampleConsensusModelPlane<PointT>(cloud, indices, random), axis_(Eigen::Vector3f::Zero()), eps_angle_(0.0), sin_angle_(-1.0)
        {
            model_name_ = "SampleConsensusModelCuboid";
            sample_size_ = 3;
            model_size_ = 4;
        }

        /** \brief Empty destructor */
        ~SampleConsensusModelCuboid() {}

        /** \brief Set the axis along which we need to search for a plane perpendicular to.
         * \param[in] ax the axis along which we need to search for a plane perpendicular to
         */
        inline void
        setAxis(const Eigen::Vector3f &ax) { axis_ = ax; }

        /** \brief Get the axis along which we need to search for a plane perpendicular to. */
        inline Eigen::Vector3f
        getAxis() const { return (axis_); }

        /** \brief Set the angle epsilon (delta) threshold.
         * \param[in] ea the maximum allowed difference between the plane normal and the given axis.
         * \note You need to specify an angle > 0 in order to activate the axis-angle constraint!
         */
        inline void
        setEpsAngle(const double ea)
        {
            eps_angle_ = ea;
            sin_angle_ = std::abs(sin(ea));
        }

        /** \brief Get the angle epsilon (delta) threshold. */
        inline double
        getEpsAngle() const { return (eps_angle_); }

        /** \brief Select all the points which respect the given model coefficients as inliers.
         * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
         * \param[out] inliers the resultant model inliers
         */
        void
        selectWithinDistance(const Eigen::VectorXf &model_coefficients,
                             const double threshold,
                             Indices &inliers) override;

        /** \brief Count all the points which respect the given model coefficients as inliers.
         *
         * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
         * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
         * \return the resultant number of inliers
         */
        std::size_t
        countWithinDistance(const Eigen::VectorXf &model_coefficients,
                            const double threshold) const override;

        /** \brief Compute all distances from the cloud data to a given plane model.
         * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
         * \param[out] distances the resultant estimated distances
         */
        void
        getDistancesToModel(const Eigen::VectorXf &model_coefficients,
                            std::vector<double> &distances) const override;

        /** \brief Return a unique id for this model (SACMODEL_PARALLEL_PLANE). */
        inline pcl::SacModel
        getModelType() const override { return (SACMODEL_PLANE); }

        void setTemp(PointCloudConstPtr temp)
        {
            temp_.reset(new PointCloud(*temp));
        }

    protected:
        using SampleConsensusModel<PointT>::sample_size_;
        using SampleConsensusModel<PointT>::model_size_;

        /** \brief Check whether a model is valid given the user constraints.
         * \param[in] model_coefficients the set of model coefficients
         */
        bool
        isModelValid(const Eigen::VectorXf &model_coefficients) const override;

        inline __m256 dist8(const std::size_t i, const __m256 &a_vec, const __m256 &b_vec, const __m256 &c_vec, const __m256 &d_vec, const __m256 &abs_help, const PointCloudConstPtr &input) const;

        void filterInliers(Indices &inliers, PointCloud &filtered, bool isfirst) override;
        void
        resetIndices(Indices &new_indices, PointCloud &filtered) override;
        bool
        computeModelCoefficientsSecond(const std::vector<int> &samples,
                                       Eigen::VectorXf &model_coefficients) const override;
        bool
        computeModelCoefficientsThird(const std::vector<int> &samples,
                                      std::vector<Eigen::VectorXf> &model_coefficients_array,
                                      Eigen::VectorXf &model_coefficients) const override;
        std::size_t
        countWithinDistanceAVX(const Eigen::VectorXf &model_coefficients,
                               const double threshold, const PointCloudConstPtr &input, const IndicesPtr &indices,
                               std::size_t i = 0) const;
        std::size_t
        countWithinDistanceStandard(const Eigen::VectorXf &model_coefficients,
                                    const double threshold, const PointCloudConstPtr &input, const IndicesPtr &indices,
                                    std::size_t i = 0) const;
        std::size_t
        countWithinDistanceSecond(const Eigen::VectorXf &model_coefficients,
                                  const double threshold) const override;
        void
        optimizeModelCoefficients(const std::vector<int> &inliers,
                                  const Eigen::VectorXf &model_coefficients,
                                  Eigen::VectorXf &optimized_coefficients) const override;

        void
        selectWithinDistanceSecond(const Eigen::VectorXf &model_coefficients,
                                   const double threshold,
                                   std::vector<int> &inliers) override;

        void
        getMaxDistance(const Eigen::VectorXf &model_coefficients,
                       const double threshold,
                       std::vector<float> &cuboid_size) override;
        /** \brief The axis along which we need to search for a plane perpendicular to. */
        Eigen::Vector3f axis_;

        /** \brief The maximum allowed difference between the plane and the given axis. */
        double eps_angle_;

        /** \brief The sine of the angle*/
        double sin_angle_;

        // PointCloudPtr filtered_pcd_;
    };
}

#include "sac_model_cuboid.hpp"