/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2009, Willow Garage, Inc.
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

#ifndef RANSAC_H_
#define RANSAC_H_

#include "ransac.h"
// #ifdef _OPENMP
#include <omp.h>
// #endif

// #if defined _OPENMP && _OPENMP >= 201107 // We need OpenMP 3.1 for the atomic constructs
// #define OPENMP_AVAILABLE_RANSAC true
// #else
// #define OPENMP_AVAILABLE_RANSAC false
// #endif
#define OPENMP_AVAILABLE_RANSAC true
#include <ros/ros.h>
#include <fstream>

//////////////////////////////////////////////////////////////////////////
template <typename PointT>
bool pcl::RandomSampleConsensus<PointT>::computeModel(int)
{
    std::ofstream outfile1("/home/jyh/catkin_ws/src/ransac_comparison/data/ransac_new_time.txt", std::ios::app);
    std::ofstream outfile2("/home/jyh/catkin_ws/src/ransac_comparison/data/ransac_new_iter.txt", std::ios::app);

    // Warn and exit if no threshold was set
    if (threshold_ == std::numeric_limits<double>::max())
    {
        PCL_ERROR("[pcl::RandomSampleConsensus::computeModel] No threshold set!\n");
        return (false);
    }

    iterations_ = 0;
    std::size_t n_best_inliers_count = 0;
    double k = std::numeric_limits<double>::max();

    std::vector<int> selection;
    Eigen::VectorXf model_coefficients(4);

    const double log_probability = std::log(1.0 - probability_);
    const double one_over_indices = 1.0 / static_cast<double>(sac_model_->getIndices()->size());

    std::size_t n_inliers_count;
    unsigned skipped_count = 0;
    // suppress infinite loops by just allowing 10 x maximum allowed iterations for invalid model parameters!
    const unsigned max_skip = max_iterations_ * 10;

    int threads = threads_;

    cuboid_size_.clear();

    PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] Computing not parallel.\n");
    double start1 = ros::Time::now().toNSec();

    // Iterate
    while (true) // infinite loop with four possible breaks
    {
        // Get X samples which satisfy the model criteria

        {
            sac_model_->getSamples(iterations_, selection); // The random number generator used when choosing the samples should not be called in parallel
        }
        if (selection.empty())
        {
            PCL_ERROR("[pcl::RandomSampleConsensus::computeModel] No samples could be selected!\n");
            break;
        }
        // Search for inliers in the point cloud for the current plane model M
        if (!sac_model_->computeModelCoefficients(selection, model_coefficients)) // This function has to be thread-safe
        {
            //++iterations_;
            unsigned skipped_count_tmp;

            skipped_count_tmp = ++skipped_count;
            if (skipped_count_tmp < max_skip)
                continue;
            else
                break;
        }

        // Select the inliers that are within threshold_ from the model
        // sac_model_->selectWithinDistance (model_coefficients, threshold_, inliers);
        // if (inliers.empty () && k > 1.0)
        //  continue;

        n_inliers_count = sac_model_->countWithinDistance(model_coefficients, threshold_); // This functions has to be thread-safe. Most work is done here

        std::size_t n_best_inliers_count_tmp;

        n_best_inliers_count_tmp = n_best_inliers_count;

        if (n_inliers_count > n_best_inliers_count_tmp) // This condition is false most of the time, and the critical region is not entered, hopefully leading to more efficient concurrency
        {
            {
                // Better match ?
                if (n_inliers_count > n_best_inliers_count)
                {
                    n_best_inliers_count = n_inliers_count; // This write and the previous read of n_best_inliers_count must be consecutive and must not be interrupted!
                    n_best_inliers_count_tmp = n_best_inliers_count;

                    // Save the current model/inlier/coefficients selection as being the best so far
                    model_ = selection;
                    model_coefficients_ = model_coefficients;

                    // Compute the k parameter (k=std::log(z)/std::log(1-w^n))
                    const double w = static_cast<double>(n_best_inliers_count) * one_over_indices;
                    double p_no_outliers = 1.0 - std::pow(w, static_cast<double>(selection.size()));
                    p_no_outliers = (std::max)(std::numeric_limits<double>::epsilon(), p_no_outliers);       // Avoid division by -Inf
                    p_no_outliers = (std::min)(1.0 - std::numeric_limits<double>::epsilon(), p_no_outliers); // Avoid division by 0.
                    k = log_probability / std::log(p_no_outliers);
                }
            } // omp critical
        }

        int iterations_tmp;
        double k_tmp;

        iterations_tmp = ++iterations_;

        k_tmp = k;

        PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] Trial %d out of %f: %u inliers (best is: %u so far).\n", iterations_tmp, k_tmp, n_inliers_count, n_best_inliers_count_tmp);

        if (iterations_tmp > k_tmp)
        {
            PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] RANSAC reached the k param.\n");
            outfile2 << iterations_tmp << "\t";
            break;
        }
        if (iterations_tmp > max_iterations_)
        {
            PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] RANSAC reached the maximum number of trials.\n");
            outfile2 << iterations_tmp << "\t";
            break;
        }
    } // while
    double end1 = ros::Time::now().toNSec();

    PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] Model: %lu size, %u inliers.\n", model_.size(), n_best_inliers_count);

    if (model_.empty())
    {
        PCL_ERROR("[pcl::RandomSampleConsensus::computeModel] RANSAC found no model.\n");
        inliers_.clear();
        return (false);
    }

    // Get the set of inliers that correspond to the best model found so far
    sac_model_->selectWithinDistance(model_coefficients_, threshold_, inliers_);
    // pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);

    Eigen::VectorXf optimized_coefficients;
    sac_model_->optimizeModelCoefficients(inliers_, model_coefficients_, optimized_coefficients);
    model_coefficients_ = optimized_coefficients;
    sac_model_->filterInliers(inliers_, *temp_, true);
    sac_model_->getMaxDistance(model_coefficients_, threshold_, cuboid_size_);
    std::cout << inliers_.size() << std::endl;
    model_coefficients_vector_.push_back(model_coefficients_);

    // First plane found. now second plane.

    if (temp_->size() < 10)
    {
        PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] Not enough points to find second plane : %u/%u\n", temp_->size(), 10);
        for (int i = 1; i < 3; i++)
        {
            model_coefficients_ = Eigen::Vector4f::Zero();
            model_coefficients_vector_.push_back(model_coefficients_);
            cuboid_size_.push_back(0);
        }
        return true;
    }

    std::vector<int> new_indices;
    sac_model_->resetIndices(new_indices, *temp_);
    sac_model_->setTemp(temp_);

    selection.clear();
    iterations_ = 0;
    std::size_t n_best_inliers_count2 = 0;
    k = std::numeric_limits<double>::max();
    const double one_over_indices2 = 1.0 / static_cast<double>(new_indices.size());

    std::size_t n_inliers_count2;
    unsigned skipped_count2 = 0;

    double start2 = ros::Time::now().toNSec();

    while (true)
    {

        {
            sac_model_->getSamplesSecond(iterations_, selection, new_indices); // The random number generator used when choosing the samples should not be called in parallel
        }
        if (selection.empty())
        {
            PCL_ERROR("[pcl::RandomSampleConsensus::computeModel] No samples could be selected!\n");
            break;
        }
        // Search for inliers in the point cloud for the current plane model M
        if (!sac_model_->computeModelCoefficientsSecond(selection, model_coefficients)) // This function has to be thread-safe
        {
            //++iterations_;
            unsigned skipped_count_tmp;

            skipped_count_tmp = ++skipped_count2;
            if (skipped_count_tmp < max_skip)
                continue;
            else
                break;
        }

        n_inliers_count2 = sac_model_->countWithinDistanceSecond(model_coefficients, threshold_); // This functions has to be thread-safe. Most work is done here

        std::size_t n_best_inliers_count_tmp2;

        n_best_inliers_count_tmp2 = n_best_inliers_count2;

        if (n_inliers_count2 > n_best_inliers_count_tmp2) // This condition is false most of the time, and the critical region is not entered, hopefully leading to more efficient concurrency
        {

            {
                // Better match ?
                if (n_inliers_count2 > n_best_inliers_count2)
                {
                    n_best_inliers_count2 = n_inliers_count2; // This write and the previous read of n_best_inliers_count must be consecutive and must not be interrupted!
                    n_best_inliers_count_tmp2 = n_best_inliers_count2;

                    // Save the current model/inlier/coefficients selection as being the best so far
                    model_ = selection;
                    model_coefficients_ = model_coefficients;

                    // Compute the k parameter (k=std::log(z)/std::log(1-w^n))
                    const double w2 = static_cast<double>(n_best_inliers_count2) * one_over_indices2;
                    double p_no_outliers = 1.0 - std::pow(w2, static_cast<double>(selection.size()));
                    // double p_no_outliers = 1.0 - std::pow(w2, static_cast<double>(2));
                    p_no_outliers = (std::max)(std::numeric_limits<double>::epsilon(), p_no_outliers);       // Avoid division by -Inf
                    p_no_outliers = (std::min)(1.0 - std::numeric_limits<double>::epsilon(), p_no_outliers); // Avoid division by 0.
                    k = log_probability / std::log(p_no_outliers);
                }
            } // omp critical
        }

        int iterations_tmp2;
        double k_tmp2;

        iterations_tmp2 = ++iterations_;

        k_tmp2 = k;

        PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] Trial %d out of %f: %u inliers (best is: %u so far).\n", iterations_tmp2, k_tmp2, n_inliers_count2, n_best_inliers_count_tmp2);

        if (iterations_tmp2 > k_tmp2)
        {
            PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] RANSAC reached the k param.\n");
            outfile2 << iterations_tmp2 << "\t";
            break;
        }
        if (iterations_tmp2 > max_iterations_)
        {
            PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] RANSAC reached the maximum number of trials.\n");
            outfile2 << iterations_tmp2 << "\t";
            break;
        }
    } // while
    double end2 = ros::Time::now().toNSec();

    std::vector<int> inliers_second;
    // Get the set of inliers that correspond to the best model found so far
    sac_model_->selectWithinDistanceSecond(model_coefficients_, threshold_, inliers_second);

    // pcl::PointCloud<pcl::PointXYZ>::Ptr temp2(new pcl::PointCloud<pcl::PointXYZ>);

    sac_model_->filterInliers(inliers_second, *temp_, false);
    sac_model_->getMaxDistance(model_coefficients_, threshold_, cuboid_size_);
    std::cout << inliers_second.size() << std::endl;
    model_coefficients_vector_.push_back(model_coefficients_);
    inliers_.insert(inliers_.end(), inliers_second.begin(), inliers_second.end());

    // Second plane found. now Third plane.

    if (temp_->size() < 10)
    {
        PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] Not enough points to find third plane : %u/%u\n", temp_->size(), 10);
        for (int i = 2; i < 3; i++)
        {
            model_coefficients_ = Eigen::Vector4f::Zero();
            model_coefficients_vector_.push_back(model_coefficients_);
            cuboid_size_.push_back(0);
        }
        return true;
    }

    std::vector<int> new_indices2;
    sac_model_->resetIndices(new_indices2, *temp_);

    sac_model_->setTemp(temp_);

    iterations_ = 0;
    std::size_t n_best_inliers_count3 = 0;
    k = std::numeric_limits<double>::max();
    const double one_over_indices3 = 1.0 / static_cast<double>(new_indices2.size());

    std::size_t n_inliers_count3;
    unsigned skipped_count3 = 0;
    double start3 = ros::Time::now().toNSec();

    while (true)
    {

        {
            sac_model_->getSamplesSecond(iterations_, selection, new_indices2); // The random number generator used when choosing the samples should not be called in parallel
        }
        if (selection.empty())
        {
            PCL_ERROR("[pcl::RandomSampleConsensus::computeModel] No samples could be selected!\n");
            break;
        }
        // Search for inliers in the point cloud for the current plane model M
        if (!sac_model_->computeModelCoefficientsThird(selection, model_coefficients_vector_, model_coefficients)) // This function has to be thread-safe
        {
            //++iterations_;
            unsigned skipped_count_tmp;

            skipped_count_tmp = ++skipped_count3;
            if (skipped_count_tmp < max_skip)
                continue;
            else
                break;
        }

        n_inliers_count3 = sac_model_->countWithinDistanceSecond(model_coefficients, threshold_); // This functions has to be thread-safe. Most work is done here

        std::size_t n_best_inliers_count_tmp3;

        n_best_inliers_count_tmp3 = n_best_inliers_count3;

        if (n_inliers_count3 > n_best_inliers_count_tmp3) // This condition is false most of the time, and the critical region is not entered, hopefully leading to more efficient concurrency
        {

            {
                // Better match ?
                if (n_inliers_count3 > n_best_inliers_count3)
                {
                    n_best_inliers_count3 = n_inliers_count3; // This write and the previous read of n_best_inliers_count must be consecutive and must not be interrupted!
                    n_best_inliers_count_tmp3 = n_best_inliers_count3;

                    // Save the current model/inlier/coefficients selection as being the best so far
                    model_ = selection;
                    model_coefficients_ = model_coefficients;

                    // Compute the k parameter (k=std::log(z)/std::log(1-w^n))
                    const double w3 = static_cast<double>(n_best_inliers_count3) * one_over_indices3;
                    double p_no_outliers = 1.0 - std::pow(w3, static_cast<double>(selection.size()));
                    // double p_no_outliers = 1.0 - std::pow(w3, static_cast<double>(1));
                    p_no_outliers = (std::max)(std::numeric_limits<double>::epsilon(), p_no_outliers);       // Avoid division by -Inf
                    p_no_outliers = (std::min)(1.0 - std::numeric_limits<double>::epsilon(), p_no_outliers); // Avoid division by 0.
                    k = log_probability / std::log(p_no_outliers);
                }
            } // omp critical
        }

        int iterations_tmp3;
        double k_tmp3;

        iterations_tmp3 = ++iterations_;

        k_tmp3 = k;

        PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] Trial %d out of %f: %u inliers (best is: %u so far).\n", iterations_tmp3, k_tmp3, n_inliers_count3, n_best_inliers_count_tmp3);

        if (iterations_tmp3 > k_tmp3)
        {
            PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] RANSAC reached the k param.\n");
            outfile2 << iterations_tmp3 << std::endl;
            break;
        }
        if (iterations_tmp3 > max_iterations_)
        {
            PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] RANSAC reached the maximum number of trials.\n");
            outfile2 << iterations_tmp3 << std::endl;
            break;
        }
    } // while
    double end3 = ros::Time::now().toNSec();

    std::vector<int> inliers_third;
    // Get the set of inliers that correspond to the best model found so far
    sac_model_->selectWithinDistanceSecond(model_coefficients_, threshold_, inliers_third);
    sac_model_->getMaxDistance(model_coefficients_, threshold_, cuboid_size_);

    if (cuboid_size_[2] == 0)
    {
        PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] Third plane divides cuboid.\n");
        for (int i = 2; i < 3; i++)
        {
            model_coefficients_ = Eigen::Vector4f::Zero();
            model_coefficients_vector_.push_back(model_coefficients_);
        }
        return true;
    }

    model_coefficients_vector_.push_back(model_coefficients_);
    inliers_.insert(inliers_.end(), inliers_third.begin(), inliers_third.end());
    std::cout << inliers_third.size() << std::endl;
    outfile1 << (end1 - start1) / 1000000 << "\t" << (end2 - start2) / 1000000 << "\t" << (end3 - start3) / 1000000 << std::endl;
    outfile1.close();
    outfile2.close();

    return (true);
}

#define PCL_INSTANTIATE_RandomSampleConsensus(T) template class PCL_EXPORTS pcl::RandomSampleConsensus<T>;

#endif // PCL_SAMPLE_CONSENSUS_IMPL_RANSAC_H_
