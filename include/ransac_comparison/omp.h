#include <pcl/filters/random_sample.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>

#include <Eigen/Dense>
#include <cmath>
#include <omp.h>
#include <ros/ros.h>

#define OPENMP_AVAILABLE_RANSAC_ true

void SelectWithinDistance(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_,
                          pcl::ModelCoefficients &model_coefficients,
                          float threshold_,
                          pcl::Indices &inliers)
{
    inliers.clear();
    inliers.reserve(cloud_->size());

    for (std::size_t i = 0; i < cloud_->size(); ++i)
    {
        // Calculate the distance from the point to the plane normal as the dot product
        // D = (P-A).N/|N|

        float distance = std::abs(model_coefficients.values[0] * (*cloud_)[i].x + model_coefficients.values[1] * (*cloud_)[i].y + model_coefficients.values[2] * (*cloud_)[i].z + model_coefficients.values[3] * 1.0f);

        if (distance < threshold_)
        {
            // Returns the indices of the points whose distances are smaller than the threshold
            inliers.push_back(i);
        }
    }
}

std::size_t CountWithinDistance(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_,
                                pcl::ModelCoefficients &model_coefficients,
                                float threshold_)
{
    std::size_t nr_p = 0;

    // Iterate through the 3d points and calculate the distances from them to the line
    for (std::size_t i = 0; i < cloud_->size(); ++i)
    {
        // Calculate the distance from the point to the line
        // D = ||(P2-P1) x (P1-P0)|| / ||P2-P1|| = norm (cross (p2-p1, p2-p0)) / norm(p2-p1)

        float distance = std::abs(model_coefficients.values[0] * (*cloud_)[i].x + model_coefficients.values[1] * (*cloud_)[i].y + model_coefficients.values[2] * (*cloud_)[i].z + model_coefficients.values[3] * 1.0f);

        if (distance < threshold_)
            nr_p++;
    }
    return (nr_p);
}

bool ComputeModelCoefficients(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_,
                              pcl::Indices &samples,
                              pcl::ModelCoefficients &model_coefficients)
{
    model_coefficients.values.resize(4);
    double Dx1, Dy1, Dz1, Dx2, Dy2, Dz2, Dy1Dy2;

    Dx1 = cloud_->points.at(samples.at(1)).x - cloud_->points.at(samples.at(0)).x;
    Dy1 = cloud_->points.at(samples.at(1)).y - cloud_->points.at(samples.at(0)).y;
    Dz1 = cloud_->points.at(samples.at(1)).z - cloud_->points.at(samples.at(0)).z;

    Dx2 = cloud_->points.at(samples.at(2)).x - cloud_->points.at(samples.at(0)).x;
    Dy2 = cloud_->points.at(samples.at(2)).y - cloud_->points.at(samples.at(0)).y;
    Dz2 = cloud_->points.at(samples.at(2)).z - cloud_->points.at(samples.at(0)).z;

    Dy1Dy2 = Dy1 / Dy2;
    if (((Dx1 / Dx2) == Dy1Dy2) && (Dy1Dy2 == (Dz1 / Dz2))) // Check for collinearity
    {
        return (false);
    }
    model_coefficients.values[0] = (cloud_->points.at(samples.at(1)).y - cloud_->points.at(samples.at(0)).y) *
                                       (cloud_->points.at(samples.at(2)).z - cloud_->points.at(samples.at(0)).z) -
                                   (cloud_->points.at(samples.at(1)).z - cloud_->points.at(samples.at(0)).z) *
                                       (cloud_->points.at(samples.at(2)).y - cloud_->points.at(samples.at(0)).y);

    model_coefficients.values[1] = (cloud_->points.at(samples.at(1)).z - cloud_->points.at(samples.at(0)).z) *
                                       (cloud_->points.at(samples.at(2)).x - cloud_->points.at(samples.at(0)).x) -
                                   (cloud_->points.at(samples.at(1)).x - cloud_->points.at(samples.at(0)).x) *
                                       (cloud_->points.at(samples.at(2)).z - cloud_->points.at(samples.at(0)).z);

    model_coefficients.values[2] = (cloud_->points.at(samples.at(1)).x - cloud_->points.at(samples.at(0)).x) *
                                       (cloud_->points.at(samples.at(2)).y - cloud_->points.at(samples.at(0)).y) -
                                   (cloud_->points.at(samples.at(1)).y - cloud_->points.at(samples.at(0)).y) *
                                       (cloud_->points.at(samples.at(2)).x - cloud_->points.at(samples.at(0)).x);
    // calculate the 2-norm: norm (x) = sqrt (sum (abs (v)^2))
    // nx ny nz (aka: ax + by + cz ...
    double n_norm = sqrt(model_coefficients.values[0] * model_coefficients.values[0] +
                         model_coefficients.values[1] * model_coefficients.values[1] +
                         model_coefficients.values[2] * model_coefficients.values[2]);
    model_coefficients.values[0] /= n_norm;
    model_coefficients.values[1] /= n_norm;
    model_coefficients.values[2] /= n_norm;

    // ... + d = 0
    model_coefficients.values[3] = -1 * (model_coefficients.values[0] * cloud_->points.at(samples.at(0)).x +
                                         model_coefficients.values[1] * cloud_->points.at(samples.at(0)).y +
                                         model_coefficients.values[2] * cloud_->points.at(samples.at(0)).z);

    return (true);
}

bool isSampleGood(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::Indices &samples)
{
    if (
        std::abs((*cloud)[samples[0]].x - (*cloud)[samples[1]].x) <= std::numeric_limits<float>::epsilon() &&
        std::abs((*cloud)[samples[0]].y - (*cloud)[samples[1]].y) <= std::numeric_limits<float>::epsilon() &&
        std::abs((*cloud)[samples[0]].z - (*cloud)[samples[1]].z) <= std::numeric_limits<float>::epsilon())
        return (false);

    return (true);
}

void GetSamples(int iterations, pcl::Indices &samples, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    // pcl::PCLBase<pcl::PointXYZ> base;
    // base.setInputCloud(cloud);
    // ROS_INFO("GetSamplest");
    pcl::IndicesPtr object_indices(new pcl::Indices);
    // ROS_INFO("GetSamplesss");
    // object_indices = base.getIndices();

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.filter(*object_indices);

    int sample_size = 6;
    int index_size = object_indices->size();
    for (unsigned int iter = 0; iter < sample_size * 10; ++iter)
    {
        // Choose the random indices
        for (std::size_t i = 0; i < sample_size; ++i)
        {
            std::swap((*object_indices)[i], (*object_indices)[i + (rand() % (index_size - i))]);
        }
        std::copy((*object_indices).begin(), (*object_indices).begin() + sample_size, samples.begin());

        // If it's a good sample, stop here
        if (isSampleGood(cloud, samples))
        {
            return;
        }
    }
}

bool ComputeModel(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                  pcl::Indices &inliers,
                  pcl::ModelCoefficients &model_coefficients,
                  int max_iterations,
                  float threshold_,
                  int &n_best_inliers_count)
{

    // openmp
    int iterations = 0;
    double k = std::numeric_limits<double>::max();

    pcl::Indices selection(3); // random sampled indices (6)
    pcl::ModelCoefficients::Ptr model_coefficients_temp(new pcl::ModelCoefficients);

    double probability = 0.99;
    const double log_probability = std::log(1.0 - probability);
    const double one_over_indices = 1.0 / static_cast<double>(cloud->size());

    unsigned skipped_count = 0;

    // suppress infinite loops by just allowing 10 x maximum allowed iterations for invalid model parameters!
    const unsigned max_skip = max_iterations * 10;

    int threads = 0;
    if (threads >= 0)
    {
#if OPENMP_AVAILABLE_RANSAC_
        if (threads == 0)
        {
            threads = omp_get_num_procs();
            PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] Automatic number of threads requested, choosing %i threads.\n", threads);
        }
#else
        // Parallelization desired, but not available
        PCL_WARN("[pcl::RandomSampleConsensus::computeModel] Parallelization is requested, but OpenMP 3.1 is not available! Continuing without parallelization.\n");
        threads = -1;
#endif
    }

#if OPENMP_AVAILABLE_RANSAC_
#pragma omp parallel if (threads > 0) num_threads(threads) shared(k, skipped_count, n_best_inliers_count) firstprivate(selection, model_coefficients_temp) // would be nice to have a default(none)-clause here, but then some compilers complain about the shared const variables
#endif
    {
#if OPENMP_AVAILABLE_RANSAC_
        if (omp_in_parallel())
#pragma omp master
            PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] Computing in parallel with up to %i threads.\n", omp_get_num_threads());
        else
#endif
            PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] Computing not parallel.\n");

        // Iterate
        while (true) // infinite loop with four possible breaks
        {
            // Get X samples which satisfy the model criteria
#if OPENMP_AVAILABLE_RANSAC_
#pragma omp critical(samples)
#endif

            {
                GetSamples(iterations, selection, cloud); // The random number generator used when choosing the samples should not be called in parallel
            }

            if (selection.empty())
            {
                PCL_ERROR("[pcl::RandomSampleConsensus::computeModel] No samples could be selected!\n");
                break;
            }

            // Search for inliers in the point cloud for the current plane model M
            if (!ComputeModelCoefficients(cloud, selection, *model_coefficients_temp)) // This function has to be thread-safe
            {
                //++iterations_;
                unsigned skipped_count_tmp;
#if OPENMP_AVAILABLE_RANSAC_
#pragma omp atomic capture
#endif
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
            std::size_t n_inliers_count = CountWithinDistance(cloud, *model_coefficients_temp, threshold_); // This functions has to be thread-safe. Most work is done here
            std::size_t n_best_inliers_count_tmp;

#if OPENMP_AVAILABLE_RANSAC_
#pragma omp atomic read
#endif
            n_best_inliers_count_tmp = n_best_inliers_count;

            if (n_inliers_count > n_best_inliers_count_tmp) // This condition is false most of the time, and the critical region is not entered, hopefully leading to more efficient concurrency
            {
#if OPENMP_AVAILABLE_RANSAC_
#pragma omp critical(update) // n_best_inliers_count, model_, model_coefficients.values, k are shared and read/write must be protected
#endif
                {
                    // Better match ?
                    if (n_inliers_count > n_best_inliers_count)
                    {
                        n_best_inliers_count = n_inliers_count; // This write and the previous read of n_best_inliers_count must be consecutive and must not be interrupted!
                        n_best_inliers_count_tmp = n_best_inliers_count;

                        model_coefficients = *model_coefficients_temp;

                        // Compute the k parameter (k=std::log(z)/std::log(1-w^n))
                        const double w = static_cast<double>(n_best_inliers_count) * one_over_indices;
                        double p_outliers = 1.0 - std::pow(w, static_cast<double>(selection.size()));      // Probability that selection is contaminated by at least one outlier
                        p_outliers = (std::max)(std::numeric_limits<double>::epsilon(), p_outliers);       // Avoid division by -Inf
                        p_outliers = (std::min)(1.0 - std::numeric_limits<double>::epsilon(), p_outliers); // Avoid division by 0.
                        k = log_probability / std::log(p_outliers);

                        ROS_INFO("k : %d", k);
                        ROS_INFO("k : %f", p_outliers);
                    }
                } // omp critical
            }

            int iterations_tmp;
            double k_tmp;
#if OPENMP_AVAILABLE_RANSAC_
#pragma omp atomic capture
#endif
            iterations_tmp = ++iterations;
#if OPENMP_AVAILABLE_RANSAC_
#pragma omp atomic read
#endif
            k_tmp = k;
#if OPENMP_AVAILABLE_RANSAC_
            PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] Trial %d out of %f: %u inliers (best is: %u so far) (thread %d).\n", iterations_tmp, k_tmp, n_inliers_count, n_best_inliers_count_tmp, omp_get_thread_num());
#else
            PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] Trial %d out of %f: %u inliers (best is: %u so far).\n", iterations_tmp, k_tmp, n_inliers_count, n_best_inliers_count_tmp);
#endif
            if (iterations_tmp > k_tmp)
                break;
            if (iterations_tmp > max_iterations)
            {
                PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] RANSAC reached the maximum number of trials.\n");
                break;
            }
        } // while
    }     // omp parallel

    // Get the set of inliers that correspond to the best model found so far
    SelectWithinDistance(cloud, model_coefficients, threshold_, inliers);

    return (true);
    // getSamples
}

void OptimizeModelCoefficients(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                               pcl::Indices &inliers,
                               pcl::ModelCoefficients &model_coefficients,
                               pcl::ModelCoefficients &coeff_refined)
{
    Eigen::Vector4f plane_parameters;

    // Use Least-Squares to fit the plane through all the given sample points and find out its coefficients
    Eigen::Matrix3f covariance_matrix;
    Eigen::Vector4f xyz_centroid;

    if (0 == pcl::computeMeanAndCovarianceMatrix(*cloud, inliers, covariance_matrix, xyz_centroid))
    {
        PCL_ERROR("[pcl::SampleConsensusModelPlane::optimizeModelCoefficients] computeMeanAndCovarianceMatrix failed (returned 0) because there are no valid inliers.\n");
        coeff_refined = model_coefficients;
        return;
    }

    // Compute the model coefficients
    Eigen::Vector3f::Scalar eigen_value;
    Eigen::Vector3f eigen_vector;
    pcl::eigen33(covariance_matrix, eigen_value, eigen_vector);

    // Hessian form (D = nc . p_plane (centroid here) + p)
    coeff_refined.values.resize(1);
    coeff_refined.values[0] = eigen_vector[0];
    coeff_refined.values[1] = eigen_vector[1];
    coeff_refined.values[2] = eigen_vector[2];
    coeff_refined.values[3] = 0.0f;
    coeff_refined.values[3] = -1.0f * (coeff_refined.values[0] * xyz_centroid(0) + coeff_refined.values[1] * xyz_centroid(1) + coeff_refined.values[2] * xyz_centroid(2) + coeff_refined.values[3] * xyz_centroid(3));

    // coeff_refined = model_coefficients;
    return;
}

void CuboidRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                  float thresh,
                  int max_iteration,
                  std::vector<Eigen::Vector4f> &best_eqs,
                  int &best_num_inliers)
{
    pcl::Indices inliers;
    pcl::ModelCoefficients::Ptr model_coefficients(new pcl::ModelCoefficients);
    Eigen::Vector4f plane;

    if (!ComputeModel(cloud, inliers, *model_coefficients, max_iteration, thresh, best_num_inliers))
    {
        inliers.clear();
        model_coefficients->values.clear();
        return;
    }

    // optimize

    pcl::ModelCoefficients::Ptr coeff_refined(new pcl::ModelCoefficients);
    OptimizeModelCoefficients(cloud, inliers, *model_coefficients, *coeff_refined);
    model_coefficients->values.resize(coeff_refined->values.size());
    memcpy(model_coefficients->values.data(), coeff_refined->values.data(), coeff_refined->values.size() * sizeof(float));
    // Refine inliers
    SelectWithinDistance(cloud, *coeff_refined, thresh, inliers);

    for (int i = 0; i < 4; i++)
    {
        plane(i) = coeff_refined->values[i];
    }

    best_eqs.push_back(plane);

    return;
}