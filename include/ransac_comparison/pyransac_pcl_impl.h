#define OPENMP_AVAILABLE_RANSAC_ true

#include <pmmintrin.h>
#include <immintrin.h>

template <typename PointT>
inline __m256 pcl::Pyransac<PointT>::dist8(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_,
                                    const std::size_t i, const __m256 &a_vec, const __m256 &b_vec, const __m256 &c_vec, const __m256 &d_vec, const __m256 &abs_help) const
{
    // The andnot-function realizes an abs-operation: the sign bit is removed
    return _mm256_andnot_ps(abs_help,
                            _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(a_vec, _mm256_set_ps((*cloud_)[i].x, (*cloud_)[i + 1].x, (*cloud_)[i + 2].x, (*cloud_)[i + 3].x, (*cloud_)[i + 4].x, (*cloud_)[i + 5].x, (*cloud_)[i + 6].x, (*cloud_)[i + 7].x)),
                                                        _mm256_mul_ps(b_vec, _mm256_set_ps((*cloud_)[i].y, (*cloud_)[i + 1].y, (*cloud_)[i + 2].y, (*cloud_)[i + 3].y, (*cloud_)[i + 4].y, (*cloud_)[i + 5].y, (*cloud_)[i + 6].y, (*cloud_)[i + 7].y))),
                                          _mm256_add_ps(_mm256_mul_ps(c_vec, _mm256_set_ps((*cloud_)[i].z, (*cloud_)[i + 1].z, (*cloud_)[i + 2].z, (*cloud_)[i + 3].z, (*cloud_)[i + 4].z, (*cloud_)[i + 5].z, (*cloud_)[i + 6].z, (*cloud_)[i + 7].z)),
                                                        d_vec))); // TODO this could be replaced by three fmadd-instructions (if available), but the speed gain would probably be minimal
}

template <typename PointT>
void pcl::Pyransac<PointT>::SelectWithinDistance(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_,
                                                 std::vector<Eigen::Vector4f> &model_coefficients,
                                                 float threshold_,
                                                 pcl::Indices &inliers)
{
    inliers.clear();
    inliers.reserve(cloud_->size());

    for (std::size_t i = 0; i < cloud_->size(); ++i)
    {
        // Calculate the distance from the point to the plane normal as the dot product
        // D = (P-A).N/|N|
        float min_distance = std::numeric_limits<double>::max();
        Eigen::Vector4f pt((*cloud_)[i].x,
                           (*cloud_)[i].y,
                           (*cloud_)[i].z,
                           1.0f);
        for (std::size_t j = 0; j < model_coefficients.size(); j++)
        {
            float distance = std::abs(model_coefficients[j].dot(pt));

            min_distance = min_distance < distance ? min_distance : distance;
        }

        if (min_distance < threshold_)
        {
            // Returns the indices of the points whose distances are smaller than the threshold
            inliers.push_back(i);
        }
    }
}

template <typename PointT>
std::size_t pcl::Pyransac<PointT>::CountWithinDistance(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_,
                                                       std::vector<Eigen::Vector4f> &model_coefficients,
                                                       float threshold_,
                                                       std::size_t i)

{
    std::size_t nr_p = 0;

    // Iterate through the 3d points and calculate the distances from them to the line
    for (; i < indices_->size(); ++i)
    {
        // Calculate the distance from the point to the line
        // D = ||(P2-P1) x (P1-P0)|| / ||P2-P1|| = norm (cross (p2-p1, p2-p0)) / norm(p2-p1)
        float min_distance = std::numeric_limits<double>::max();
        Eigen::Vector4f pt((*cloud_)[i].x,
                           (*cloud_)[i].y,
                           (*cloud_)[i].z,
                           1.0f);
        for (std::size_t j = 0; j < model_coefficients.size(); j++)
        {

            float distance = std::abs(model_coefficients[j].dot(pt));

            min_distance = min_distance < distance ? min_distance : distance;
        }
        if (min_distance < threshold_)
            nr_p++;
    }
    return (nr_p);
}

template <typename PointT>
std::size_t pcl::Pyransac<PointT>::CountWithinDistanceAVX(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_,
                                                          std::vector<Eigen::Vector4f> &model_coefficients,
                                                          float threshold_,
                                                          std::size_t i)
{
    Eigen::Vector4f model_coefficient = model_coefficients[0];
    std::size_t nr_p = 0;
    const __m256 a_vec = _mm256_set1_ps(model_coefficient[0]);
    const __m256 b_vec = _mm256_set1_ps(model_coefficient[1]);
    const __m256 c_vec = _mm256_set1_ps(model_coefficient[2]);
    const __m256 d_vec = _mm256_set1_ps(model_coefficient[3]);
    const __m256 threshold_vec = _mm256_set1_ps(threshold_);
    const __m256 abs_help = _mm256_set1_ps(-0.0F); // -0.0F (negative zero) means that all bits are 0, only the sign bit is 1
    __m256i res = _mm256_set1_epi32(0);            // This corresponds to nr_p: 8 32bit integers that, summed together, hold the number of inliers
    for (; (i + 8) <= cloud_->size(); i += 8)
    {
        const __m256 mask = _mm256_cmp_ps(dist8(cloud_, i, a_vec, b_vec, c_vec, d_vec, abs_help), threshold_vec, _CMP_LT_OQ); // The mask contains 1 bits if the corresponding points are inliers, else 0 bits
        res = _mm256_add_epi32(res, _mm256_and_si256(_mm256_set1_epi32(1), _mm256_castps_si256(mask)));                                                       // The latter part creates a vector with ones (as 32bit integers) where the points are inliers
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
    nr_p += CountWithinDistance(cloud_, model_coefficients, threshold_, i);
    return (nr_p);
}

template <typename PointT>
bool pcl::Pyransac<PointT>::isSampleGood(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::Indices &samples)
{
    if (
        std::abs((*cloud)[samples[0]].x - (*cloud)[samples[1]].x) <= std::numeric_limits<float>::epsilon() &&
        std::abs((*cloud)[samples[0]].y - (*cloud)[samples[1]].y) <= std::numeric_limits<float>::epsilon() &&
        std::abs((*cloud)[samples[0]].z - (*cloud)[samples[1]].z) <= std::numeric_limits<float>::epsilon())
        return (false);

    return (true);
}

template <typename PointT>
int pcl::Pyransac<PointT>::rnd()
{
    return (*rng_gen_)();
}

template <typename PointT>
void pcl::Pyransac<PointT>::GetSamples(int iterations, pcl::Indices &samples, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
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

    int sample_size = 3;
    // samples.clear();
    int index_size = object_indices->size();
    for (unsigned int iter = 0; iter < sample_size * 10; ++iter)
    {
        // Choose the random indices
        for (std::size_t i = 0; i < sample_size; ++i)
        {
            std::swap((*object_indices)[i], (*object_indices)[i + (rnd() % (index_size - i))]);
        }
        std::copy((*object_indices).begin(), (*object_indices).begin() + sample_size, samples.begin());

        // If it's a good sample, stop here
        if (isSampleGood(cloud, samples))
        {
            return;
        }
    }
}

template <typename PointT>
bool pcl::Pyransac<PointT>::ComputeModelCoefficients(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_,
                                                     pcl::Indices &samples,
                                                     std::vector<Eigen::Vector4f> &model_coefficients)
{
    // First Plane
    Eigen::Vector3f p0 = (*cloud_)[samples.at(0)].getVector3fMap();
    Eigen::Vector3f p1 = (*cloud_)[samples.at(1)].getVector3fMap();
    Eigen::Vector3f p2 = (*cloud_)[samples.at(2)].getVector3fMap();

    const Eigen::Vector3f cross = (p1 - p0).cross(p2 - p0);
    const float crossNorm = cross.stableNorm();

    // Checking for collinearity here
    if (crossNorm < Eigen::NumTraits<float>::dummy_precision())
    {
        PCL_ERROR("[pcl::SampleConsensusModelPlane::computeModelCoefficients] Chosen samples are collinear!\n");
        return (false);
    }

    // Compute the plane coefficients from the 3 given points in a straightforward manner
    // calculate the plane normal n = (p2-p1) x (p3-p1) = cross (p2-p1, p3-p1)
    Eigen::Vector4f model_coefficient1;
    model_coefficient1.resize(4);
    model_coefficient1.template head<3>() = cross / crossNorm;

    // ... + d = 0
    model_coefficient1[3] = -1.0f * (model_coefficient1.template head<3>().dot(p0));

    PCL_DEBUG("[pcl::SampleConsensusModelPlane::computeModelCoefficients] Model is (%g,%g,%g,%g).\n",
              model_coefficient1[0], model_coefficient1[1], model_coefficient1[2], model_coefficient1[3]);

    model_coefficients.push_back(model_coefficient1);
    return (true);

    // // Second Plane
    // pcl::ModelCoefficients::Ptr model_coefficients2(new pcl::ModelCoefficients);
    // model_coefficients2->values.resize(4);
    // double Dx3, Dy3, Dz3, Dx4, Dy4, Dz4, Dy3Dy4;

    // Eigen::Vector3f pt4(cloud_->points.at(samples.at(3)).x,
    //                     cloud_->points.at(samples.at(3)).y,
    //                     cloud_->points.at(samples.at(3)).z);

    // double dist_plane = (
    //     model_coefficients1->values[0] * cloud_->points.at(samples.at(3)).x
    //      + model_coefficients1->values[1] * cloud_->points.at(samples.at(3)).y
    //      + model_coefficients1->values[2] * cloud_->points.at(samples.at(3)).z
    //      + model_coefficients1->values[3]) / n_norm1;

    // Eigen::Vector3f pt5(cloud_->points.at(samples.at(4)).x,
    //                     cloud_->points.at(samples.at(4)).y,
    //                     cloud_->points.at(samples.at(4)).z);

    // Eigen::Vector3f vecC(model_coefficients1->values[0],
    //                      model_coefficients1->values[1],
    //                      model_coefficients1->values[2]);
    // vecC.normalize();
    // Eigen::Vector3f vecD = -dist_plane * vecC;
    // Eigen::Vector3f vecE = pt5 - pt4;
    // Eigen::Vector3f vecF = vecD.cross(vecE);
    // vecF.normalize();

    // model_coefficients2->values[0] = vecF(0);
    // model_coefficients2->values[1] = vecF(1);
    // model_coefficients2->values[2] = vecF(2);
    // model_coefficients2->values[3] = -vecF.dot(pt5);

    // model_coefficients.push_back(model_coefficients2);

    // // Third Plane
    // pcl::ModelCoefficients::Ptr model_coefficients3(new pcl::ModelCoefficients);
    // model_coefficients3->values.resize(4);

    // Eigen::Vector3f vecG = vecC.cross(vecF);
    // Eigen::Vector3f pt6(cloud_->points.at(samples.at(5)).x,
    //                     cloud_->points.at(samples.at(5)).y,
    //                     cloud_->points.at(samples.at(5)).z);

    // model_coefficients3->values[0] = vecG(0);
    // model_coefficients3->values[1] = vecG(1);
    // model_coefficients3->values[2] = vecG(2);
    // model_coefficients3->values[3] = -vecG.dot(pt6);
    // model_coefficients.push_back(model_coefficients3);

    // return (true);
}

template <typename PointT>
bool pcl::Pyransac<PointT>::ComputeModel(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                         pcl::Indices &inliers,
                                         std::vector<Eigen::Vector4f> &model_coefficients,
                                         float threshold_,
                                         int &n_best_inliers_count)
{
    // Warn and exit if no threshold was set
    if (threshold_ == std::numeric_limits<double>::max())
    {
        PCL_ERROR("[pcl::RandomSampleConsensus::computeModel] No threshold set!\n");
        return (false);
    }

    int iterations_ = 0;
    double k = std::numeric_limits<double>::max();

    pcl::Indices selection(6); // random sampled indices (6)
    std::vector<Eigen::Vector4f> model_coefficients_temp;
    // pcl::ModelCoefficients::Ptr model_coefficients_temp(new pcl::ModelCoefficients);

    const double log_probability = std::log(1.0 - probability_);
    const double one_over_indices = 1.0 / static_cast<double>(cloud->size());

    unsigned skipped_count = 0;

    // suppress infinite loops by just allowing 10 x maximum allowed iterations for invalid model parameters!
    const unsigned max_skip = max_iterations_ * 10;

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
        while (1) // infinite loop with four possible breaks
        {
            model_coefficients_temp.clear();
            // Get X samples which satisfy the model criteria
#if OPENMP_AVAILABLE_RANSAC_
#pragma omp critical(samples)
#endif
            {
                double start = ros::Time::now().toNSec();
                GetSamples(iterations_, selection, cloud); // The random number generator used when choosing the samples should not be called in parallel
                double end = ros::Time::now().toNSec();
                std::cout << "Sampling : " << (end - start) / 1000000 << " ms" << std::endl;
            }

            if (selection.empty())
            {
                ROS_INFO("[pcl::RandomSampleConsensus::computeModel] No samples could be selected!\n");
                break;
            }
            double start = ros::Time::now().toNSec();
            // Search for inliers in the point cloud for the current plane model M
            if (!ComputeModelCoefficients(cloud, selection, model_coefficients_temp)) // This function has to be thread-safe
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
            double end = ros::Time::now().toNSec();
            std::cout << "ComputeModelCoefficients : " << (end - start) / 1000000 << " ms" << std::endl;
            // Select the inliers that are within threshold_ from the model
            // sac_model_->selectWithinDistance (model_coefficients, threshold_, inliers);
            // if (inliers.empty () && k > 1.0)
            //  continue;
            double start_CountWithinDistance = ros::Time::now().toNSec();
            std::size_t n_inliers_count = CountWithinDistanceAVX(cloud, model_coefficients_temp, threshold_); // This functions has to be thread-safe. Most work is done here
            std::size_t n_best_inliers_count_tmp;
            double end_CountWithinDistance = ros::Time::now().toNSec();
            std::cout << "CountWithinDistance : " << (end_CountWithinDistance - start_CountWithinDistance) / 1000000 << " ms" << std::endl;
#if OPENMP_AVAILABLE_RANSAC_
#pragma omp atomic read
#endif
            n_best_inliers_count_tmp = n_best_inliers_count;

            if (n_inliers_count > n_best_inliers_count_tmp) // This condition is false most of the time, and the critical region is not entered, hopefully leading to more efficient concurrency
            {
#if OPENMP_AVAILABLE_RANSAC_
#pragma omp critical(update) // n_best_inliers_count, model_, model_coefficients_, k are shared and read/write must be protected
#endif
                {
                    // Better match ?
                    if (n_inliers_count > n_best_inliers_count)
                    {
                        n_best_inliers_count = n_inliers_count; // This write and the previous read of n_best_inliers_count must be consecutive and must not be interrupted!
                        n_best_inliers_count_tmp = n_best_inliers_count;

                        // Save the current model/inlier/coefficients selection as being the best so far
                        model_ = selection;
                        model_coefficients = model_coefficients_temp;

                        // Compute the k parameter (k=std::log(z)/std::log(1-w^n))
                        const double w = static_cast<double>(n_best_inliers_count) * one_over_indices;
                        double p_outliers = 1.0 - std::pow(w, static_cast<double>(selection.size())); // Probability that selection is contaminated by at least one outlier
                        p_outliers = (std::max)(std::numeric_limits<double>::epsilon(), p_outliers);  // Avoid division by -Inf

                        p_outliers = (std::min)(1.0 - std::numeric_limits<double>::epsilon(), p_outliers); // Avoid division by 0.
                        k = 3 * log_probability / std::log(p_outliers);
                    }
                } // omp critical
            }

            int iterations_tmp;
            double k_tmp;
#if OPENMP_AVAILABLE_RANSAC_
#pragma omp atomic capture
#endif
            iterations_tmp = ++iterations_;
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
            {
                // ROS_INFO("Stop by k");
                break;
            }
            if (iterations_tmp > max_iterations_)
            {
                // ROS_INFO("Stop by iteration");
                PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] RANSAC reached the maximum number of trials.\n");
                break;
            }
        } // while
    }     // omp parallel

    PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] Model: %lu size, %u inliers.\n", model_.size(), n_best_inliers_count);

    if (model_.empty())
    {
        PCL_ERROR("[pcl::RandomSampleConsensus::computeModel] RANSAC found no model.\n");
        inliers.clear();
        return (false);
    }

    // Get the set of inliers that correspond to the best model found so far
    double start_SelectWithinDistance = ros::Time::now().toNSec();
    SelectWithinDistance(cloud, model_coefficients, threshold_, inliers);
    double end_SelectWithinDistance = ros::Time::now().toNSec();
    std::cout << "SelectWithinDistance : " << (end_SelectWithinDistance - start_SelectWithinDistance) / 1000000 << " ms" << std::endl;
    return (true);
}

template <typename PointT>
void pcl::Pyransac<PointT>::OptimizeModelCoefficients(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
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

template <typename PointT>
void pcl::Pyransac<PointT>::CuboidRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                         float thresh,
                                         std::vector<pcl::ModelCoefficients::Ptr> &model_coefficients,
                                         std::vector<Eigen::Vector4f> &best_eqs,
                                         int &best_num_inliers)
{
    pcl::Indices inliers;
    // std::vector<pcl::ModelCoefficients::Ptr> model_coefficients;
    Eigen::Vector4f plane;

    if (!ComputeModel(cloud, inliers, best_eqs, thresh, best_num_inliers))
    {
        inliers.clear();
        best_eqs.clear();
        return;
    }

    // optimize

    // pcl::ModelCoefficients::Ptr coeff_refined(new pcl::ModelCoefficients);
    // OptimizeModelCoefficients(cloud, inliers, *model_coefficients, *coeff_refined);
    // model_coefficients->values.resize(coeff_refined->values.size());
    // memcpy(model_coefficients->values.data(), coeff_refined->values.data(), coeff_refined->values.size() * sizeof(float));
    // // Refine inliers
    // SelectWithinDistance(cloud, *coeff_refined, thresh, inliers);

    // for (int i = 0; i < 4; i++)
    // {
    //     plane(i) = coeff_refined->values[i];
    // }
    // ROS_INFO("adad");

    // for (int j = 0; j < model_coefficients.size(); j++)
    // {
    //     for (int i = 0; i < 4; i++)
    //     {
    //         plane(i) = model_coefficients[j]->values[i];
    //     }
    //     best_eqs.push_back(plane);
    // }
    return;
}
