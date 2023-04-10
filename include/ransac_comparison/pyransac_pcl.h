#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/random_sample.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>

#include <boost/random.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <ctime>
#include <omp.h>
#include <ros/ros.h>

namespace pcl
{
    template <typename PointT>
    class Pyransac : public SACSegmentation<PointT>
    {

    public:
        using SACSegmentation<PointT>::input_;
        using SACSegmentation<PointT>::indices_;

        using PointCloud = pcl::PointCloud<PointT>;
        using PointCloudPtr = typename PointCloud::Ptr;
        using PointCloudConstPtr = typename PointCloud::ConstPtr;
        using SearchPtr = typename pcl::search::Search<PointT>::Ptr;

        using SampleConsensusPtr = typename SampleConsensus<PointT>::Ptr;
        using SampleConsensusModelPtr = typename SampleConsensusModel<PointT>::Ptr;

        Pyransac(bool random = false)
            : model_(), sac_(), model_type_(-1), method_type_(0), threshold_(0), optimize_coefficients_(true), radius_min_(-std::numeric_limits<double>::max()), radius_max_(std::numeric_limits<double>::max()), samples_radius_(0.0), samples_radius_search_(), eps_angle_(0.0), axis_(Eigen::Vector3f::Zero()), max_iterations_(50), threads_(-1), probability_(0.99), random_(random), rng_dist_(new boost::uniform_int<>(0, std::numeric_limits<int>::max()))
        {
            if (random)
                rng_alg_.seed(static_cast<unsigned>(std::time(nullptr)));
            else
                rng_alg_.seed(12345u);

            rng_gen_.reset(new boost::variate_generator<boost::mt19937 &, boost::uniform_int<>>(rng_alg_, *rng_dist_));
        }
        ~Pyransac() override = default;

        void SelectWithinDistance(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_,
                                  std::vector<pcl::ModelCoefficients::Ptr> &model_coefficients,
                                  float threshold_,
                                  pcl::Indices &inliers);

        std::size_t CountWithinDistance(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_,
                                        std::vector<pcl::ModelCoefficients::Ptr> &model_coefficients,
                                        float threshold_);

        void SetProbability(float prob) { probability_ = prob; }

        void setMaxIterations(int max_iteration) { max_iterations_ = max_iteration; }

        bool isSampleGood(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::Indices &samples);

        void GetSamples(int iterations, pcl::Indices &samples, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

        bool ComputeModelCoefficients(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_,
                                      pcl::Indices &samples,
                                      std::vector<pcl::ModelCoefficients::Ptr> &model_coefficients);

        bool ComputeModel(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                          pcl::Indices &inliers,
                          std::vector<pcl::ModelCoefficients::Ptr> &model_coefficients,
                          float threshold_,
                          int &n_best_inliers_count);

        void OptimizeModelCoefficients(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                       pcl::Indices &inliers,
                                       pcl::ModelCoefficients &model_coefficients,
                                       pcl::ModelCoefficients &coeff_refined);

        void CuboidRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                          float thresh,
                          std::vector<pcl::ModelCoefficients::Ptr>& model_coefficients,
                          std::vector<Eigen::Vector4f> &best_eqs,
                          int &best_num_inliers);

        int rnd();

    protected:
        /** \brief The model that needs to be segmented. */
        pcl::Indices model_;

        /** \brief The sample consensus segmentation method. */
        SampleConsensusPtr sac_;

        /** \brief The type of model to use (user given parameter). */
        int model_type_;

        /** \brief The type of sample consensus method to use (user given parameter). */
        int method_type_;

        /** \brief Distance to the model threshold (user given parameter). */
        double threshold_;

        /** \brief Set to true if a coefficient refinement is required. */
        bool optimize_coefficients_;

        /** \brief The minimum and maximum radius limits for the model. Applicable to all models that estimate a radius. */
        double radius_min_, radius_max_;

        /** \brief The maximum distance of subsequent samples from the first (radius search) */
        double samples_radius_;

        /** \brief The search object for picking subsequent samples using radius search */
        SearchPtr samples_radius_search_;

        /** \brief The maximum allowed difference between the model normal and the given axis. */
        double eps_angle_;

        /** \brief The axis along which we need to search for a model perpendicular to. */
        Eigen::Vector3f axis_;

        /** \brief Maximum number of iterations before giving up (user given parameter). */
        int max_iterations_;

        /** \brief The number of threads the scheduler should use, or a negative number if no parallelization is wanted. */
        int threads_;

        /** \brief Desired probability of choosing at least one sample free from outliers (user given parameter). */
        double probability_;

        /** \brief Set to true if we need a random seed. */
        bool random_;

        /** \brief Boost-based random number generator algorithm. */
        boost::mt19937 rng_alg_;

        /** \brief Boost-based random number generator distribution. */
        std::shared_ptr<boost::uniform_int<>> rng_dist_;

        /** \brief Boost-based random number generator. */
        std::shared_ptr<boost::variate_generator<boost::mt19937 &, boost::uniform_int<>>> rng_gen_;
    };
}

#include "pyransac_pcl_impl.h"