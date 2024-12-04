#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cassert>

class CameraIntrinsic {
public:
    int width;
    int height;
    cv::Mat cam_k;
    cv::Mat cam_dist;
    CameraIntrinsic() = default;
    CameraIntrinsic(int width_, int height_, const cv::Mat& cam_k_, const cv::Mat& cam_dist_);
};

class HomographyTransformer {
public:
    cv::Mat H;
    int new_plane_width;
    int new_plane_height;
    int multiplication_ratio;
    int starting_x;
    int starting_y;
    CameraIntrinsic intrinsic;
    cv::Mat depth_map;
    cv::Mat mapping;

    HomographyTransformer(int new_plane_width_, int new_plane_height_, const CameraIntrinsic& intrinsic_,
                          const cv::Mat& depth_map_ = cv::Mat(), int multiplication_ratio_ = 96,
                          int starting_x_ = 0, int starting_y_ = 0);

    void compute_homography(const std::vector<cv::Point2f>& image_points,
                            const std::vector<cv::Point2f>& ground_points);

    void init_mapping();

    cv::Mat undistort_image(const cv::Mat& image);

    cv::Mat transform_image(const cv::Mat& origin_plane);
};

cv::Mat preprocess_image(const cv::Mat& inputImage,  const size_t& imageWidth, const size_t& imageHeight);

void select_correspondences(const cv::Mat& depth_map, const CameraIntrinsic& intrinsic,
                            std::vector<cv::Point2f>& image_points,
                            std::vector<cv::Point2f>& ground_points);
