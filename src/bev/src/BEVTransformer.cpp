#include "BEVTransformer.hpp"

#ifdef BEV_DEBUG_OUTPUT
    #define DEBUG_LOG(msg) std::cout << msg << std::endl
#else
    #define DEBUG_LOG(msg)
#endif


CameraIntrinsic::CameraIntrinsic(int width_, int height_, const cv::Mat& cam_k_, const cv::Mat& cam_dist_)
    : width(width_), height(height_), cam_k(cam_k_.clone()), cam_dist(cam_dist_.clone()) {}


HomographyTransformer::HomographyTransformer(int new_plane_width_, int new_plane_height_, const CameraIntrinsic& intrinsic_,
                                             const cv::Mat& depth_map_, int multiplication_ratio_,
                                             int starting_x_, int starting_y_)
    : new_plane_width(new_plane_width_), new_plane_height(new_plane_height_),
      multiplication_ratio(multiplication_ratio_), starting_x(starting_x_),
      starting_y(starting_y_), intrinsic(intrinsic_), depth_map(depth_map_.clone())
{
    H = cv::Mat();
    mapping = cv::Mat();
}

cv::Mat HomographyTransformer::get_homography_mat() {
    return H;
}

void HomographyTransformer::compute_homography(const std::vector<cv::Point2f>& image_points,
                                               const std::vector<cv::Point2f>& ground_points)
{
    assert(image_points.size() == ground_points.size() && "Error: Image points and ground points must have the same size.");
    cv::Mat status;
    H = cv::findHomography(ground_points, image_points, cv::RANSAC, 3, status);
    // if (!H.empty()) {
    //     init_mapping();
    // }
}

void HomographyTransformer::init_mapping() {
    mapping = cv::Mat(3, new_plane_width * new_plane_height, CV_64F, cv::Scalar(0));

    tbb::parallel_for(tbb::blocked_range<int>(0, new_plane_width),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i < r.end(); ++i) {
                for (int j = 0; j < new_plane_height; ++j) {
                    int idx = i * new_plane_height + j;
                    mapping.at<double>(0, idx) = (i + starting_x) / static_cast<double>(multiplication_ratio);
                    mapping.at<double>(1, idx) = (j + starting_y) / static_cast<double>(multiplication_ratio);
                    mapping.at<double>(2, idx) = 1.0;
                }
            }
        }
    );

    mapping = H * mapping;  
}

cv::Mat HomographyTransformer::undistort_image(const cv::Mat& image) {
    cv::Mat map1, map2;
    cv::Size size(intrinsic.width, intrinsic.height);
    cv::Mat identity = cv::Mat::eye(3, 3, CV_32F);
    cv::initUndistortRectifyMap(intrinsic.cam_k, intrinsic.cam_dist, identity, intrinsic.cam_k, size, CV_32FC1, map1, map2);
    cv::Mat dst;
    cv::remap(image, dst, map1, map2, cv::INTER_LINEAR);
    return dst;
}

cv::Mat HomographyTransformer::transform_image(const cv::Mat& origin_plane) {
    cv::Mat new_plane(new_plane_height, new_plane_width, origin_plane.type(), cv::Scalar::all(0));
    int height = origin_plane.rows;
    int width = origin_plane.cols;
    const int total = new_plane_width * new_plane_height;

    tbb::parallel_for(tbb::blocked_range<int>(0, total),
        [&](const tbb::blocked_range<int>& range) {
            for (int idx = range.begin(); idx != range.end(); ++idx) {
                double mapped_x = mapping.at<double>(0, idx);
                double mapped_y = mapping.at<double>(1, idx);
                double mapped_z = mapping.at<double>(2, idx);

                int new_x = -1;
                int new_y = -1;
                if (mapped_z != 0) {
                    new_x = static_cast<int>(std::round(mapped_x / mapped_z));
                    new_y = static_cast<int>(std::round(mapped_y / mapped_z));
                }

                int i = idx / new_plane_height;
                int j = idx % new_plane_height;

                if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                    new_plane.at<cv::Vec3b>(j, i) = origin_plane.at<cv::Vec3b>(new_y, new_x);
                } else {
                    new_plane.at<cv::Vec3b>(j, i) = cv::Vec3b(0, 0, 0);
                }
            }
        }
    );

    return new_plane;
}

cv::Mat preprocess_image(const cv::Mat& inputImage, const size_t& imageWidth, const size_t& imageHeight) {
    cv::Mat imageRgb;
    cv::cvtColor(inputImage, imageRgb, cv::COLOR_RGB2BGR);
    imageRgb.convertTo(imageRgb, CV_32FC3, 1.0f / 255.0f);

    cv::Vec3f mean(0.485f, 0.456f, 0.406f);
    cv::Vec3f stddev(0.229f, 0.224f, 0.225f);

    // Parallel normalization
    tbb::parallel_for(tbb::blocked_range<int>(0, imageRgb.rows),
        [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                cv::Vec3f* row_ptr = imageRgb.ptr<cv::Vec3f>(i);
                for (int j = 0; j < imageRgb.cols; ++j) {
                    row_ptr[j][0] = (row_ptr[j][0] - mean[0]) / stddev[0];
                    row_ptr[j][1] = (row_ptr[j][1] - mean[1]) / stddev[1];
                    row_ptr[j][2] = (row_ptr[j][2] - mean[2]) / stddev[2];
                }
            }
        }
    );

    cv::Mat resizedImage;
    cv::resize(imageRgb, resizedImage, cv::Size(static_cast<int>(imageWidth), static_cast<int>(imageHeight)));

    std::vector<cv::Mat> chwChannels(3);
    cv::split(resizedImage, chwChannels);
    cv::Mat chwImage;
    cv::vconcat(chwChannels, chwImage);

    return chwImage;
}

void select_correspondences(const cv::Mat& depth_map, const CameraIntrinsic& intrinsic,
                            std::vector<cv::Point2f>& image_points,
                            std::vector<cv::Point2f>& ground_points)
{
    int height = depth_map.rows;
    int width = depth_map.cols;
    cv::Mat K_inv = intrinsic.cam_k.inv();

    // Define a fixed trapezoid in the image
    std::vector<cv::Point2f> trapezoid = {
        cv::Point2f(width * 0.3f, height * 0.6f), // Top-left
        cv::Point2f(width * 0.7f, height * 0.6f), // Top-right
        cv::Point2f(width * 0.9f, height * 0.9f), // Bottom-right
        cv::Point2f(width * 0.1f, height * 0.9f)  // Bottom-left
    };

    cv::Mat trapezoid_mask = cv::Mat::zeros(height, width, CV_8UC1);
    std::vector<std::vector<cv::Point>> trapezoid_pts = {
        { cv::Point(static_cast<int>(trapezoid[0].x), static_cast<int>(trapezoid[0].y)),
          cv::Point(static_cast<int>(trapezoid[1].x), static_cast<int>(trapezoid[1].y)),
          cv::Point(static_cast<int>(trapezoid[2].x), static_cast<int>(trapezoid[2].y)),
          cv::Point(static_cast<int>(trapezoid[3].x), static_cast<int>(trapezoid[3].y)) }
    };
    cv::fillPoly(trapezoid_mask, trapezoid_pts, cv::Scalar(255));

    const int row_step = 20; 
    const int col_step = 20;

    // Use concurrent vectors to avoid locking overhead
    tbb::concurrent_vector<cv::Point2f> img_pts_concurrent;
    tbb::concurrent_vector<cv::Point2f> grd_pts_concurrent;

    tbb::parallel_for(tbb::blocked_range<int>(0, height, row_step),
        [&](const tbb::blocked_range<int>& row_range) {
            cv::Mat pixel(3, 1, CV_64F);
            for (int v = row_range.begin(); v < row_range.end(); v += row_step) {
                for (int u = 0; u < width; u += col_step) {
                    if (trapezoid_mask.at<uchar>(v, u) == 0) continue;

                    float d = depth_map.at<float>(v, u);
                    if (d <= 0) continue;

                    pixel.at<double>(0,0) = static_cast<double>(u);
                    pixel.at<double>(1,0) = static_cast<double>(v);
                    pixel.at<double>(2,0) = 1.0;

                    cv::Mat cam_coords = d * K_inv * pixel;
                    double X_c = cam_coords.at<double>(0,0);
                    double Y_c = cam_coords.at<double>(1,0);
                    double Z_c = cam_coords.at<double>(2,0);

                    if (Z_c <= 0) continue; // Ignore points behind the camera

                    double X_w = X_c;
                    double Z_w = Z_c;

                    grd_pts_concurrent.push_back(cv::Point2f(static_cast<float>(X_w), static_cast<float>(Z_w)));
                    img_pts_concurrent.push_back(cv::Point2f(static_cast<float>(u), static_cast<float>(v)));
                }
            }
        }
    );

    // Move concurrent_vector to standard vector
    image_points.assign(img_pts_concurrent.begin(), img_pts_concurrent.end());
    ground_points.assign(grd_pts_concurrent.begin(), grd_pts_concurrent.end());
}
