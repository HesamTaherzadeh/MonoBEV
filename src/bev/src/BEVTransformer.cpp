#include "BEVTransformer.hpp"


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

cv::Mat HomographyTransformer::get_homography_mat(){
    return H;
} 

void HomographyTransformer::compute_homography(const std::vector<cv::Point2f>& image_points,
                                               const std::vector<cv::Point2f>& ground_points)
{
    assert(image_points.size() == ground_points.size() && "Error: Image points and ground points must have the same size.");
    std::cout << "The image points size and ground pointes are : " << image_points.size() << std::endl;
    cv::Mat status;
    H = cv::findHomography(ground_points, image_points, cv::RANSAC, 3, status);
    std::cout <<"Homography matrix" <<  H << std::endl;
    if (!H.empty()) {
        std::cout << "Homography matrix computed." << std::endl;
        init_mapping();
    } else {
        std::cout << "Error computing homography." << std::endl;
    }
}

void HomographyTransformer::init_mapping()
{
    mapping = cv::Mat(3, new_plane_width * new_plane_height, CV_64F, cv::Scalar(0));

    for (int i = 0; i < new_plane_width; ++i) {
        for (int j = 0; j < new_plane_height; ++j) {
            int idx = i * new_plane_height + j;
            mapping.at<double>(0, idx) = (i + starting_x) / static_cast<double>(multiplication_ratio);
            mapping.at<double>(1, idx) = (j + starting_y) / static_cast<double>(multiplication_ratio);
            mapping.at<double>(2, idx) = 1.0;
        }
    }

    mapping = H * mapping;  // H is 3x3, mapping is 3 x N
    std::cout << "Mapping matrix initialized." << std::endl;
}

cv::Mat HomographyTransformer::undistort_image(const cv::Mat& image)
{
    cv::Mat map1, map2;
    cv::Size size(intrinsic.width, intrinsic.height);
    cv::Mat identity = cv::Mat::eye(3, 3, CV_32F);
    cv::initUndistortRectifyMap(intrinsic.cam_k, intrinsic.cam_dist, identity, intrinsic.cam_k, size, CV_32FC1, map1, map2);
    cv::Mat dst;
    cv::remap(image, dst, map1, map2, cv::INTER_LINEAR);
    return dst;
}

cv::Mat HomographyTransformer::transform_image(const cv::Mat& origin_plane)
{
    cv::Mat new_plane(new_plane_height, new_plane_width, origin_plane.type(), cv::Scalar::all(0));
    int height = origin_plane.rows;
    int width = origin_plane.cols;
    int out_of_bounds_count = 0;

    for (int idx = 0; idx < new_plane_width * new_plane_height; ++idx) {
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
            out_of_bounds_count++;
            new_plane.at<cv::Vec3b>(j, i) = cv::Vec3b(0, 0, 0);
        }
    }

    std::cout << "Out of bounds mappings: " << out_of_bounds_count << std::endl;
    return new_plane;
}

cv::Mat preprocess_image(const cv::Mat& inputImage, const size_t& imageWidth, const size_t& imageHeight) {
    cv::Mat imageRgb;
    cv::cvtColor(inputImage, imageRgb, cv::COLOR_RGB2BGR);
    // imageRgb = inputImage;
    
    imageRgb.convertTo(imageRgb, CV_32FC3, 1.0 / 255.0);
    cv::Vec3f mean = cv::Vec3f(0.485, 0.456, 0.406);
    cv::Vec3f stddev = cv::Vec3f(0.229, 0.224, 0.225);
    
    for (int i = 0; i < imageRgb.rows; ++i) {
        for (int j = 0; j < imageRgb.cols; ++j) {
            cv::Vec3f& pixel = imageRgb.at<cv::Vec3f>(i, j);
            pixel[0] = (pixel[0] - mean[0]) / stddev[0];
            pixel[1] = (pixel[1] - mean[1]) / stddev[1];
            pixel[2] = (pixel[2] - mean[2]) / stddev[2];
        }
    }


    cv::Mat resizedImage;
    cv::resize(imageRgb, resizedImage, cv::Size(imageWidth, imageHeight));

    cv::imshow("new", imageRgb);
    cv::waitKey(0);


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
        { trapezoid[0], trapezoid[1], trapezoid[2], trapezoid[3] } 
    };
    cv::fillPoly(trapezoid_mask, trapezoid_pts, cv::Scalar(255));

    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            if (trapezoid_mask.at<uchar>(v, u) == 0) continue;

            float d = depth_map.at<float>(v, u);
            if (d <= 0) continue; 

            cv::Mat pixel = (cv::Mat_<double>(3, 1) << u, v, 1);
            cv::Mat cam_coords = d * K_inv * pixel;

            double X_c = cam_coords.at<double>(0);
            double Y_c = cam_coords.at<double>(1);
            double Z_c = cam_coords.at<double>(2);

            if (Z_c <= 0) continue; // Ignore points behind the camera

            double X_w = X_c;
            double Z_w = Z_c;

            // Store the correspondence
            ground_points.push_back(cv::Point2f(static_cast<float>(X_w), static_cast<float>(Z_w)));
            image_points.push_back(cv::Point2f(static_cast<float>(u), static_cast<float>(v)));
        }
    }

    // cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(0));
    // for (const auto& pt : image_points) {
    //     cv::circle(canvas, pt, 2, cv::Scalar(0, 255, 0), -1);
    // }
    // cv::imshow("Image Points", canvas);
    // cv::waitKey(0);


    std::cout << "Selected " << image_points.size() << " correspondences inside the trapezoid." << std::endl;
}
