#include <iostream>
#include <filesystem>
#include "BEVTransformer.hpp"

// #define IMSHOW


int main(){
    std::string input_image_path = "/home/hesam/Desktop/datasets/kitti-odom/sequences/00/image_0/000025.png";  // Replace with your image path
    std::string output_folder = "out/";
    std::string output_image_path = output_folder + "BEV_output.png";

    // Parameters for BEV
    int output_width = 3000;
    int output_height = 2000;

    int multiplication_ratio = 200;

    // Load the ONNX model for depth estimation
    std::string model_path = "/home/hesam/Desktop/playground/depth_node/model/unidepth/unidepthv2_vits14_simp.onnx";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "depth_to_bev");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // Camera intrinsic parameters
    cv::Mat cam_k = (cv::Mat_<double>(3, 3) << 721.5377, 0, 609.5593,
                                               0, 721.5377, 172.854,
                                               0, 0, 1);
    cv::Mat cam_dist = cv::Mat::zeros(1, 5, CV_64F);  // Assuming zero distortion

    CameraIntrinsic intrinsic(1920, 1080, cam_k, cam_dist);

    // Ensure the output folder exists
    std::filesystem::create_directories(output_folder);

    // Load the image
    cv::Mat original_image = cv::imread(input_image_path);

    if (original_image.empty()) {
        std::cout << "Error: Cannot read input image " << input_image_path << std::endl;
        return -1;
    }

    int imageWidth_ = 644;
    int imageHeight_ = 364;

    cv::Mat inputImage = preprocess_image(original_image, imageWidth_, imageHeight_);

    const char* inputNames[] = {"image"};
    const char* outputNames[] = {"depth"};
    std::vector<int64_t> inputDims = {1, 3, imageHeight_, imageWidth_};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<float> inputData((float*)inputImage.datastart, (float*)inputImage.dataend);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputData.data(), inputData.size(), inputDims.data(), inputDims.size());

    std::array<Ort::Value*, 1> inputTensors = {&inputTensor};


    auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames, 
                                        *inputTensors.data(), 
                                        inputTensors.size(), outputNames, 1);

    float* outputData = outputTensors.front().GetTensorMutableData<float>();

    cv::Mat depthMat(imageHeight_, imageWidth_, CV_32FC1, outputData);

    cv::Mat depth_map_resized;
    cv::resize(depthMat, depth_map_resized, cv::Size(original_image.cols, original_image.rows));

    // Initialize transformer
    HomographyTransformer homography_transformer(
        output_width, output_height * 2, intrinsic, depth_map_resized, multiplication_ratio, -1500, 1000
    );

    // Select correspondences and compute homography
    std::vector<cv::Point2f> image_points, ground_points;
    select_correspondences(depth_map_resized, intrinsic, image_points, ground_points);

    if (image_points.size() < 4) {
        std::cout << "Not enough correspondences to compute homography." << std::endl;
        return -1;
    }

    homography_transformer.compute_homography(image_points, ground_points);

    if (homography_transformer.H.empty()) {
        return -1;
    }

    // Undistort image if necessary
    cv::Mat undistorted_image = homography_transformer.undistort_image(original_image);
    if (undistorted_image.empty()) {
        std::cout << "Error: undistortion failed." << std::endl;
        return -1; 
    }

    // Perform homography transformation
    double begin = static_cast<double>(cv::getTickCount());
    cv::Mat full_bev_image = homography_transformer.transform_image(undistorted_image);
    double end = static_cast<double>(cv::getTickCount());
    double elapsed_secs = (end - begin) / cv::getTickFrequency();
    std::cout << "Elapsed seconds for transformation: " << elapsed_secs << std::endl;

    // Save the transformed image
    #ifndef IMSHOW
        cv::imwrite(output_image_path, full_bev_image);
        std::cout << "Saved transformed image to " << output_image_path << std::endl;
    // #else
    //     cv::imshow("BEV", full_bev_image);
    //     cv::waitKey(0);
    #endif

    // Optionally, save depth map for visualization
    cv::Mat depth_map_visual;
    cv::normalize(depth_map_resized, depth_map_visual, 0, 255, cv::NORM_MINMAX);
    depth_map_visual.convertTo(depth_map_visual, CV_8U);

    #ifndef IMSHOW
        cv::imwrite(output_folder + "Depth_output.png", depth_map_visual);
        std::cout << "Saved depth map visualization to Depth_output.png" << std::endl;
    #else
        cv::imshow("depth", depth_map_visual);
        cv::waitKey(0);
    #endif
}