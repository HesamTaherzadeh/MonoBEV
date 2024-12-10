#include <iostream>

// Enable or disable debug output
// #define BEV_DEBUG_OUTPUT

#ifdef BEV_DEBUG_OUTPUT
    #define DEBUG_LOG(msg) std::cout << msg << std::endl
#else
    #define DEBUG_LOG(msg)
#endif

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include "BEVTransformer.hpp"
#include "bev_interface/msg/homography.hpp" 
#include "rclcpp/qos.hpp"

class BEVNode : public rclcpp::Node
{
public:
    BEVNode() : Node("bev_node"),
                input_node_names_({"image"}), 
                output_node_names_({"depth"}),
                intrinsic_image_width_(0),
                intrinsic_image_height_(0),
                depth_model_height_(0),
                depth_model_width_(0),
                output_width_(0),
                output_height_(0),
                multiplication_ratio_(0),
                starting_x_(0),
                starting_y_(0)
    {
        DEBUG_LOG("Initializing BEVNode...");

        // Load parameters
        this->declare_parameter<std::string>("model_path", "");
        this->declare_parameter<int>("output_width", 3000);
        this->declare_parameter<int>("output_height", 2000);
        this->declare_parameter<int>("multiplication_ratio", 250);
        this->declare_parameter<int>("starting_x", 0);
        this->declare_parameter<int>("starting_y", 0);
        this->declare_parameter<int>("depth_model_height", 364);
        this->declare_parameter<int>("depth_model_width", 644);
        this->declare_parameter<int>("intrinsic_image_width", 1242);
        this->declare_parameter<int>("intrinsic_image_height", 375);
        this->declare_parameter<std::vector<double>>("camera_intrinsics.K", std::vector<double>(9));
        this->declare_parameter<std::vector<double>>("camera_intrinsics.dist", std::vector<double>(5));

        homography_matrix = cv::Mat::zeros(3, 3, CV_64F);

        // Load parameters into member variables
        load_parameters();

        // Create publishers and subscribers
        rclcpp::QoS qos_best_effort = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));

        image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("BEV", 10);
        depth_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("depth", 10);
        rgb_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("rgb", 10);

        camera_info_publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_params", qos_best_effort);

        image_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/kitti/camera_color_left/image_raw", 0,
            std::bind(&BEVNode::image_callback, this, std::placeholders::_1));
        
        homography_publisher_ = this->create_publisher<bev_interface::msg::Homography>("homography", 10);


        // Initialize model and camera parameters
        load_camera_parameters();
        load_model();
    }

private:
    void load_parameters()
    {
    DEBUG_LOG("Loading parameters...");

    model_path_ = this->get_parameter("model_path").as_string();
    DEBUG_LOG("Model path: " << model_path_);

    output_width_ = this->get_parameter("output_width").as_int();
    DEBUG_LOG("Output width: " << output_width_);

    output_height_ = this->get_parameter("output_height").as_int();
    DEBUG_LOG("Output height: " << output_height_);

    multiplication_ratio_ = this->get_parameter("multiplication_ratio").as_int();
    DEBUG_LOG("Multiplication ratio: " << multiplication_ratio_);

    starting_x_ = this->get_parameter("starting_x").as_int();
    DEBUG_LOG("Starting X: " << starting_x_);

    starting_y_ = this->get_parameter("starting_y").as_int();
    DEBUG_LOG("Starting Y: " << starting_y_);

    depth_model_height_ = this->get_parameter("depth_model_height").as_int();
    DEBUG_LOG("Depth model height: " << depth_model_height_);

    depth_model_width_ = this->get_parameter("depth_model_width").as_int();
    DEBUG_LOG("Depth model width: " << depth_model_width_);

    intrinsic_image_width_ = this->get_parameter("intrinsic_image_width").as_int();
    DEBUG_LOG("Intrinsic image width: " << intrinsic_image_width_);

    intrinsic_image_height_ = this->get_parameter("intrinsic_image_height").as_int();
    DEBUG_LOG("Intrinsic image height: " << intrinsic_image_height_);
    }


    void load_camera_parameters()
    {
        DEBUG_LOG("Loading camera parameters...");

        auto K = this->get_parameter("camera_intrinsics.K").as_double_array();
        auto dist = this->get_parameter("camera_intrinsics.dist").as_double_array();

        cam_k_ = (cv::Mat_<double>(3, 3) << K[0], K[1], K[2], K[3], K[4], K[5], K[6], K[7], K[8]);
        cam_dist_ = (cv::Mat_<double>(1, 5) << dist[0], dist[1], dist[2], dist[3], dist[4]);
        intrinsic_ = CameraIntrinsic(intrinsic_image_width_, intrinsic_image_height_, cam_k_, cam_dist_);

        DEBUG_LOG("Camera parameters loaded successfully.");
    }

    void publish_camera_info(const std_msgs::msg::Header& header){
        auto camera_info_msg = sensor_msgs::msg::CameraInfo();
        camera_info_msg.header = header;
        camera_info_msg.width = intrinsic_image_width_;
        camera_info_msg.height = intrinsic_image_height_;
        camera_info_msg.distortion_model = "plumb_bob";
        camera_info_msg.k = {cam_k_.at<double>(0), cam_k_.at<double>(1), cam_k_.at<double>(2), cam_k_.at<double>(3), cam_k_.at<double>(4), cam_k_.at<double>(5), cam_k_.at<double>(6), cam_k_.at<double>(7), cam_k_.at<double>(8)};
        camera_info_msg.p = {cam_k_.at<double>(0), cam_k_.at<double>(1), cam_k_.at<double>(2), 
                     0.0, 
                     cam_k_.at<double>(3), cam_k_.at<double>(4), cam_k_.at<double>(5), 
                     0.0, 
                     cam_k_.at<double>(6), cam_k_.at<double>(7), cam_k_.at<double>(8), 
                     0.0};
        camera_info_publisher_->publish(camera_info_msg);
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        publish_camera_info(msg->header);
        DEBUG_LOG("Received image callback.");

        try
        {
            rgb_publisher_->publish(*msg);

            // Convert the ROS image message to OpenCV image
            cv::Mat original_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;

            if (original_image.empty())
            {
                RCLCPP_ERROR(this->get_logger(), "Error: Received an empty image.");
                return;
            }

            // Check if the dimensions of the input image match the expected intrinsic dimensions
            if (original_image.cols != intrinsic_image_width_ || original_image.rows != intrinsic_image_height_)
            {
                RCLCPP_WARN(this->get_logger(), 
                            "Warning: Input image size (%dx%d) does not match intrinsic size (%dx%d). Resizing...",
                            original_image.cols, original_image.rows, intrinsic_image_width_, intrinsic_image_height_);
            }

            cv::Mat bev_image = process_image_to_bev(original_image, msg->header);


            auto homography_msg = bev_interface::msg::Homography();
            homography_msg.header = msg->header;

           for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    homography_msg.matrix[i * 3 + j] = homography_matrix.at<double>(i, j);
                }
            }

        homography_publisher_->publish(homography_msg);
            

            if (bev_image.empty())
            {
                RCLCPP_ERROR(this->get_logger(), "Error: BEV image is empty after processing.");
                return;
            }

            // Publish the BEV image
            publish_bev_image(bev_image);
        }
        catch (const cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge error: %s", e.what());
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in image_callback: %s", e.what());
        }
    }


    void load_model()
    {
        DEBUG_LOG("Loading model...");
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "depth_to_bev");
        Ort::SessionOptions session_options;
        session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), session_options);
        DEBUG_LOG("Model loaded from path: " << model_path_);
    }

    cv::Mat run_model_inference(const cv::Mat& image)
    {

        auto preprocessed_image = preprocess_image(image, depth_model_width_, depth_model_height_);
        DEBUG_LOG("Started Depth Inference");

        std::vector<int64_t> input_node_dims = {1, 3, depth_model_height_, depth_model_width_};

        std::vector<float> input_data((float*)preprocessed_image.datastart, (float*)preprocessed_image.dataend);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, input_data.data(), input_data.size(),
            input_node_dims.data(), input_node_dims.size());

        const char* input_names[] = {"image"};
        const char* output_names[] = {"depth"};

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        float* output_data = output_tensors.front().GetTensorMutableData<float>();

        cv::Mat depth_map(depth_model_height_, depth_model_width_, CV_32F, output_data);

        double min_depth, max_depth;
        cv::minMaxLoc(depth_map, &min_depth, &max_depth);
        std::cout << "Depth map range: [" << min_depth << ", " << max_depth << "]" << std::endl;

        DEBUG_LOG("Ended Depth Inference");

        return depth_map.clone();
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

        std::vector<cv::Mat> chwChannels(3);
        cv::split(resizedImage, chwChannels);
        cv::Mat chwImage;
        cv::vconcat(chwChannels, chwImage);
        
        return chwImage;
    }

    cv::Mat process_image_to_bev(const cv::Mat& input_image, const std_msgs::msg::Header&  header)
    {
        DEBUG_LOG("Started BEV process");

        cv::Mat depth_map = run_model_inference(input_image);

        if (depth_map.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Error: Depth map is empty after inference.");
            return cv::Mat();
        }
        else{
            publish_depth_image(depth_map, header);
        }

        if (cv::countNonZero(depth_map) == 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Error: Depth map contains only zeros.");
            return cv::Mat();
        }

        if (depth_map.rows != depth_model_height_ || depth_map.cols != depth_model_width_)
        {
            RCLCPP_ERROR(this->get_logger(), 
                        "Error: Depth map dimensions (%dx%d) do not match expected size (%dx%d).",
                        depth_map.rows, depth_map.cols, depth_model_height_, depth_model_width_);
            return cv::Mat();
        }

        DEBUG_LOG("Depth map inference completed.");

        HomographyTransformer transformer(output_width_, output_height_, intrinsic_, depth_map,
                                        multiplication_ratio_, starting_x_, starting_y_);

        std::vector<cv::Point2f> image_points, ground_points;
        select_correspondences(depth_map, intrinsic_, image_points, ground_points);

        if (image_points.size() < 4 || ground_points.size() < 4)
        {
            RCLCPP_ERROR(this->get_logger(), 
                        "Error: Insufficient correspondences for homography computation. "
                        "Image points: %zu, Ground points: %zu.", 
                        image_points.size(), ground_points.size());
            return cv::Mat();
        }

        try
        {
            transformer.compute_homography(image_points, ground_points);
        }


        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error computing homography: %s", e.what());
            return cv::Mat();
        }

        // cv::Mat undistorted_image = transformer.undistort_image(input_image);

        // cv::Mat bev_image = transformer.transform_image(undistorted_image);

        DEBUG_LOG("BEV process completed.");
        return cv::Mat();
    }

    void publish_depth_image(const cv::Mat& depth_map, const std_msgs::msg::Header& header)
        {
            auto depth_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_32FC1, depth_map).toImageMsg();
            depth_publisher_->publish(*depth_msg);
            RCLCPP_INFO(this->get_logger(), "Published depth image.");
        }

    void publish_bev_image(const cv::Mat& bev_image)
        {
            auto bev_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", bev_image).toImageMsg();
            image_publisher_->publish(*bev_msg);
            RCLCPP_INFO(this->get_logger(), "Published BEV image");
        }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;
    rclcpp::Publisher<bev_interface::msg::Homography>::SharedPtr homography_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_publisher_;

    std::string model_path_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
    Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    cv::Mat cam_k_;
    cv::Mat cam_dist_;
    CameraIntrinsic intrinsic_;
    bool should_publish_bev_ = true;
    cv::Mat homography_matrix;

    int intrinsic_image_width_;
    int intrinsic_image_height_;
    int depth_model_height_;
    int depth_model_width_;
    int output_width_;
    int output_height_;
    int multiplication_ratio_;
    int starting_x_;
    int starting_y_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    DEBUG_LOG("Starting BEVNode...");
    auto node = std::make_shared<BEVNode>();
    rclcpp::spin(node);
    DEBUG_LOG("Shutting down BEVNode...");
    rclcpp::shutdown();
    return 0;
}
