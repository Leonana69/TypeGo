#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <fstream>
#include <cmath>
#include <queue>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "typego_interface/msg/way_point.hpp"
#include "typego_interface/msg/way_point_array.hpp"

static const char* labels[] = {
    "hallway",
    "room",
    "free space"
};

struct Waypoint {
    int id;
    double x;
    double y;
    std::string label;
};

// Helper function for HTTP requests
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
    size_t total_size = size * nmemb;
    output->append((char*)contents, total_size);
    return total_size;
}

int grid_distance(const nav_msgs::msg::OccupancyGrid &grid, int sx, int sy, int gx, int gy) {
    if (sx == gx && sy == gy) return 0;  // Same cell.

    const int width = grid.info.width;
    const int height = grid.info.height;
    const auto &data = grid.data;

    auto index = [&](int x, int y) { return y * width + x; };

    // Check bounds.
    if (sx < 0 || sx >= width || sy < 0 || sy >= height) return -1;  // Start invalid.
    if (gx < 0 || gx >= width || gy < 0 || gy >= height) return -1;  // Goal invalid.

    // Check if start or goal is blocked (assuming >50 is occupied).
    if (data[index(sx, sy)] > 50 || data[index(gx, gy)] > 50) return -1;

    // BFS setup.
    std::queue<std::pair<int, int>> q;
    std::vector<std::vector<int>> dist(height, std::vector<int>(width, -1));
    q.push({sx, sy});
    dist[sy][sx] = 0;

    constexpr int dx[] = {1, -1, 0, 0};
    constexpr int dy[] = {0, 0, 1, -1};

    while (!q.empty()) {
        auto [x, y] = q.front(); q.pop();
        for (int i = 0; i < 4; ++i) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;  // Out of bounds.
            if (data[index(nx, ny)] > 50 || dist[ny][nx] != -1) continue;   // Blocked or visited.
            dist[ny][nx] = dist[y][x] + 1;
            if (nx == gx && ny == gy) return dist[ny][nx];  // Found goal.
            q.push({nx, ny});
        }
    }
    return -1;  // Goal unreachable.
}

class AdaptiveWaypointNode : public rclcpp::Node {
public:
    AdaptiveWaypointNode()
        : Node("adaptive_waypoint_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", 10, std::bind(&AdaptiveWaypointNode::map_callback, this, std::placeholders::_1));
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10, std::bind(&AdaptiveWaypointNode::image_callback, this, std::placeholders::_1));
        timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&AdaptiveWaypointNode::on_timer, this));
        waypoint_pub_ = this->create_publisher<typego_interface::msg::WayPointArray>("/waypoints", 10);
        waypoint_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/waypoint_markers", 10);
        load_waypoints(waypoint_file_);

        const char* edge_service_ip = std::getenv("EDGE_SERVICE_IP");
        edge_service_ip_ = edge_service_ip ? std::string(edge_service_ip) : "localhost";
        if (edge_service_ip_.empty()) {
            RCLCPP_WARN(this->get_logger(), "EDGE_SERVICE_IP not set, using default: localhost");
        } else {
            RCLCPP_INFO(this->get_logger(), "Using EDGE_SERVICE_IP: %s", edge_service_ip_.c_str());
        }
    }

private:
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<typego_interface::msg::WayPointArray>::SharedPtr waypoint_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr waypoint_marker_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    nav_msgs::msg::OccupancyGrid::SharedPtr latest_map_;
    sensor_msgs::msg::Image::SharedPtr latest_image_;
    std::vector<Waypoint> waypoints_;
    std::string waypoint_file_ = getWaypointFilePath();
    std::string edge_service_ip_;
    const double threshold_distance_meters_ = 3.0;
    std::mutex image_mutex_;

    int next_id_ = 0;

    std::string getWaypointFilePath() {
        const char* resource_dir = std::getenv("RESOURCE_DIR");

        std::string resource_path;
        if (resource_dir == nullptr) {
            // Use default path if RESOURCE_DIR is not set
            resource_path = "/home/guojun/Documents/Go2-Livox-ROS2/src/typego_sdk/resource";
            std::cout << "[INFO] RESOURCE_DIR not set. Using default: " << resource_path << std::endl;
        } else {
            resource_path = resource_dir;
        }

        return resource_path + "/waypoints.csv";
    }

    void load_waypoints(const std::string &file) {
        std::ifstream f(file);
        int id;
        double x, y;
        std::string label;
        while (f >> id >> x >> y) {
            std::getline(f, label);  // Read the rest of the line as label
            // Trim leading whitespace from label
            label.erase(0, label.find_first_not_of(" \t"));
            waypoints_.push_back({id, x, y, label});
        }
        if (!waypoints_.empty()) {
            next_id_ = waypoints_.back().id + 1;  // Increment ID for the next waypoint
        }
    }

    void save_waypoints(const std::string &file) {
        std::ofstream f(file);
        for (auto &wp : waypoints_) {
            f << wp.id << " " << wp.x << " " << wp.y << " " << wp.label << "\n";
        }
    }

    void map_callback(nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        latest_map_ = msg;
    }

    void image_callback(sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(image_mutex_);
        latest_image_ = msg;
    }

    void on_timer() {
        if (!latest_map_) return;

        geometry_msgs::msg::TransformStamped tf;
        try {
            tf = tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);
        } catch (const tf2::TransformException &e) {
            RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", e.what());
            return;
        }

        double x = tf.transform.translation.x;
        double y = tf.transform.translation.y;

        auto grid = *latest_map_;

        int robot_cx = (x - grid.info.origin.position.x) / grid.info.resolution;
        int robot_cy = (y - grid.info.origin.position.y) / grid.info.resolution;

        double min_dist = std::numeric_limits<double>::max();
        bool valid_waypoint_found = true;
        for (auto &wp : waypoints_) {
            // skip if the absolute distance is greater than threshold
            if ((x - wp.x) * (x - wp.x) + (y - wp.y) * (y - wp.y) > threshold_distance_meters_ * threshold_distance_meters_) {
                continue;
            }

            int wp_cx = (wp.x - grid.info.origin.position.x) / grid.info.resolution;
            int wp_cy = (wp.y - grid.info.origin.position.y) / grid.info.resolution;

            int dist = grid_distance(grid, robot_cx, robot_cy, wp_cx, wp_cy);
            if (dist >= 0) {
                double meters = dist * grid.info.resolution;
                min_dist = std::min(min_dist, meters);
            } else {
                valid_waypoint_found = false;  // No valid path to this waypoint
            }
        }

        if (min_dist > threshold_distance_meters_ && valid_waypoint_found) {
            RCLCPP_INFO(this->get_logger(), "Adding new waypoint at (%.2f, %.2f)", x, y);

            // First process the latest image if available
            int label_index = -1;
            {
                std::lock_guard<std::mutex> lock(image_mutex_);
                if (latest_image_) {
                    try {
                        // Check for bgr8 encoding
                        if (latest_image_->encoding != "bgr8") {
                            RCLCPP_WARN(this->get_logger(), "Unsupported image encoding: %s", latest_image_->encoding.c_str());
                            return;
                        }

                        // Create cv::Mat without cv_bridge
                        cv::Mat image(
                            latest_image_->height,
                            latest_image_->width,
                            CV_8UC3, // for "bgr8"
                            const_cast<uint8_t*>(latest_image_->data.data()),
                            latest_image_->step
                        );

                        // Clone the image to decouple from ROS buffer
                        cv::Mat resized_image;
                        cv::resize(image, resized_image, cv::Size(640, 360));

                        label_index = send_image_for_detection(resized_image);
                    } catch (const std::exception& e) {
                        RCLCPP_ERROR(this->get_logger(), "Image conversion exception: %s", e.what());
                    }
                }
            }
            waypoints_.push_back({next_id_, x, y, labels[label_index]});
            next_id_++;
            save_waypoints(waypoint_file_);
        }

        publish_waypoints();
    }

    int send_image_for_detection(const cv::Mat& image) {
        // Convert image to webp format
        std::vector<uchar> buffer;
        cv::imencode(".webp", image, buffer);
        
        // Prepare JSON data
        nlohmann::json json_data = {
            {"robot_info", "go2"},
            {"service_type", "clip"},
            {"image_id", 1},
            {"queries", {
                "a photo taken in a hallway",
                "a photo taken inside a room",
                "an open empty space without walls or furniture"
            }}
        };
        
        // Prepare the HTTP request
        CURL *curl = curl_easy_init();
        if (curl) {
            curl_mime *mime = curl_mime_init(curl);

            // Add image part
            curl_mimepart *part = curl_mime_addpart(mime);
            curl_mime_name(part, "image");
            curl_mime_filename(part, "image.webp");
            curl_mime_data(part, reinterpret_cast<const char*>(buffer.data()), buffer.size());
            curl_mime_type(part, "image/webp");

            // Add JSON part
            std::string json_str = json_data.dump();
            curl_mimepart *json_part = curl_mime_addpart(mime);
            curl_mime_name(json_part, "json_data");
            curl_mime_data(json_part, json_str.c_str(), json_str.size());

            // Set the URL
            std::string url = "http://" + edge_service_ip_ + ":50049/process";
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);

            // Response handling
            std::string response_string;
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

            // Perform the request
            CURLcode res = curl_easy_perform(curl);
            if (res != CURLE_OK) {
                RCLCPP_ERROR(this->get_logger(), "curl_easy_perform() failed: %s", curl_easy_strerror(res));
            } else {
                try {
                    auto response_json = nlohmann::json::parse(response_string);
                    auto results = response_json["result"].get<std::vector<float>>();
                    // Find the index of the highest score
                    auto max_it = std::max_element(results.begin(), results.end());
                    size_t best_index = std::distance(results.begin(), max_it);
                    RCLCPP_INFO(this->get_logger(), "CLIP Response: %s", response_string.c_str());
                    curl_mime_free(mime);
                    curl_easy_cleanup(curl);
                    return static_cast<int>(best_index);
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to parse JSON response: %s", e.what());
                }
            }

            // Cleanup
            curl_mime_free(mime);
            curl_easy_cleanup(curl);
        }

        return -1;  // Error
    }

    void publish_waypoints() {
        typego_interface::msg::WayPointArray array;
        visualization_msgs::msg::MarkerArray marker_array;
        for (size_t i = 0; i < waypoints_.size(); ++i) {
            auto &wp = waypoints_[i];
            visualization_msgs::msg::Marker m;
            m.header.frame_id = "map";
            m.header.stamp = this->now();
            m.ns = "waypoints";
            m.id = i;
            m.type = m.SPHERE;
            m.action = m.ADD;
            m.pose.position.x = wp.x;
            m.pose.position.y = wp.y;
            m.pose.position.z = 0.0;
            m.scale.x = m.scale.y = m.scale.z = 0.2;
            m.color.r = 0.0f;
            m.color.g = 1.0f;
            m.color.b = 0.0f;
            m.color.a = 1.0f;
            marker_array.markers.push_back(m);

            // Add text marker for the label
            visualization_msgs::msg::Marker text_marker;
            text_marker.header.frame_id = "map";
            text_marker.header.stamp = this->now();
            text_marker.ns = "waypoint_labels";
            text_marker.id = i;
            text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::msg::Marker::ADD;
            text_marker.pose.position.x = wp.x;
            text_marker.pose.position.y = wp.y;
            text_marker.pose.position.z = 0.5;  // Above the waypoint
            text_marker.scale.z = 0.3;  // Text height
            text_marker.color.r = 1.0f;
            text_marker.color.g = 1.0f;
            text_marker.color.b = 1.0f;
            text_marker.color.a = 1.0f;
            text_marker.text = wp.label.empty() ? "No label" : wp.label;
            marker_array.markers.push_back(text_marker);
            typego_interface::msg::WayPoint wp_msg;
            wp_msg.id = wp.id;
            wp_msg.x = wp.x;
            wp_msg.y = wp.y;
            wp_msg.label = wp.label;
            array.waypoints.push_back(wp_msg);
        }
        waypoint_pub_->publish(array);
        waypoint_marker_pub_->publish(marker_array);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<AdaptiveWaypointNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}