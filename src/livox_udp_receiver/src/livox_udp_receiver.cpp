#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "sensor_msgs/msg/laser_scan.hpp"
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#define MSG_POINTCLOUD 0x01
#define MSG_HIGHSTATE  0x02

class LivoxReceiver : public rclcpp::Node {
public:
    LivoxReceiver() : Node("livox_udp_receiver") {
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("livox_points", 10);
        laserscan_publisher_ = this->create_publisher<sensor_msgs::msg::LaserScan>("scan", 10);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        socket_ = socket(AF_INET, SOCK_DGRAM, 0);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(8888);  // Match sender port
        addr.sin_addr.s_addr = INADDR_ANY;
        if (bind(socket_, (sockaddr*)&addr, sizeof(addr)) < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to bind socket");
            return;
        }

        last_pub_time_ = now();
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(66),
            std::bind(&LivoxReceiver::publish_aggregated_cloud, this)
        );

        thread_ = std::thread([this]() { this->receive_loop(); });
    }

    ~LivoxReceiver() {
        rclcpp::shutdown(); // Optional if you need to stop ROS first
        keep_running_ = false;  // Signal the thread to stop
        close(socket_);
        if (thread_.joinable()) {
            thread_.join();  // Wait for thread to finish
        }
    }

private:
    std::atomic<bool> keep_running_{true};
    void publish_aggregated_cloud() {
        std::vector<float> points;
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            if (point_buffer_.empty()) return;
            points.swap(point_buffer_);  // Move data out of buffer
        }
    
        size_t num_points = points.size() / 3;
        rclcpp::Time timestamp = now();

        sensor_msgs::msg::PointCloud2 cloud_msg;
        cloud_msg.header.stamp = timestamp;
        cloud_msg.header.frame_id = "base_link";
        cloud_msg.height = 1;
        cloud_msg.width = num_points;
        cloud_msg.is_dense = true;
        cloud_msg.is_bigendian = false;
        cloud_msg.point_step = 12;
        cloud_msg.row_step = cloud_msg.point_step * num_points;
    
        cloud_msg.fields.clear();
        sensor_msgs::msg::PointField field;
        field.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field.count = 1;
    
        field.name = "x"; field.offset = 0; cloud_msg.fields.push_back(field);
        field.name = "y"; field.offset = 4; cloud_msg.fields.push_back(field);
        field.name = "z"; field.offset = 8; cloud_msg.fields.push_back(field);
    
        cloud_msg.data.resize(points.size() * sizeof(float));
        memcpy(cloud_msg.data.data(), points.data(), cloud_msg.data.size());
    
        publisher_->publish(cloud_msg);

        // === Create LaserScan ===
        sensor_msgs::msg::LaserScan scan_msg;
        scan_msg.header.stamp = timestamp;
        scan_msg.header.frame_id = "base_link";

        float angle_min = -M_PI;
        float angle_max = M_PI;
        float angle_increment = M_PI / 180.0; // 1 degree
        int num_beams = std::round((angle_max - angle_min) / angle_increment);

        scan_msg.angle_min = angle_min;
        scan_msg.angle_max = angle_max;
        scan_msg.angle_increment = angle_increment;
        scan_msg.time_increment = 0.0;
        scan_msg.scan_time = 0.1; // assuming 10Hz
        scan_msg.range_min = 0.1;
        scan_msg.range_max = 20.0;

        scan_msg.ranges.assign(num_beams, std::numeric_limits<float>::infinity());

        for (size_t i = 0; i < num_points; ++i) {
            float x = points[3 * i];
            float y = points[3 * i + 1];
            float z = points[3 * i + 2];

            if (std::abs(z) > 0.1f) continue;  // Keep near-ground points

            float angle = std::atan2(y, x);
            float range = std::sqrt(x * x + y * y);

            int index = static_cast<int>((angle - angle_min) / angle_increment);
            if (index >= 0 && index < num_beams) {
                // Use closest point for each beam
                if (range < scan_msg.ranges[index])
                    scan_msg.ranges[index] = range;
            }
        }

        laserscan_publisher_->publish(scan_msg);
    }
    void receive_loop() {
        std::vector<uint8_t> buffer(65536);  // Large enough for max UDP payload
        while (keep_running_ && rclcpp::ok()) {
            ssize_t recv_len = recvfrom(socket_, buffer.data(), buffer.size(), 0, nullptr, nullptr);
            if (recv_len <= 0) {
                printf("Error receiving data\n");
                continue;
            }
    
            uint8_t msg_type = buffer[0];
    
            if (msg_type == MSG_POINTCLOUD) {
                const float* float_data = reinterpret_cast<const float*>(buffer.data() + 1);
                size_t num_floats = (recv_len - 1) / sizeof(float);
                if (num_floats % 3 != 0) return;

                std::lock_guard<std::mutex> lock(buffer_mutex_);
                point_buffer_.insert(point_buffer_.end(), float_data, float_data + num_floats);
            }
            else if (msg_type == MSG_HIGHSTATE) {
                if (recv_len < 1 + (int)sizeof(float) * 7) {
                    printf("Received MSG_HIGHSTATE but size is too small\n");
                    continue;
                }
                const float* data = reinterpret_cast<float*>(buffer.data() + 1);
    
                geometry_msgs::msg::TransformStamped tf_msg;
                tf_msg.header.stamp = now();
                tf_msg.header.frame_id = "odom";           // or "map", depending on your setup
                tf_msg.child_frame_id = "base_link";
    
                tf_msg.transform.translation.x = data[0];
                tf_msg.transform.translation.y = data[1];
                tf_msg.transform.translation.z = data[2];
    
                tf_msg.transform.rotation.x = data[4]; // x
                tf_msg.transform.rotation.y = data[5]; // y
                tf_msg.transform.rotation.z = data[6]; // z
                tf_msg.transform.rotation.w = data[3]; // w
    
                tf_broadcaster_->sendTransform(tf_msg);
            }
        }
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laserscan_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<float> point_buffer_;
    std::mutex buffer_mutex_;
    rclcpp::Time last_pub_time_;
    int socket_;
    std::thread thread_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LivoxReceiver>());
    rclcpp::shutdown();
    return 0;
}