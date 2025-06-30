#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <cmath>
#include <arpa/inet.h>

#define MSG_POINTCLOUD 0x01
#define MSG_HIGHSTATE  0x02

class LivoxReceiver : public rclcpp::Node {
public:
    LivoxReceiver() : Node("livox_udp_receiver") {
        printf("[LivoxReceiver] Initializing...\n");
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("livox_points", 10);
        laserscan_publisher_ = this->create_publisher<sensor_msgs::msg::LaserScan>("scan", 10);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        socket_ = socket(AF_INET, SOCK_DGRAM, 0);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(8888);
        addr.sin_addr.s_addr = INADDR_ANY;
        bind(socket_, (sockaddr*)&addr, sizeof(addr));
        struct timeval tv;
        tv.tv_sec = 1;  // 1 second timeout
        tv.tv_usec = 0;
        setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

        // Send a dummy packet to notify server of this client's address
        sockaddr_in server_addr{};
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(8888);  // server port must match

        const char* go2_ip = std::getenv("GO2_IP");
        std::string go2_ip_ = go2_ip ? std::string(go2_ip) : "192.168.0.253";

        inet_pton(AF_INET, go2_ip_.c_str(), &server_addr.sin_addr);  // use server's actual IP

        uint8_t init_packet[1] = {0};
        sendto(socket_, init_packet, sizeof(init_packet), 0, (sockaddr*)&server_addr, sizeof(server_addr));

        thread_ = std::thread([this]() { this->receive_loop(); });

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&LivoxReceiver::publish_aggregated_cloud, this)
        );
    }

    ~LivoxReceiver() {
        keep_running_ = false;
        close(socket_);
        if (thread_.joinable()) thread_.join();
    }

private:
    void publish_aggregated_cloud() {
        std::vector<float> local_buffer;
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            if (point_buffer_.empty()) return;
            point_buffer_.swap(local_buffer);
        }

        size_t num_points = local_buffer.size() / 3;
        auto timestamp = now();

        sensor_msgs::msg::PointCloud2 cloud_msg;
        cloud_msg.header.stamp = timestamp;
        cloud_msg.header.frame_id = "base_link";
        cloud_msg.height = 1;
        cloud_msg.width = num_points;
        cloud_msg.is_dense = true;
        cloud_msg.is_bigendian = false;
        cloud_msg.point_step = 12;
        cloud_msg.row_step = cloud_msg.point_step * num_points;

        sensor_msgs::msg::PointField field;
        field.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field.count = 1;
        field.name = "x"; field.offset = 0; cloud_msg.fields.push_back(field);
        field.name = "y"; field.offset = 4; cloud_msg.fields.push_back(field);
        field.name = "z"; field.offset = 8; cloud_msg.fields.push_back(field);

        cloud_msg.data.resize(local_buffer.size() * sizeof(float));
        memcpy(cloud_msg.data.data(), local_buffer.data(), cloud_msg.data.size());
        publisher_->publish(cloud_msg);

        sensor_msgs::msg::LaserScan scan_msg;
        scan_msg.header.stamp = timestamp;
        scan_msg.header.frame_id = "base_link";

        float angle_min = -M_PI;
        float angle_max = M_PI;
        float angle_increment = M_PI / 180.0f;
        int num_beams = static_cast<int>((angle_max - angle_min) / angle_increment);

        scan_msg.angle_min = angle_min;
        scan_msg.angle_max = angle_max;
        scan_msg.angle_increment = angle_increment;
        scan_msg.scan_time = 0.1f;
        scan_msg.time_increment = 0.0f;
        scan_msg.range_min = 0.1f;
        scan_msg.range_max = 20.0f;
        scan_msg.ranges.assign(num_beams, std::numeric_limits<float>::infinity());

        for (size_t i = 0; i < num_points; ++i) {
            float x = local_buffer[3 * i];
            float y = local_buffer[3 * i + 1];
            float z = local_buffer[3 * i + 2];

            if (std::abs(z) > 0.1f) continue;
            float angle = std::atan2(y, x);
            float range = std::hypot(x, y);
            int index = static_cast<int>((angle - angle_min) / angle_increment);
            if (index >= 0 && index < num_beams && range < scan_msg.ranges[index])
                scan_msg.ranges[index] = range;
        }

        laserscan_publisher_->publish(scan_msg);
    }

    void receive_loop() {
        std::vector<uint8_t> buffer(2048);
        while (keep_running_ && rclcpp::ok()) {
            ssize_t rlen = recvfrom(socket_, buffer.data(), buffer.size(), 0, nullptr, nullptr);
            if (rlen <= 0) continue;
            unsigned long recv_len = static_cast<unsigned long>(rlen);

            uint8_t msg_type = buffer[0];
            if (msg_type == MSG_POINTCLOUD) {
                constexpr float scale = 1.0f / 1000.0f;
                const int16_t* data = reinterpret_cast<const int16_t*>(buffer.data() + 3);
                size_t num_pts = (recv_len - 3) / sizeof(int16_t);
                if (num_pts % 3 != 0) continue;

                std::lock_guard<std::mutex> lock(buffer_mutex_);
                for (size_t i = 0; i < num_pts; ++i)
                    point_buffer_.emplace_back(data[i] * scale);
            }
            else if (msg_type == MSG_HIGHSTATE && recv_len >= 1 + sizeof(float) * 7) {
                const float* data = reinterpret_cast<float*>(buffer.data() + 1);

                geometry_msgs::msg::TransformStamped tf_msg;
                tf_msg.header.stamp = now();
                tf_msg.header.frame_id = "odom";
                tf_msg.child_frame_id = "base_link";

                tf_msg.transform.translation.x = data[0];
                tf_msg.transform.translation.y = data[1];
                tf_msg.transform.translation.z = data[2];
                tf_msg.transform.rotation.x = data[4];
                tf_msg.transform.rotation.y = data[5];
                tf_msg.transform.rotation.z = data[6];
                tf_msg.transform.rotation.w = data[3];

                tf_broadcaster_->sendTransform(tf_msg);
            }
        }
    }

    std::atomic<bool> keep_running_{true};
    std::vector<float> point_buffer_;
    std::mutex buffer_mutex_;
    int socket_;
    std::thread thread_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laserscan_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LivoxReceiver>());
    rclcpp::shutdown();
    return 0;
}