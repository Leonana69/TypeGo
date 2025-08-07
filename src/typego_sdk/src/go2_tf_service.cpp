#include "rclcpp/rclcpp.hpp"
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

class Go2TFService : public rclcpp::Node {
public:
    Go2TFService() : Node("go2_tf_service") {
        printf("[Go2 TF service] Initializing...\n");
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        socket_ = socket(AF_INET, SOCK_DGRAM, 0);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(8889);
        addr.sin_addr.s_addr = INADDR_ANY;
        bind(socket_, (sockaddr*)&addr, sizeof(addr));
        struct timeval tv;
        tv.tv_sec = 1;  // 1 second timeout
        tv.tv_usec = 0;
        setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

        // Send a dummy packet to notify server of this client's address
        sockaddr_in server_addr{};
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(8889);  // server port must match

        const char* go2_ip = std::getenv("GO2_IP");
        std::string go2_ip_ = go2_ip ? std::string(go2_ip) : "192.168.0.253";

        inet_pton(AF_INET, go2_ip_.c_str(), &server_addr.sin_addr);  // use server's actual IP

        uint8_t init_packet[1] = {0};
        sendto(socket_, init_packet, sizeof(init_packet), 0, (sockaddr*)&server_addr, sizeof(server_addr));

        thread_ = std::thread([this]() { this->receive_loop(); });
    }

    ~Go2TFService() {
        keep_running_ = false;
        close(socket_);
        if (thread_.joinable()) thread_.join();
    }

private:
    void receive_loop() {
        std::vector<uint8_t> buffer(2048);
        while (keep_running_ && rclcpp::ok()) {
            ssize_t rlen = recvfrom(socket_, buffer.data(), buffer.size(), 0, nullptr, nullptr);
            if (rlen < static_cast<ssize_t>(7 * sizeof(float))) continue;

            const float* data = reinterpret_cast<float*>(buffer.data());

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

    std::atomic<bool> keep_running_{true};
    int socket_;
    std::thread thread_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Go2TFService>());
    rclcpp::shutdown();
    return 0;
}