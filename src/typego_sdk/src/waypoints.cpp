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
#include <ament_index_cpp/get_package_share_directory.hpp>

struct Waypoint {
    double x;
    double y;
};

int grid_distance(const nav_msgs::msg::OccupancyGrid &grid, int sx, int sy, int gx, int gy) {
    if (sx == gx && sy == gy) return 0;

    const int width = grid.info.width;
    const int height = grid.info.height;
    const auto &data = grid.data;

    auto index = [&](int x, int y) { return y * width + x; };
    if (sx < 0 || sy < 0 || gx < 0 || gy < 0 || sx >= width || sy >= height || gx >= width || gy >= height)
        return -1;
    if (data[index(sx, sy)] > 50 || data[index(gx, gy)] > 50)
        return -1; // blocked

    std::queue<std::pair<int, int>> q;
    std::vector<std::vector<int>> dist(height, std::vector<int>(width, -1));
    q.push({sx, sy});
    dist[sy][sx] = 0;

    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};

    while (!q.empty()) {
        auto [x, y] = q.front(); q.pop();
        for (int i = 0; i < 4; ++i) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
            if (data[index(nx, ny)] > 50 || dist[ny][nx] != -1) continue;
            dist[ny][nx] = dist[y][x] + 1;
            if (nx == gx && ny == gy) return dist[ny][nx];
            q.push({nx, ny});
        }
    }
    return -1;
}

class AdaptiveWaypointNode : public rclcpp::Node {
public:
    AdaptiveWaypointNode()
        : Node("adaptive_waypoint_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", 10, std::bind(&AdaptiveWaypointNode::map_callback, this, std::placeholders::_1));
        timer_ = this->create_wall_timer(std::chrono::seconds(1), std::bind(&AdaptiveWaypointNode::on_timer, this));
        waypoint_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/waypoints", 10);
        std::string path = ament_index_cpp::get_package_share_directory("typego_sdk");
        std::string waypoint_file = path + "/resource/waypoints.csv";
        load_waypoints(waypoint_file);
    }

private:
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr waypoint_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    nav_msgs::msg::OccupancyGrid::SharedPtr latest_map_;
    std::vector<Waypoint> waypoints_;
    const double threshold_distance_meters_ = 3.0;

    void load_waypoints(const std::string &file) {
        std::ifstream f(file);
        double x, y;
        while (f >> x >> y) waypoints_.push_back({x, y});
    }

    void save_waypoints(const std::string &file) {
        std::ofstream f(file);
        for (auto &wp : waypoints_) f << wp.x << " " << wp.y << "\n";
    }

    void map_callback(nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        latest_map_ = msg;
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
        for (auto &wp : waypoints_) {
            int wp_cx = (wp.x - grid.info.origin.position.x) / grid.info.resolution;
            int wp_cy = (wp.y - grid.info.origin.position.y) / grid.info.resolution;
            int dist = grid_distance(grid, robot_cx, robot_cy, wp_cx, wp_cy);
            if (dist >= 0) {
                double meters = dist * grid.info.resolution;
                min_dist = std::min(min_dist, meters);
            }
        }

        if (min_dist > threshold_distance_meters_) {
            RCLCPP_INFO(this->get_logger(), "Adding new waypoint at (%.2f, %.2f)", x, y);
            waypoints_.push_back({x, y});
            save_waypoints("waypoints.csv");
        }

        publish_waypoints();
    }

    void publish_waypoints() {
        visualization_msgs::msg::MarkerArray array;
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
            array.markers.push_back(m);
        }
        waypoint_pub_->publish(array);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<AdaptiveWaypointNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}