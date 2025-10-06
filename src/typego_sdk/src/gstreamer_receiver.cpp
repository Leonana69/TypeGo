#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <atomic>
#include <thread>
#include <cstring>

class GStreamerStream {
public:
    GStreamerStream(rclcpp::Node* node,
                    const std::string& topic_name,
                    const std::string& encoding,
                    const std::string& iface,
                    int port,
                    bool is_depth = false)
        : node_(node), encoding_(encoding), is_depth_(is_depth)
    {
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        publisher_ = node_->create_publisher<sensor_msgs::msg::Image>(topic_name, qos);

        // For depth, we receive BGR (containing packed depth), for color we receive BGR
        std::string format = (encoding == "bgr8" || is_depth) ? "BGR" : "GRAY8";
        
        std::string pipeline_str =
            "udpsrc address=230.1.1.1 port=" + std::to_string(port) +
            " multicast-iface=" + iface + " "
            "! application/x-rtp, media=video, encoding-name=H264 "
            "! rtph264depay "
            "! h264parse "
            "! avdec_h264 "
            "! videoconvert "
            "! video/x-raw,format=" + format + " "
            "! appsink name=appsink sync=false max-buffers=1 drop=true";

        GError* error = nullptr;
        pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
        if (!pipeline_ || error) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to create GStreamer pipeline on port %d: %s",
                         port, error ? error->message : "unknown error");
            if (error) g_clear_error(&error);
            return;
        }

        appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "appsink");
        gst_element_set_state(pipeline_, GST_STATE_PLAYING);

        thread_ = std::thread([this]() { run(); });

        RCLCPP_INFO(node_->get_logger(), "Started GStreamer stream on port %d", port);
    }

    ~GStreamerStream() {
        running_ = false;
        if (thread_.joinable()) thread_.join();
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        if (appsink_) gst_object_unref(appsink_);
        if (pipeline_) gst_object_unref(pipeline_);
    }

private:
    void run() {
        while (rclcpp::ok() && running_) {
            GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink_), 10 * GST_MSECOND);
            if (!sample) continue;

            GstBuffer* buffer = gst_sample_get_buffer(sample);
            GstCaps* caps = gst_sample_get_caps(sample);
            GstStructure* s = gst_caps_get_structure(caps, 0);

            int width = 0, height = 0;
            gst_structure_get_int(s, "width", &width);
            gst_structure_get_int(s, "height", &height);

            GstMapInfo map;
            if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
                gst_sample_unref(sample);
                continue;
            }

            sensor_msgs::msg::Image msg;
            msg.header.stamp = node_->get_clock()->now();
            msg.header.frame_id = is_depth_ ? "depth_frame" : "color_frame";
            msg.height = height;
            msg.width = width;

            if (is_depth_) {
                // 8-bit grayscale
                msg.encoding = "8UC1";
                msg.is_bigendian = false;
                msg.step = width;

                // Allocate space for 8-bit depth data
                msg.data.resize(width * height);
                uint8_t* depth_ptr = reinterpret_cast<uint8_t*>(msg.data.data());

                // Convert BGR to grayscale (take any channel since they're all the same)
                // Then scale back to millimeters: grayscale_value = depth_in_mm / 25
                for (int i = 0; i < width * height; ++i) {
                    uint8_t gray_value = map.data[i * 3];  // Take B channel (all BGR channels are same)
                    depth_ptr[i] = static_cast<uint8_t>(gray_value);  // Convert back to mm
                }
            } else {
                // Regular color image
                msg.encoding = encoding_;
                msg.is_bigendian = false;
                msg.step = width * 3;
                msg.data.assign(map.data, map.data + map.size);
            }

            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);

            publisher_->publish(msg);
        }
    }

    rclcpp::Node* node_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    GstElement* pipeline_{nullptr};
    GstElement* appsink_{nullptr};
    std::thread thread_;
    std::atomic<bool> running_{true};
    std::string encoding_;
    bool is_depth_;
};

class DualGStreamerNode : public rclcpp::Node {
public:
    DualGStreamerNode() : Node("dual_gstreamer_node") {
        gst_init(nullptr, nullptr);
        const char* iface = std::getenv("GSTREAMER_INTERFACE");
        std::string interface_str = iface ? iface : "wlp38s0";

        color_stream_ = std::make_unique<GStreamerStream>(this,
                        "/camera/color/image_raw", "bgr8", interface_str, 1722, false);
        depth_stream_ = std::make_unique<GStreamerStream>(this,
                        "/camera/depth/image_raw", "16UC1", interface_str, 1723, true);

        RCLCPP_INFO(this->get_logger(), "Dual GStreamer node initialized.");
    }

private:
    std::unique_ptr<GStreamerStream> color_stream_;
    std::unique_ptr<GStreamerStream> depth_stream_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DualGStreamerNode>());
    rclcpp::shutdown();
    return 0;
}