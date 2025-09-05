#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <atomic>

class GStreamerNode : public rclcpp::Node {
public:
    GStreamerNode() : Node("gstreamer_node") {
        // QoS optimized for low latency
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1))
                    .best_effort()
                    .durability_volatile();

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("camera/image_raw", qos);

        gst_init(nullptr, nullptr);

        const char* interface = std::getenv("GSTREAMER_INTERFACE");
        std::string interface_str = interface ? std::string(interface) : "wlp38s0";

        // Simplified pipeline - closer to your working gst-launch
        std::string pipeline_str =
            "udpsrc address=230.1.1.2 port=1721 multicast-iface=" + interface_str + " "
            "! application/x-rtp, media=video, encoding-name=H264 "
            "! rtph264depay "
            "! h264parse "
            "! avdec_h264 "
            "! videoconvert "
            "! video/x-raw, format=BGR "
            "! appsink name=appsink sync=false max-buffers=1 drop=true";

        GError* error = nullptr;
        pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
        if (!pipeline_ || error) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create GStreamer pipeline: %s", error->message);
            g_clear_error(&error);
            return;
        }

        appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "appsink");
        
        // Set pipeline to PLAYING state first
        gst_element_set_state(pipeline_, GST_STATE_PLAYING);

        // Create a timer to poll for samples instead of callbacks
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),  // 100Hz polling
            std::bind(&GStreamerNode::timer_callback, this)
        );

        RCLCPP_INFO(this->get_logger(), "GStreamer node started");
    }

    ~GStreamerNode() {
        if (timer_) {
            timer_->cancel();
        }
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(appsink_);
        gst_object_unref(pipeline_);
    }

private:
    void timer_callback() {
        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink_), 0);
        if (sample) {
            process_sample(sample);
            gst_sample_unref(sample);
            frame_count_++;
            
            // Log frame rate every 100 frames
            static auto last_time = std::chrono::steady_clock::now();
            if (frame_count_ % 100 == 0) {
                auto now = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time);
                double fps = 100000.0 / duration.count();
                RCLCPP_INFO(this->get_logger(), "Processing at %.1f FPS", fps);
                last_time = now;
            }
        }
    }

    void process_sample(GstSample* sample) {
        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstCaps* caps = gst_sample_get_caps(sample);
        GstStructure* s = gst_caps_get_structure(caps, 0);

        int width, height;
        if (!gst_structure_get_int(s, "width", &width) ||
            !gst_structure_get_int(s, "height", &height)) {
            return;
        }

        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            return;
        }

        // Pre-allocate message
        msg_.header.stamp = this->now();
        msg_.header.frame_id = "base_link";
        msg_.height = height;
        msg_.width = width;
        msg_.encoding = "bgr8";
        msg_.is_bigendian = false;
        msg_.step = width * 3;

        // Direct copy without intermediate allocation
        msg_.data.resize(map.size);

        std::memcpy(msg_.data.data(), map.data, map.size);

        gst_buffer_unmap(buffer, &map);
        
        auto t7 = std::chrono::steady_clock::now();
        // Publish directly
        publisher_->publish(msg_);
        auto t_end = std::chrono::steady_clock::now();
        auto dur8 = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t7).count();
        RCLCPP_INFO(this->get_logger(), "Publish took %ld ms", dur8 / 1000);
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    GstElement* pipeline_ = nullptr;
    GstElement* appsink_ = nullptr;
    std::atomic<uint64_t> frame_count_{0};
    sensor_msgs::msg::Image msg_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GStreamerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}