#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>

class GStreamerNode : public rclcpp::Node {
public:
    GStreamerNode() : Node("gstreamer_node") {
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("camera/image_raw", 10);

        gst_init(nullptr, nullptr);

        const char* interface = std::getenv("GSTREAMER_INTERFACE");
        std::string interface_str = interface ? std::string(interface) : "wlp38s0";

        std::string pipeline_str =
            "udpsrc address=230.1.1.2 port=1721 multicast-iface=" + interface_str + " "
            "! application/x-rtp, media=video, encoding-name=H264"
            "! rtph264depay"
            "! h264parse"
            "! avdec_h264"
            "! videoconvert"
            "! video/x-raw, format=BGR"
            "! appsink name=appsink emit-signals=true max-buffers=1 drop=true";

        GError* error = nullptr;
        pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
        if (!pipeline_ || error) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create GStreamer pipeline: %s", error->message);
            g_clear_error(&error);
            return;
        }

        appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "appsink");
        gst_app_sink_set_emit_signals((GstAppSink*)appsink_, true);
        gst_app_sink_set_max_buffers((GstAppSink*)appsink_, 1);
        gst_app_sink_set_drop((GstAppSink*)appsink_, true);
        gst_element_set_state(pipeline_, GST_STATE_PLAYING);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33),
            std::bind(&GStreamerNode::timer_callback, this)
        );
    }

    ~GStreamerNode() {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(appsink_);
        gst_object_unref(pipeline_);
    }

private:
    void timer_callback() {
        GstSample* sample = gst_app_sink_try_pull_sample((GstAppSink*)appsink_, 0);
        if (!sample) return;

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstCaps* caps = gst_sample_get_caps(sample);
        GstStructure* s = gst_caps_get_structure(caps, 0);

        int width, height;
        gst_structure_get_int(s, "width", &width);
        gst_structure_get_int(s, "height", &height);

        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_READ);

        auto msg = sensor_msgs::msg::Image();
        msg.header.stamp = this->now();
        msg.header.frame_id = "base_link";
        msg.height = height;
        msg.width = width;
        msg.encoding = "bgr8";
        msg.is_bigendian = false;
        msg.step = width * 3;
        msg.data = std::vector<uint8_t>(map.data, map.data + map.size);

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);

        publisher_->publish(msg);
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    GstElement* pipeline_;
    GstElement* appsink_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GStreamerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}