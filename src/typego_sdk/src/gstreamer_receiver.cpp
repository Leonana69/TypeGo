#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <mutex>

class GStreamerNode : public rclcpp::Node {
public:
    GStreamerNode() : Node("gstreamer_node") {
        // Use QoS settings optimized for low latency
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1))
                   .best_effort()
                   .durability_volatile();
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("camera/image_raw", qos);

        gst_init(nullptr, nullptr);

        const char* interface = std::getenv("GSTREAMER_INTERFACE");
        std::string interface_str = interface ? std::string(interface) : "wlp38s0";

        // Optimized pipeline with latency reduction settings
        std::string pipeline_str =
            "udpsrc address=230.1.1.2 port=1721 multicast-iface=" + interface_str + " "
            "! application/x-rtp, media=video, encoding-name=H264"
            "! rtph264depay"
            "! h264parse"
            "! avdec_h264"  // Reduce decoder threads for lower latency
            "! videoconvert"  // Limit conversion threads
            "! video/x-raw, format=BGR"
            "! appsink name=appsink emit-signals=true max-buffers=1 drop=true sync=false";  // Disable sync for lower latency

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
        
        // Configure appsink with proper synchronization
        GstAppSinkCallbacks callbacks;
        callbacks.eos = nullptr;
        callbacks.new_preroll = nullptr;
        callbacks.new_sample = &GStreamerNode::new_sample_callback;
        gst_app_sink_set_callbacks((GstAppSink*)appsink_, &callbacks, this, nullptr);
        
        // Set pipeline to PLAYING state
        gst_element_set_state(pipeline_, GST_STATE_PLAYING);

        // Use a dedicated thread for processing samples
        processing_thread_ = std::thread(&GStreamerNode::processing_loop, this);
    }

    ~GStreamerNode() {
        running_ = false;
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(appsink_);
        gst_object_unref(pipeline_);
    }

private:
    static GstFlowReturn new_sample_callback(GstAppSink* sink, gpointer user_data) {
        GStreamerNode* self = static_cast<GStreamerNode*>(user_data);
        GstSample* sample = gst_app_sink_pull_sample(sink);
        
        if (sample) {
            std::lock_guard<std::mutex> lock(self->sample_mutex_);
            if (self->current_sample_) {
                gst_sample_unref(self->current_sample_);
            }
            self->current_sample_ = sample;
            self->sample_available_ = true;
        }
        
        return GST_FLOW_OK;
    }

    void processing_loop() {
        while (rclcpp::ok() && running_) {
            GstSample* sample_to_process = nullptr;
            
            {
                std::lock_guard<std::mutex> lock(sample_mutex_);
                if (sample_available_) {
                    sample_to_process = current_sample_;
                    current_sample_ = nullptr;
                    sample_available_ = false;
                }
            }
            
            if (sample_to_process) {
                process_sample(sample_to_process);
                gst_sample_unref(sample_to_process);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    void process_sample(GstSample* sample) {
        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstCaps* caps = gst_sample_get_caps(sample);
        GstStructure* s = gst_caps_get_structure(caps, 0);

        int width, height;
        gst_structure_get_int(s, "width", &width);
        gst_structure_get_int(s, "height", &height);

        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            return;
        }

        auto msg = std::make_unique<sensor_msgs::msg::Image>();
        msg->header.stamp = this->now();
        msg->header.frame_id = "base_link";
        msg->height = height;
        msg->width = width;
        msg->encoding = "bgr8";
        msg->is_bigendian = false;
        msg->step = width * 3;
        msg->data.assign(map.data, map.data + map.size);

        gst_buffer_unmap(buffer, &map);
        publisher_->publish(std::move(msg));
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    GstElement* pipeline_ = nullptr;
    GstElement* appsink_ = nullptr;
    
    std::thread processing_thread_;
    std::mutex sample_mutex_;
    GstSample* current_sample_ = nullptr;
    bool sample_available_ = false;
    bool running_ = true;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GStreamerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}