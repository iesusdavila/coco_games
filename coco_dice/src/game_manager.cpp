#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/bool.hpp"
#include "coco_interfaces/msg/pose_result.hpp"
#include <chrono>
#include <memory>
#include <thread>

using namespace std::chrono_literals;

class CocoGameManager : public rclcpp::Node
{
public:
    CocoGameManager() : Node("coco_game_manager")
    {
        next_challenge_publisher_ = this->create_publisher<std_msgs::msg::Bool>(
            "/next_challenge", 10);
        feedback_publisher_ = this->create_publisher<std_msgs::msg::String>(
            "/game_feedback", 10);
        
        pose_result_subscription_ = this->create_subscription<coco_interfaces::msg::PoseResult>(
            "/pose_result", 10,
            std::bind(&CocoGameManager::handle_pose_result, this, std::placeholders::_1));
        
        audio_status_subscription_ = this->create_subscription<std_msgs::msg::Bool>(
            "/audio_playing", 10,
            std::bind(&CocoGameManager::handle_audio_status, this, std::placeholders::_1));
        
        current_challenge_ = 0;
        score_ = 0;
        audio_playing_ = false;
        detection_ongoing_ = false;
        challenge_timeout_ = 0.0;
        waiting_for_pose_ = false;
        correct_pose_start_time_ = 0.0;
        correct_pose_duration_ = 1.0;
        
        challenge_timer_ = this->create_wall_timer(
            500ms, std::bind(&CocoGameManager::check_challenge_timeout, this));
        
        RCLCPP_INFO(this->get_logger(), "Coco Game Manager started successfully");
        RCLCPP_INFO(this->get_logger(), "Game started");
    }

private:
    void handle_audio_status(const std_msgs::msg::Bool::SharedPtr msg)
    {
        audio_playing_ = msg->data;
        
        if (!audio_playing_ && !detection_ongoing_)
        {
            start_detection();
        }
    }
    
    void start_detection()
    {
        detection_ongoing_ = true;
        waiting_for_pose_ = true;
        challenge_timeout_ = get_current_time() + 10.0;
        RCLCPP_INFO(this->get_logger(), "Detection phase started");
    }
    
    void check_challenge_timeout()
    {
        if (!waiting_for_pose_ || challenge_timeout_ == 0.0)
        {
            return;
        }
        
        if (get_current_time() > challenge_timeout_)
        {
            RCLCPP_INFO(this->get_logger(), "Challenge timed out");
            handle_failed_challenge("¡Tiempo agotado! Vamos a intentar con otro desafío.");
        }
    }
    
    void handle_pose_result(const coco_interfaces::msg::PoseResult::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "LEYENDO DATA DE POSE: %s", 
                    msg->detected_poses ? "true" : "false");
        
        if (!waiting_for_pose_ || audio_playing_)
        {
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "Pose result received");
        
        current_challenge_ = msg->challenge;
        bool detected_poses = msg->detected_poses;
        
        if (detected_poses)
        {
            if (correct_pose_start_time_ == 0.0)
            {
                correct_pose_start_time_ = get_current_time();
                RCLCPP_INFO(this->get_logger(), "Correct pose detected, starting timer");
            }
            else if (get_current_time() - correct_pose_start_time_ >= correct_pose_duration_)
            {
                handle_successful_challenge();
            }
        }
        else
        {
            correct_pose_start_time_ = 0.0;
        }
    }
    
    void handle_successful_challenge()
    {
        waiting_for_pose_ = false;
        detection_ongoing_ = false;
        correct_pose_start_time_ = 0.0;
        score_++;
        
        RCLCPP_INFO(this->get_logger(), 
                    "Challenge completed successfully! Score: %d", score_);
        
        auto feedback_msg = std::make_unique<std_msgs::msg::String>();
        feedback_msg->data = "¡Muy bien! Has completado el desafío. Tu puntuación es " + 
                           std::to_string(score_) + ".";
        feedback_publisher_->publish(std::move(feedback_msg));
        
        // Sleep for 2 seconds
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        auto next_challenge_msg = std::make_unique<std_msgs::msg::Bool>();
        next_challenge_msg->data = true;
        next_challenge_publisher_->publish(std::move(next_challenge_msg));
    }
    
    void handle_failed_challenge(const std::string& feedback_text)
    {
        waiting_for_pose_ = false;
        detection_ongoing_ = false;
        correct_pose_start_time_ = 0.0;
        
        RCLCPP_INFO(this->get_logger(), "Challenge failed: %s", feedback_text.c_str());
        
        auto feedback_msg = std::make_unique<std_msgs::msg::String>();
        feedback_msg->data = feedback_text;
        feedback_publisher_->publish(std::move(feedback_msg));
        
        // Sleep for 2 seconds
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        auto next_challenge_msg = std::make_unique<std_msgs::msg::Bool>();
        next_challenge_msg->data = true;
        next_challenge_publisher_->publish(std::move(next_challenge_msg));
    }
    
    double get_current_time()
    {
        return static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()) / 1000.0;
    }
    
    // Publishers
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr next_challenge_publisher_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr feedback_publisher_;
    
    // Subscriptions
    rclcpp::Subscription<coco_interfaces::msg::PoseResult>::SharedPtr pose_result_subscription_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr audio_status_subscription_;
    
    // Timers
    rclcpp::TimerBase::SharedPtr challenge_timer_;
    
    // Game state variables
    int current_challenge_;
    int score_;
    bool audio_playing_;
    bool detection_ongoing_;
    double challenge_timeout_;
    bool waiting_for_pose_;
    double correct_pose_start_time_;
    double correct_pose_duration_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CocoGameManager>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}