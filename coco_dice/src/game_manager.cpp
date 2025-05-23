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
        correct_pose_duration_ = 0.5;
        
        challenge_timer_ = this->create_wall_timer(
            500ms, std::bind(&CocoGameManager::check_challenge_timeout, this));
        
        RCLCPP_INFO(this->get_logger(), "Game started");
    }

private:
    void handle_audio_status(const std_msgs::msg::Bool::SharedPtr msg)
    {
        audio_playing_ = msg->data;
        
        if (!audio_playing_ && !detection_ongoing_) start_detection();
    }
    
    void start_detection()
    {
        detection_ongoing_ = true;
        waiting_for_pose_ = true;
        challenge_timeout_ = get_current_time() + 20.0;
        RCLCPP_INFO(this->get_logger(), "Detection phase started");
    }
    
    void check_challenge_timeout()
    {
        if (!waiting_for_pose_ || challenge_timeout_ == 0.0) return;
        
        if (get_current_time() > challenge_timeout_)
        {
            RCLCPP_INFO(this->get_logger(), "Challenge timed out");
            int random_index = rand() % defeat_texts_.size();
            std::string defeat_text = defeat_texts_[random_index];
            handle_failed_challenge(defeat_text);
        }
    }
    
    void handle_pose_result(const coco_interfaces::msg::PoseResult::SharedPtr msg)
    {        
        if (!waiting_for_pose_ || audio_playing_) return;
                
        current_challenge_ = msg->challenge;
        bool detected_poses = msg->detected_poses;
        
        if (detected_poses)
        {
            RCLCPP_INFO(this->get_logger(), "Pose result received");

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
        int random_index = rand() % victory_texts_.size();
        std::string victory_text = victory_texts_[random_index];
        feedback_msg->data = victory_text + std::to_string(score_) + ".";
        feedback_publisher_->publish(std::move(feedback_msg));
        
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
    
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr next_challenge_publisher_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr feedback_publisher_;
    
    rclcpp::Subscription<coco_interfaces::msg::PoseResult>::SharedPtr pose_result_subscription_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr audio_status_subscription_;
    
    rclcpp::TimerBase::SharedPtr challenge_timer_;

    std::vector<std::string> victory_texts_ = {
        " ¡Muy bien! Has completado el desafío. Tu puntuación es "
        " Increíble, has superado el desafío. Tu puntaje actual es ",
        " ¡Fantástico! Has logrado el desafío. Tu puntuación es ",
        " Que bien lo hiciste! Has completado el desafío. Tu puntuación es ",
        " Eres el mejor jugador del mundo en este juego, sigue asi! Tu puntuación es ",
        " ¡Impresionante! Has superado el desafío. Tu puntuación es ",
    };

    std::vector<std::string> defeat_texts_ = {
        " ¡Oh no! Has fallado el desafío, no te preocupes, puedes intentarlo de nuevo.",
        " Desafortunadamente, no has logrado el desafío, se que a la próxima lo harás mejor.",
        " No te preocupes, puedes intentarlo de nuevo, nadie te quitara tu puesto de campeon."
        " No te desanimes, sigue practicando, la practica hace al maestro.",
        " ¡No te rindas! Puedes hacerlo mejor, todos fallamos alguna vez.",
    };
    
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