#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int16.hpp"
#include "coco_interfaces/msg/pose_result.hpp"
#include "coco_interfaces/msg/pose_landmarks.hpp"
#include <chrono>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <cmath>

using namespace std::chrono_literals;

class CocoPoseDetector : public rclcpp::Node
{
public:
    CocoPoseDetector() : Node("coco_pose_detector")
    {
        pose_result_publisher_ = this->create_publisher<coco_interfaces::msg::PoseResult>(
            "/pose_result", 10);
        
        current_challenge_subscription_ = this->create_subscription<std_msgs::msg::Int16>(
            "/current_challenge", 10,
            std::bind(&CocoPoseDetector::handle_new_challenge, this, std::placeholders::_1));
        
        pose_landmarks_subscription_ = this->create_subscription<coco_interfaces::msg::PoseLandmarks>(
            "/pose_landmarks", 10,
            std::bind(&CocoPoseDetector::detect_poses, this, std::placeholders::_1));
        
        current_challenge_ = 0;
        game_active_ = false;
        
        // Set up body landmark indices
        NOSE = 0;
        LEFT_EYE = 1;
        RIGHT_EYE = 2;
        LEFT_EAR = 3;
        RIGHT_EAR = 4;
        LEFT_SHOULDER = 5;
        RIGHT_SHOULDER = 6;
        LEFT_ELBOW = 7;
        RIGHT_ELBOW = 8;
        LEFT_WRIST = 9;
        RIGHT_WRIST = 10;
        LEFT_HIP = 11;
        RIGHT_HIP = 12;
        
        // Initialize gesture dictionary
        initialize_gesture_dict();
        
        RCLCPP_INFO(this->get_logger(), "Coco Pose Detector started successfully");
    }

private:
    void initialize_gesture_dict()
    {
        DICT_GESTURES[1] = "Brazo derecho arriba";
        DICT_GESTURES[2] = "Brazo izquierdo arriba";
        DICT_GESTURES[3] = "Ambos brazos arriba";
        DICT_GESTURES[4] = "Ambos brazos abajo";
        DICT_GESTURES[5] = "Brazo derecho hacia delante";
        DICT_GESTURES[6] = "Brazo izquierdo hacia delante";
        DICT_GESTURES[7] = "Ambos brazos hacia delante";
        DICT_GESTURES[8] = "Símbolo X con los brazos";
        DICT_GESTURES[9] = "Tocando nariz con muneca derecha";
        DICT_GESTURES[10] = "Tocando nariz con muneca izquierda";
        DICT_GESTURES[11] = "Tocando ojo izquierdo con muneca izquierda";
        DICT_GESTURES[12] = "Tocando ojo izquierdo con muneca derecha";
        DICT_GESTURES[13] = "Tocando ojo derecho con muneca derecha";
        DICT_GESTURES[14] = "Tocando ojo derecho con muneca izquierda";
        DICT_GESTURES[15] = "Tocando oreja derecha con mano derecha";
        DICT_GESTURES[16] = "Tocando oreja derecha con mano izquierda";
        DICT_GESTURES[17] = "Tocando oreja derecha con ambas manos";
        DICT_GESTURES[18] = "Tocando oreja izquierda con mano izquierda";
        DICT_GESTURES[19] = "Tocando oreja izquierda con mano derecha";
        DICT_GESTURES[20] = "Tocando oreja izquierda con ambas manos";
        DICT_GESTURES[21] = "Tocando hombro izquierdo con mano izquierda";
        DICT_GESTURES[22] = "Tocando hombro izquierdo con mano derecha";
        DICT_GESTURES[23] = "Tocando hombro derecho con mano derecha";
        DICT_GESTURES[24] = "Tocando hombro derecho con mano izquierda";
        DICT_GESTURES[25] = "Tocando codo izquierdo con mano derecha";
        DICT_GESTURES[26] = "Tocando codo derecho con mano izquierda";
    }
    
    void handle_new_challenge(const std_msgs::msg::Int16::SharedPtr msg)
    {
        current_challenge_ = msg->data;
        RCLCPP_INFO(this->get_logger(), "New challenge received: %d", current_challenge_);
    }
    
    bool is_above(const std::pair<float, float>& point1, const std::pair<float, float>& point2, float threshold = 30.0f)
    {
        return (point2.second - point1.second) > threshold;
    }
    
    bool is_below(const std::pair<float, float>& point1, const std::pair<float, float>& point2, float threshold = 15.0f)
    {
        return (0.0f < std::abs(point1.second - point2.second) && std::abs(point1.second - point2.second) <= threshold);
    }
    
    bool is_at_same_height(const std::pair<float, float>& point1, const std::pair<float, float>& point2, float threshold = 30.0f)
    {
        return (0.0f < std::abs(point1.second - point2.second) && std::abs(point1.second - point2.second) < threshold);
    }
    
    bool is_in_horizontal_range(const std::pair<float, float>& point1, const std::pair<float, float>& point2, float threshold = 30.0f)
    {
        return (0.0f < std::abs(point1.first - point2.first) && std::abs(point1.first - point2.first) < threshold);
    }
    
    bool is_right_of(const std::pair<float, float>& point1, const std::pair<float, float>& point2, float threshold = 30.0f)
    {
        return (point2.first - point1.first) > threshold;
    }
    
    bool is_left_of(const std::pair<float, float>& point1, const std::pair<float, float>& point2, float threshold = 30.0f)
    {
        return (point1.first - point2.first) > threshold;
    }
    
    bool is_near(const std::pair<float, float>& point1, const std::pair<float, float>& point2, float threshold = 50.0f)
    {
        float distance = std::sqrt(std::pow(point1.first - point2.first, 2) + std::pow(point1.second - point2.second, 2));
        return distance < threshold;
    }
    
    bool detect_pose_actions(const std::vector<std::pair<float, float>>& keypoints)
    {
        if (current_challenge_ == 0) {
            return false;
        }
        
        int thershold_bt_shoulders = (keypoints[LEFT_SHOULDER].first - keypoints[RIGHT_SHOULDER].first) / 3;
        int thershold_vertical = (keypoints[NOSE].second - keypoints[LEFT_EYE].second) * 2;
        int thershold_touch_eye = (keypoints[LEFT_EYE].first - keypoints[RIGHT_EYE].first) / 2;
        int thershold_touch_elbow = (keypoints[NOSE].second - keypoints[LEFT_EYE].second) * 4;
        
        switch (current_challenge_)
        {
            case 1:
                if (is_above(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER]) &&
                    is_above(keypoints[RIGHT_ELBOW], keypoints[RIGHT_SHOULDER]) &&
                    keypoints[RIGHT_WRIST].second != 0 && keypoints[RIGHT_ELBOW].second != 0)
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[1].c_str());
                    return true;
                }
                break;
                
            case 2:
                if (is_above(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER]) &&
                    is_above(keypoints[LEFT_ELBOW], keypoints[LEFT_SHOULDER]) &&
                    keypoints[LEFT_WRIST].second != 0 && keypoints[LEFT_ELBOW].second != 0)
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[2].c_str());
                    return true;
                }
                break;
                
            case 3:
            {
                bool arm_left_up = (is_above(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER]) &&
                                   is_above(keypoints[LEFT_ELBOW], keypoints[LEFT_SHOULDER]) &&
                                   keypoints[LEFT_WRIST].second != 0 && keypoints[LEFT_ELBOW].second != 0);
                                   
                bool arm_right_up = (is_above(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER]) &&
                                    is_above(keypoints[RIGHT_ELBOW], keypoints[RIGHT_SHOULDER]) &&
                                    keypoints[RIGHT_WRIST].second != 0 && keypoints[RIGHT_ELBOW].second != 0);
                                    
                if (arm_left_up && arm_right_up)
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[3].c_str());
                    return true;
                }
            }
            break;
            
            case 4:
                if (is_below(keypoints[RIGHT_WRIST], keypoints[RIGHT_HIP]) &&
                    is_below(keypoints[LEFT_WRIST], keypoints[LEFT_HIP]))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[4].c_str());
                    return true;
                }
                break;
                
            case 5:
                if (is_at_same_height(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER], 100.0f) &&
                    is_in_horizontal_range(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER], 100.0f))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[5].c_str());
                    return true;
                }
                break;
                
            case 6:
                if (is_at_same_height(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER], 100.0f) &&
                    is_in_horizontal_range(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER], 100.0f))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[6].c_str());
                    return true;
                }
                break;
                
            case 7:
            {
                bool arm_left_forward = (is_at_same_height(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER], 100.0f) &&
                                        is_in_horizontal_range(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER], 100.0f));
                                        
                bool arm_right_forward = (is_at_same_height(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER], 100.0f) &&
                                         is_in_horizontal_range(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER], 100.0f));
                                         
                if (arm_left_forward && arm_right_forward)
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[7].c_str());
                    return true;
                }
            }
            break;
            
            case 8:
                if (is_above(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER]) &&
                    is_above(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER]) &&
                    is_left_of(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER], thershold_bt_shoulders) &&
                    is_right_of(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER], thershold_bt_shoulders))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[8].c_str());
                    return true;
                }
                break;
                
            case 9:
                if (is_near(keypoints[RIGHT_WRIST], keypoints[NOSE], thershold_vertical))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[9].c_str());
                    return true;
                }
                break;
                
            case 10:
                if (is_near(keypoints[LEFT_WRIST], keypoints[NOSE], thershold_vertical))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[10].c_str());
                    return true;
                }
                break;
                
            case 11:
                if (is_near(keypoints[LEFT_WRIST], keypoints[LEFT_EYE], thershold_touch_eye))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[11].c_str());
                    return true;
                }
                break;
                
            case 12:
                if (is_near(keypoints[RIGHT_WRIST], keypoints[LEFT_EYE], thershold_touch_eye))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[12].c_str());
                    return true;
                }
                break;
                
            case 13:
                if (is_near(keypoints[RIGHT_WRIST], keypoints[RIGHT_EYE], thershold_touch_eye))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[13].c_str());
                    return true;
                }
                break;
                
            case 14:
                if (is_near(keypoints[LEFT_WRIST], keypoints[RIGHT_EYE], thershold_touch_eye))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[14].c_str());
                    return true;
                }
                break;
                
            case 15:
                if (is_near(keypoints[RIGHT_WRIST], keypoints[RIGHT_EAR], thershold_vertical))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[15].c_str());
                    return true;
                }
                break;
                
            case 16:
                if (is_near(keypoints[LEFT_WRIST], keypoints[RIGHT_EAR], thershold_vertical))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[16].c_str());
                    return true;
                }
                break;
                
            case 17:
            {
                bool wrist_right = is_near(keypoints[RIGHT_WRIST], keypoints[RIGHT_EAR], thershold_vertical);
                bool wrist_left = is_near(keypoints[LEFT_WRIST], keypoints[RIGHT_EAR], thershold_vertical);
                
                if (wrist_right && wrist_left)
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[17].c_str());
                    return true;
                }
            }
            break;
            
            case 18:
                if (is_near(keypoints[LEFT_WRIST], keypoints[LEFT_EAR], thershold_vertical))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[18].c_str());
                    return true;
                }
                break;
                
            case 19:
                if (is_near(keypoints[RIGHT_WRIST], keypoints[LEFT_EAR], thershold_vertical))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[19].c_str());
                    return true;
                }
                break;
                
            case 20:
            {
                bool wrist_left = is_near(keypoints[LEFT_WRIST], keypoints[LEFT_EAR], thershold_vertical);
                bool wrist_right = is_near(keypoints[RIGHT_WRIST], keypoints[LEFT_EAR], thershold_vertical);
                
                if (wrist_left && wrist_right)
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[20].c_str());
                    return true;
                }
            }
            break;
            
            case 21:
                if (is_near(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER], thershold_vertical))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[21].c_str());
                    return true;
                }
                break;
                
            case 22:
                if (is_near(keypoints[RIGHT_WRIST], keypoints[LEFT_SHOULDER], thershold_vertical))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[22].c_str());
                    return true;
                }
                break;
                
            case 23:
                if (is_near(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER], thershold_vertical))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[23].c_str());
                    return true;
                }
                break;
                
            case 24:
                if (is_near(keypoints[LEFT_WRIST], keypoints[RIGHT_SHOULDER], thershold_vertical))
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[24].c_str());
                    return true;
                }
                break;
                
            case 25:
                if (is_near(keypoints[RIGHT_WRIST], keypoints[LEFT_ELBOW], thershold_touch_elbow) &&
                    keypoints[LEFT_ELBOW].second != 0 && keypoints[RIGHT_WRIST].second != 0)
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[25].c_str());
                    return true;
                }
                break;
                
            case 26:
                if (is_near(keypoints[LEFT_WRIST], keypoints[RIGHT_ELBOW], thershold_touch_elbow) &&
                    keypoints[RIGHT_ELBOW].second != 0 && keypoints[LEFT_WRIST].second != 0)
                {
                    RCLCPP_INFO(this->get_logger(), "%s", DICT_GESTURES[26].c_str());
                    return true;
                }
                break;
                
            default:
                break;
        }
        
        return false;
    }
    
    void detect_poses(const coco_interfaces::msg::PoseLandmarks::SharedPtr msg)
    {
        if (current_challenge_ == 0) return;
        
        std::vector<std::pair<float, float>> person_keypoints;
        for (const auto& landmark : msg->landmarks) // Iterar sobre cada landmark
        {
            float x = landmark.x;  // Acceder al campo x del Point
            float y = landmark.y;  // Acceder al campo y del Point
            person_keypoints.emplace_back(x, y);
        }
        
        auto result_msg = std::make_unique<coco_interfaces::msg::PoseResult>();
        result_msg->challenge = current_challenge_;
        result_msg->detected_poses = detect_pose_actions(person_keypoints);
        result_msg->timestamp = this->now();
        RCLCPP_INFO(this->get_logger(), "Challenge: %d", current_challenge_);
        
        pose_result_publisher_->publish(std::move(result_msg));
    }
    
    // Publishers
    rclcpp::Publisher<coco_interfaces::msg::PoseResult>::SharedPtr pose_result_publisher_;
    
    // Subscriptions
    rclcpp::Subscription<std_msgs::msg::Int16>::SharedPtr current_challenge_subscription_;
    rclcpp::Subscription<coco_interfaces::msg::PoseLandmarks>::SharedPtr pose_landmarks_subscription_;
    
    // Game state variables
    int current_challenge_;
    bool game_active_;
    
    // Body landmark indices
    int NOSE;
    int LEFT_EYE;
    int RIGHT_EYE;
    int LEFT_EAR;
    int RIGHT_EAR;
    int LEFT_SHOULDER;
    int RIGHT_SHOULDER;
    int LEFT_ELBOW;
    int RIGHT_ELBOW;
    int LEFT_WRIST;
    int RIGHT_WRIST;
    int LEFT_HIP;
    int RIGHT_HIP;
    
    // Gesture dictionary
    std::map<int, std::string> DICT_GESTURES;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CocoPoseDetector>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
