#include <ros.h>
#include <std_msgs/Int16.h>
#include "CytronMotorDriver.h"


// Configure the motor driver.
CytronMD motor(PWM_DIR, 9, 8);  // PWM = Pin 3, DIR = Pin 4.

int speed_m = 9;
int dir_m = 8;

int encoder_Pin_1 = 2;
int encoder_Pin_2 = 3;

volatile int lastEncoded = 0;
volatile long encoderValue = 0;

long lastencoderValue = 0;

int lastMSB = 0;
int lastLSB = 0;

ros::NodeHandle nh;
std_msgs::Int16 msg;
ros::Publisher counter_pub("/counter", &msg);

// Callback function to handle the received data on topic "/feedback"
void feedbackCallback(const std_msgs::Int16& feedback_msg) {
  
  int feedback_data = feedback_msg.data;
 motor.setSpeed(feedback_data);


 
  msg.data = encoderValue;
  counter_pub.publish(&msg);
}

ros::Subscriber<std_msgs::Int16> feedback_sub("/feedback", &feedbackCallback);

void setup() {
  nh.initNode();
  nh.advertise(counter_pub);

  // Subscribe to the "/feedback" topic with the callback function
  nh.subscribe(feedback_sub);

  pinMode(encoder_Pin_1, INPUT);
  pinMode(encoder_Pin_2, INPUT);

  
  pinMode(speed_m, OUTPUT);
  pinMode(dir_m, OUTPUT);

  digitalWrite(encoder_Pin_1, HIGH); //turn pullup resistor on
  digitalWrite(encoder_Pin_2, HIGH); //turn pullup resistor on

  
  attachInterrupt(0, updateEncoder, CHANGE); 
  attachInterrupt(1, updateEncoder, CHANGE);
}


void loop(){ 
  
  msg.data = encoderValue;
  counter_pub.publish(&msg);
  nh.spinOnce();
  
}


void updateEncoder()
{
  int MSB = digitalRead(encoder_Pin_1); //MSB = most significant bit
  int LSB = digitalRead(encoder_Pin_2); //LSB = least significant bit

  int encoded = (MSB << 1) |LSB; //converting the 2 pin value to single number
  int sum  = (lastEncoded << 2) | encoded; //adding it to the previous encoded value

  if(sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encoderValue ++;
  if(sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) encoderValue --;

  lastEncoded = encoded; //store this value for next time
}
