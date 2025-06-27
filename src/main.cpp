#include <Servo.h>
#include <Arduino.h>

Servo thumbServo, indexServo, middleServo, ringServo, pinkyServo;
const int NUM_SERVOS = 5;
Servo servos[NUM_SERVOS];
String inputString = "";
boolean stringComplete = false;
int previousAngles[NUM_SERVOS] = {0, 0, 0, 0, 0}; 

const int MIN_ANGLE = 0;
const int MAX_ANGLE = 180;
const float SMOOTHING_FACTOR = 0.2; 

void setup() {
  Serial.begin(9600);
  inputString.reserve(200);

  servos[0] = thumbServo;
  servos[1] = indexServo;
  servos[2] = middleServo;
  servos[3] = ringServo;
  servos[4] = pinkyServo;
  
  int servoPins[] = {7, 6, 5, 4, 3};
  for(int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(servoPins[i]);
    servos[i].write(MIN_ANGLE);  
  }
}

void loop() {
  if (stringComplete) {
    int index = 0;
    int pos = 0;
    
    while (index < NUM_SERVOS) {
      int comma = inputString.indexOf(',', pos);
      int newAngle;
      
      if (comma == -1) {
        newAngle = inputString.substring(pos).toInt();
      } else {
        newAngle = inputString.substring(pos, comma).toInt();
        pos = comma + 1;
      }
      
    
      int smoothedAngle = (newAngle * (1 - SMOOTHING_FACTOR)) + 
                         (previousAngles[index] * SMOOTHING_FACTOR);
      
      servos[index].write(smoothedAngle);
      previousAngles[index] = smoothedAngle;
      
      if (comma == -1) break;
      index++;
    }
    
    inputString = "";
    stringComplete = false;
  }
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}
