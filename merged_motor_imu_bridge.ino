// merged_motor_imu_bridge.ino
// 电机控制 + IMU 透传到同一块板子
// 串口0(USB Serial):
//   1) 接收上位机电机目标:  "s0,s1,s2\n" 或 "s0 s1 s2\n"
//   2) 周期输出 IMU 欧拉角:  "IMU,roll,pitch\n"
//
// 串口1(Serial1): 接 IMU 模块，沿用 imu_test.ino 的帧解析方式
//
// 注意：这里默认 IMU 数据帧中两个 float 分别是 buffer[11:15], buffer[15:19]
//       这和你 imu_test.ino 里当前能正常工作的索引保持一致。

const int stepPins[] = {31, 35, 37};
const int dirPins[]  = {30, 34, 36};
const int ledPin = 13;

long current_steps[3] = {0, 0, 0};
long target_steps[3]  = {0, 0, 0};

const int pulseDelayUs = 1200;
String rxLine = "";

float imu_angle_a = 0.0f;
float imu_angle_b = 0.0f;
bool imu_valid = false;
unsigned long lastImuPublishMs = 0;
const unsigned long imuPublishPeriodMs = 20;   // 50 Hz

void setDirection(int idx, bool pulling) {
  bool pinState = LOW;
  if (idx == 0) {
    pinState = pulling ? HIGH : LOW;
  } else {
    pinState = pulling ? HIGH : LOW;
  }
  digitalWrite(dirPins[idx], pinState);
}

bool parseLineToTargets(const String& line) {
  char buf[96];
  line.toCharArray(buf, sizeof(buf));
  for (unsigned int i = 0; i < strlen(buf); ++i) {
    if (buf[i] == ',') buf[i] = ' ';
  }
  long a, b, c;
  int parsed = sscanf(buf, "%ld %ld %ld", &a, &b, &c);
  if (parsed == 3) {
    target_steps[0] = a;
    target_steps[1] = b;
    target_steps[2] = c;
    return true;
  }
  return false;
}

void pollMotorCommandSerial() {
  while (Serial.available() > 0) {
    char ch = (char)Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      if (rxLine.length() > 0) {
        parseLineToTargets(rxLine);
        rxLine = "";
      }
    } else {
      if (rxLine.length() < 90) rxLine += ch;
      else rxLine = "";
    }
  }
}

void stepTowardTargets() {
  bool anyMoving = false;
  for (int i = 0; i < 3; ++i) {
    if (current_steps[i] != target_steps[i]) {
      bool pulling = (target_steps[i] < current_steps[i]);
      setDirection(i, pulling);
      digitalWrite(stepPins[i], LOW);
      anyMoving = true;
    }
  }

  if (!anyMoving) {
    digitalWrite(ledPin, LOW);
    return;
  }

  digitalWrite(ledPin, HIGH);
  delayMicroseconds(pulseDelayUs);

  for (int i = 0; i < 3; ++i) {
    if (current_steps[i] != target_steps[i]) {
      digitalWrite(stepPins[i], HIGH);
      if (current_steps[i] < target_steps[i]) current_steps[i]++;
      else current_steps[i]--;
    }
  }

  delayMicroseconds(pulseDelayUs);
}

void pollImuSerial() {
  static uint8_t buffer[128];
  static int index = 0;

  while (Serial1.available()) {
    uint8_t c = Serial1.read();
    if (index == 0 && c != 0xAA) continue;
    if (index == 1 && c != 0x55) { index = 0; continue; }

    buffer[index++] = c;

    if (index > 3 && buffer[2] == 0x14) {
      uint8_t targetLen = buffer[3] + 4;
      if (index == targetLen) {
        float val1, val2;
        memcpy(&val1, &buffer[11], 4);
        memcpy(&val2, &buffer[15], 4);
        imu_angle_a = val1;
        imu_angle_b = val2;
        imu_valid = true;
        index = 0;
      }
    }

    if (index >= 128) index = 0;
  }
}

void publishImuIfNeeded() {
  unsigned long now = millis();
  if (!imu_valid) return;
  if (now - lastImuPublishMs < imuPublishPeriodMs) return;
  lastImuPublishMs = now;

  Serial.print("IMU,");
  Serial.print(imu_angle_a, 4);
  Serial.print(",");
  Serial.println(imu_angle_b, 4);
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(5);
  Serial1.begin(115200);

  for (int i = 0; i < 3; ++i) {
    pinMode(stepPins[i], OUTPUT);
    pinMode(dirPins[i], OUTPUT);
    digitalWrite(stepPins[i], HIGH);
    digitalWrite(dirPins[i], LOW);
  }
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);

  Serial.println("Motor + IMU merged firmware started");
}

void loop() {
  pollMotorCommandSerial();
  pollImuSerial();
  publishImuIfNeeded();
  stepTowardTargets();
}
