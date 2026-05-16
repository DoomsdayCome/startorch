#include <startorch/common.hpp>
#include <startorch/device.hpp>

#include <gtest/gtest.h>

namespace startorch {
TEST(DeviceTest, DefaultConstructorTest) {
  Device d0 = Device();

  EXPECT_EQ(d0.getDeviceType(), DeviceType::UNKNOWN_DEVICE);
}

TEST(DeviceTest, CustomConstructorTest) {
  Device d0 = Device(CPU);
  Device d1 = Device(GPU);

  EXPECT_EQ(d0.getDeviceType(), CPU);
  EXPECT_EQ(d1.getDeviceType(), GPU);
}

TEST(DeviceTest, CopyConstructorTest) {
  Device d0 = Device(CPU);
  Device d1 = Device(d0);
  Device d2 = Device(GPU);
  Device d3 = Device(d2);

  EXPECT_EQ(d1.getDeviceType(), CPU);
  EXPECT_EQ(d3.getDeviceType(), GPU);
}

TEST(DeviceTest, MoveConstructorTest) {
  Device d0 = Device(CPU);
  Device d1 = Device(std::move(d0));
  Device d2 = Device(GPU);
  Device d3 = Device(std::move(d2));

  EXPECT_EQ(d1.getDeviceType(), CPU);
  EXPECT_EQ(d3.getDeviceType(), GPU);
}

TEST(DeviceTest, CopyAssignmentTest) {
  Device d0 = Device(CPU);
  Device d1 = Device();
  d1 = d0;
  Device d2 = Device(GPU);
  Device d3 = Device();
  d3 = d2;

  EXPECT_EQ(d1.getDeviceType(), CPU);
  EXPECT_EQ(d3.getDeviceType(), GPU);
}

TEST(DeviceTest, MoveAssignmentTest) {
  Device d0 = Device(CPU);
  Device d1 = Device();
  d1 = std::move(d0);
  Device d2 = Device(GPU);
  Device d3 = Device();
  d3 = std::move(d2);

  EXPECT_EQ(d1.getDeviceType(), CPU);
  EXPECT_EQ(d3.getDeviceType(), GPU);
}

TEST(DeviceTest, EqualityOperatorTest) {
  Device d0 = Device(CPU);
  Device d1 = Device(CPU);
  Device d2 = Device(GPU);
  Device d3 = Device(GPU);
  Device d4 = Device();
  Device d5 = Device();

  EXPECT_TRUE(d0 == d1);
  EXPECT_TRUE(d2 == d3);
  EXPECT_TRUE(d4 == d5);

  EXPECT_FALSE(d0 == d2);
  EXPECT_FALSE(d0 == d4);
  EXPECT_FALSE(d2 == d4);
}

TEST(DeviceTest, InequalityOperatorTest) {
  Device d0 = Device(CPU);
  Device d1 = Device(GPU);
  Device d2 = Device(CPU);
  Device d3 = Device();

  EXPECT_TRUE(d0 != d1);
  EXPECT_TRUE(d0 != d3);
  EXPECT_TRUE(d1 != d3);
  EXPECT_FALSE(d0 != d2);
}

TEST(DevicePairTest, DefaultConstructorTest) {
  DevicePair p0 = DevicePair();
  Device d0 = Device();

  EXPECT_EQ(p0.getFirstDevice().getDeviceType(), DeviceType::UNKNOWN_DEVICE);
  EXPECT_EQ(p0.getSecondDevice().getDeviceType(), DeviceType::UNKNOWN_DEVICE);
  EXPECT_EQ(p0.getFirstDevice(), d0);
  EXPECT_EQ(p0.getSecondDevice(), d0);
}

TEST(DevicePairTest, CustomConstructorTest) {
  Device d0 = Device(CPU);
  Device d1 = Device(GPU);
  DevicePair p0 = DevicePair(d0, d1);
  DevicePair p1 = DevicePair(d1, d0);

  EXPECT_EQ(p0.getFirstDevice().getDeviceType(), CPU);
  EXPECT_EQ(p0.getSecondDevice().getDeviceType(), GPU);
  EXPECT_EQ(p1.getFirstDevice().getDeviceType(), GPU);
  EXPECT_EQ(p1.getSecondDevice().getDeviceType(), CPU);
}

TEST(DevicePairTest, GetDeviceTest) {
  Device d0 = Device(CPU);
  Device d1 = Device(CPU);
  DevicePair p0 = DevicePair(d0, d1);

  EXPECT_EQ(p0.getFirstDevice().getDeviceType(), CPU);
  EXPECT_EQ(p0.getSecondDevice().getDeviceType(), CPU);

  Device d2 = Device(GPU);
  Device d3 = Device(GPU);
  DevicePair p1 = DevicePair(d2, d3);

  EXPECT_EQ(p1.getFirstDevice().getDeviceType(), GPU);
  EXPECT_EQ(p1.getSecondDevice().getDeviceType(), GPU);
}

TEST(DevicePairTest, CopyConstructorTest) {
  Device d0 = Device(CPU);
  Device d1 = Device(GPU);
  DevicePair p0 = DevicePair(d0, d1);
  DevicePair p1 = DevicePair(p0);

  EXPECT_EQ(p1.getFirstDevice().getDeviceType(), CPU);
  EXPECT_EQ(p1.getSecondDevice().getDeviceType(), GPU);
}

TEST(DevicePairTest, MoveConstructorTest) {
  Device d0 = Device(CPU);
  Device d1 = Device(GPU);
  DevicePair p0 = DevicePair(d0, d1);
  DevicePair p1 = DevicePair(std::move(p0));

  EXPECT_EQ(p1.getFirstDevice().getDeviceType(), CPU);
  EXPECT_EQ(p1.getSecondDevice().getDeviceType(), GPU);
}

TEST(DevicePairTest, CopyAssignmentTest) {
  Device d0 = Device(CPU);
  Device d1 = Device(GPU);
  DevicePair p0 = DevicePair(d0, d1);
  DevicePair p1 = DevicePair();
  p1 = p0;

  EXPECT_EQ(p1.getFirstDevice().getDeviceType(), CPU);
  EXPECT_EQ(p1.getSecondDevice().getDeviceType(), GPU);
}

TEST(DevicePairTest, MoveAssignmentTest) {
  Device d0 = Device(CPU);
  Device d1 = Device(GPU);
  DevicePair p0 = DevicePair(d0, d1);
  DevicePair p1 = DevicePair();
  p1 = std::move(p0);

  EXPECT_EQ(p1.getFirstDevice().getDeviceType(), CPU);
  EXPECT_EQ(p1.getSecondDevice().getDeviceType(), GPU);
}

TEST(DevicePairTest, EqualityOperatorTest) {
  Device d0 = Device(CPU);
  Device d1 = Device(GPU);

  DevicePair p0 = DevicePair(d0, d1);
  DevicePair p1 = DevicePair(d0, d1);
  DevicePair p2 = DevicePair(d1, d0);
  DevicePair p3 = DevicePair(d0, d0);
  DevicePair p4 = DevicePair(d1, d1);

  EXPECT_TRUE(p0 == p1);
  EXPECT_FALSE(p0 == p2);
  EXPECT_FALSE(p0 == p3);
  EXPECT_FALSE(p0 == p4);
}

TEST(DevicePairTest, InequalityOperatorTest) {
  Device d0 = Device(CPU);
  Device d1 = Device(GPU);

  DevicePair p0 = DevicePair(d0, d1);
  DevicePair p1 = DevicePair(d0, d1);
  DevicePair p2 = DevicePair(d1, d0);

  EXPECT_FALSE(p0 != p1);
  EXPECT_TRUE(p0 != p2);
}
} // namespace startorch
