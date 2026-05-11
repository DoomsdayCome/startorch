#include <startorch/common.hpp>
#include <startorch/device.hpp>

#include <gtest/gtest.h>

namespace startorch {
TEST(DeviceTest, DefaultConstructorTest) {
  Device d0;
  EXPECT_EQ(d0.getDeviceType(), CPU);
}

TEST(DeviceTest, CustomConstructorTest) {
  Device d0(CPU);
  Device d1(GPU);

  EXPECT_EQ(d0.getDeviceType(), CPU);
  EXPECT_EQ(d1.getDeviceType(), GPU);
}

TEST(DeviceTest, EqualityOperatorTest) {
  Device d0(GPU);
  Device d1(GPU);
  Device d2(CPU);

  EXPECT_TRUE(d0 == d1);
  EXPECT_FALSE(d0 == d2); 
  EXPECT_TRUE(d0 != d2);  
}

TEST(DevicePairTest, DefaultConstructorTest) {
  DevicePair p0;
  Device d0;

  EXPECT_EQ(p0.getFirstDevice(), d0);
  EXPECT_EQ(p0.getSecondDevice(), d0);
}

TEST(DevicePairTest, CustomConstructorTest) {
  Device d0(CPU);
  Device d1(GPU);

  DevicePair p0(d0, d1);

  EXPECT_EQ(p0.getFirstDevice(), d0);
  EXPECT_EQ(p0.getSecondDevice(), d1);
  EXPECT_EQ(p0.getFirstDevice().getDeviceType(), CPU);
  EXPECT_EQ(p0.getSecondDevice().getDeviceType(), GPU);
}

TEST(DevicePairTest, EqualityOperatorTest) {
  Device d0(CPU);
  Device d1(GPU);

  DevicePair p0(d0, d1);
  DevicePair p1(d0, d1);
  DevicePair p2(d1, d0);

  EXPECT_TRUE(p0 == p1);
  EXPECT_FALSE(p0 == p2);
  EXPECT_TRUE(p0 != p2);
}
} // namespace startorch
