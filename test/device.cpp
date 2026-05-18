#include "startorch/common.hpp"
#include <cstdint>
#include <startorch/device.hpp>

#include <gtest/gtest.h>

namespace startorch {
TEST(DeviceTest, DefaultConstructorTest) {
  Device d0 = Device();

  EXPECT_EQ(d0.getData(), nullptr);
  EXPECT_EQ(d0.getBytes(), 0);
  EXPECT_EQ(d0.getOffset(), 0);
  EXPECT_EQ(d0.getMemoryType(), MemoryType::UNKNOWN_MEMORY);
  EXPECT_EQ(d0.getDeviceType(), DeviceType::UNKNOWN_DEVICE);
}

TEST(DeviceTest, CustomConstructorTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);
  Device d1 = Device(1_KiB, MemoryType::DEVICE, DeviceType::GPU);
  Device d2 = Device(1_KiB, MemoryType::UNIFIED, DeviceType::CPU);
  Device d3 = Device(1_KiB, MemoryType::PINNED, DeviceType::GPU);

  EXPECT_NE(d0.getData(), nullptr);
  EXPECT_EQ(d0.getBytes(), 1_KiB);
  EXPECT_EQ(d0.getOffset(), 0);
  EXPECT_EQ(d0.getMemoryType(), MemoryType::HOST);
  EXPECT_EQ(d0.getDeviceType(), DeviceType::CPU);

  EXPECT_NE(d1.getData(), nullptr);
  EXPECT_EQ(d1.getBytes(), 1_KiB);
  EXPECT_EQ(d1.getOffset(), 0);
  EXPECT_EQ(d1.getMemoryType(), MemoryType::DEVICE);
  EXPECT_EQ(d1.getDeviceType(), DeviceType::GPU);

  EXPECT_NE(d2.getData(), nullptr);
  EXPECT_EQ(d2.getBytes(), 1_KiB);
  EXPECT_EQ(d2.getOffset(), 0);
  EXPECT_EQ(d2.getMemoryType(), MemoryType::HOST);
  EXPECT_EQ(d2.getDeviceType(), DeviceType::CPU);

  EXPECT_NE(d3.getData(), nullptr);
  EXPECT_EQ(d3.getBytes(), 1_KiB);
  EXPECT_EQ(d3.getOffset(), 0);
  EXPECT_EQ(d3.getMemoryType(), MemoryType::DEVICE);
  EXPECT_EQ(d3.getDeviceType(), DeviceType::GPU);
}

TEST(DevicePairTest, DefaultConstructorTest) {
  DevicePair p0 = DevicePair();

  EXPECT_EQ(p0.getFirstDevice(), nullptr);
  EXPECT_EQ(p0.getSecondDevice(), nullptr);
}

TEST(DevicePairTest, CustomConstructorTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);
  Device d1 = Device(1_KiB, MemoryType::DEVICE, DeviceType::GPU);

  DevicePair p0 = DevicePair(&d0, &d1);
  DevicePair p1 = DevicePair(&d1, &d0);

  EXPECT_EQ(p0.getFirstDevice(), &d0);
  EXPECT_EQ(p0.getSecondDevice(), &d1);

  EXPECT_EQ(p1.getFirstDevice(), &d1);
  EXPECT_EQ(p1.getSecondDevice(), &d0);
}

TEST(DevicePairTest, CopyDataTest) {
  Device d0 = Device(10 * sizeof(int32_t), MemoryType::HOST, DeviceType::CPU);
  Device d1 = Device(10 * sizeof(int32_t), MemoryType::DEVICE, DeviceType::GPU);
  Device d2 = Device(10 * sizeof(int32_t), MemoryType::HOST, DeviceType::CPU);
  Device d3 = Device(10 * sizeof(int32_t), MemoryType::DEVICE, DeviceType::GPU);
  Device d4 = Device(10 * sizeof(int32_t), MemoryType::HOST, DeviceType::CPU);

  for (int32_t i = 0; i < 10; i++)
    static_cast<int32_t *>(d0.getData())[i] = i;

  DevicePair(&d2, &d0).copyData(d2.getData(), d0.getData(), 10 * sizeof(int32_t));

  for (int32_t i = 0; i < 10; i++)
    EXPECT_EQ(static_cast<int32_t *>(d2.getData())[i], i);

  for (int32_t i = 0; i < 10; i++)
    static_cast<int32_t *>(d0.getData())[i] = 10 - i;

  DevicePair(&d1, &d0).copyData(d1.getData(), d0.getData(), 10 * sizeof(int32_t));
  DevicePair(&d2, &d1).copyData(d2.getData(), d1.getData(), 10 * sizeof(int32_t));

  for (int32_t i = 0; i < 10; i++)
    EXPECT_EQ(static_cast<int32_t *>(d2.getData())[i], 10 - i);

  DevicePair(&d3, &d1).copyData(d3.getData(), d1.getData(), 10 * sizeof(int32_t));
  DevicePair(&d4, &d3).copyData(d4.getData(), d3.getData(), 10 * sizeof(int32_t));

  for (int32_t i = 0; i < 10; i++)
    EXPECT_EQ(static_cast<int32_t *>(d4.getData())[i], 10 - i);
}

TEST(DeviceTest, SetBytesTest) {
  Device d0 = Device(20 * sizeof(int32_t), MemoryType::HOST, DeviceType::CPU);
  Device d1 = Device(10 * sizeof(int32_t), MemoryType::DEVICE, DeviceType::GPU);
  Device d2 = Device(10 * sizeof(int32_t), MemoryType::HOST, DeviceType::CPU);

  for (int32_t i = 0; i < 20; i++)
    static_cast<int32_t *>(d0.getData())[i] = i;

  DevicePair(&d1, &d0).copyData(d1.getData(), d0.getData(), 10 * sizeof(int32_t));

  d0.setBytes(10 * sizeof(int32_t));
  d1.setBytes(20 * sizeof(int32_t));

  EXPECT_EQ(d0.getBytes(), 10 * sizeof(int32_t));

  for (int32_t i = 0; i < 10; i++)
    EXPECT_EQ(static_cast<int32_t *>(d0.getData())[i], i);

  EXPECT_EQ(d1.getBytes(), 20 * sizeof(int32_t));

  DevicePair(&d2, &d1).copyData(d2.getData(), d1.getData(), 10 * sizeof(int32_t));

  for (int32_t i = 0; i < 10; i++)
    EXPECT_EQ(static_cast<int32_t *>(d2.getData())[i], i);
}

TEST(DeviceTest, MakeDataTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);

  void *p0 = d0.makeData(1_KiB);
  void *p1 = d0.makeData(1_KiB);

  EXPECT_NE(p0, nullptr);
  EXPECT_EQ(p1, nullptr);

  EXPECT_EQ(d0.getOffset(), 1_KiB);
}

TEST(DeviceTest, FreeDataTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);

  d0.makeData(1_KiB);
  d0.freeData(512);

  EXPECT_EQ(d0.getOffset(), 512);
}

TEST(DeviceTest, WipeDataTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);

  d0.makeData(1_KiB);
  d0.freeData(512);
  d0.wipeData();

  EXPECT_EQ(d0.getOffset(), 0);
}
} // namespace startorch
