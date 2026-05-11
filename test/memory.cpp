#include <startorch/common.hpp>
#include <startorch/device.hpp>
#include <startorch/memory.hpp>

#include <gtest/gtest.h>

namespace startorch {
TEST(ArenaTest, DefaultConstructorTest) {
  Arena a0;

  EXPECT_EQ(a0.getSize(), 0);
  EXPECT_EQ(a0.getOffset(), 0);
}

TEST(ArenaTest, CustomConstructorTest) {
  Arena a0(2_MB, HOST, CPU);

  EXPECT_EQ(a0.getSize(), 2_MB);
  EXPECT_EQ(a0.getOffset(), 0);
  EXPECT_EQ(a0.getMemoryType(), HOST);
  EXPECT_EQ(a0.getDevice().getDeviceType(), CPU);

  Arena a1(2_MB, DEVICE, GPU);

  EXPECT_EQ(a1.getSize(), 2_MB);
  EXPECT_EQ(a1.getMemoryType(), DEVICE);
  EXPECT_EQ(a1.getDevice().getDeviceType(), GPU);

  Arena a2(0, HOST, GPU);

  EXPECT_EQ(a2.getSize(), 0);
  EXPECT_EQ(a2.getOffset(), 0);
  EXPECT_EQ(a2.getMemoryType(), DEVICE);
  EXPECT_EQ(a2.getDevice().getDeviceType(), GPU);
}

TEST(ArenaTest, MakeMemoryTest) {
  Arena a0(2_MB, HOST, CPU);

  void *p0 = a0.makeMemory(1_MB);
  void *p1 = a0.makeMemory(1_MB);

  EXPECT_NE(p0, nullptr);
  EXPECT_NE(p1, nullptr);
  EXPECT_EQ(a0.getOffset(), 2_MB);

  EXPECT_EQ((uint8_t *)p0 + 1_MB, (uint8_t *)p1);

  void *p2 = a0.makeMemory(60);
  EXPECT_EQ(p2, nullptr);
  EXPECT_EQ(a0.getOffset(), 2_MB);
}

TEST(ArenaTest, FreeMemoryTest) {
  Arena a0(100_MB, HOST, CPU);

  a0.makeMemory(50_MB);
  EXPECT_EQ(a0.getOffset(), 50_MB);

  a0.freeMemory(20_MB);
  EXPECT_EQ(a0.getOffset(), 30_MB);

  a0.freeMemory(100_MB);
  EXPECT_EQ(a0.getOffset(), 0);
}

TEST(ArenaTest, WipeMemoryTest) {
  Arena a0(500_MB, HOST, CPU);

  a0.makeMemory(100_MB);
  a0.makeMemory(200_MB);
  EXPECT_EQ(a0.getOffset(), 300_MB);

  a0.wipeMemory();
  EXPECT_EQ(a0.getOffset(), 0);

  void *p_new = a0.makeMemory(10);
  EXPECT_NE(p_new, nullptr);
}

TEST(ArenaTest, CopyMemoryTest) {
  uint64_t size = 4;
  Arena a0(size, HOST, CPU);
  Arena a1(size, HOST, CPU);

  uint8_t *p0 = (uint8_t *)a0.makeMemory(size);
  uint8_t *p1 = (uint8_t *)a1.makeMemory(size);

  p0[0] = 10;
  p0[1] = 20;
  p0[2] = 30;
  p0[3] = 40;

  DevicePair d0(CPU, CPU);
  Arena::copyMemory(p1, p0, size, d0);

  EXPECT_EQ(p1[0], 10);
  EXPECT_EQ(p1[3], 40);
}
} // namespace startorch
