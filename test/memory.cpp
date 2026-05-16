#include <startorch/common.hpp>
#include <startorch/device.hpp>
#include <startorch/format.hpp>
#include <startorch/memory.hpp>

#include <cstdint>

#include <gtest/gtest.h>

namespace startorch {
TEST(ArenaTest, DefaultConstructorTest) {
  Arena a0 = Arena();

  EXPECT_EQ(a0.getData(), nullptr);
  EXPECT_EQ(a0.getSize(), 0u);
  EXPECT_EQ(a0.getOffset(), 0u);
  EXPECT_EQ(a0.getMemoryType(), MemoryType::UNKNOWN_MEMORY);
  EXPECT_EQ(a0.getDevice().getDeviceType(), DeviceType::UNKNOWN_DEVICE);
}

TEST(ArenaTest, CustomConstructorPinnedTest) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));

  EXPECT_NE(a0.getData(), nullptr);
  EXPECT_EQ(a0.getSize(), 1_MiB);
  EXPECT_EQ(a0.getOffset(), 0u);
  EXPECT_EQ(a0.getMemoryType(), PINNED);
  EXPECT_EQ(a0.getDevice().getDeviceType(), CPU);
}

TEST(ArenaTest, CustomConstructorHostTest) {
  Arena a0 = Arena(512_KiB, HOST, Device(CPU));

  EXPECT_NE(a0.getData(), nullptr);
  EXPECT_EQ(a0.getSize(), 512_KiB);
  EXPECT_EQ(a0.getMemoryType(), HOST);
  EXPECT_EQ(a0.getDevice().getDeviceType(), CPU);
}

TEST(ArenaTest, CustomConstructorDeviceTest) {
  Arena a0 = Arena(1_MiB, DEVICE, Device(GPU));

  EXPECT_NE(a0.getData(), nullptr);
  EXPECT_EQ(a0.getSize(), 1_MiB);
  EXPECT_EQ(a0.getOffset(), 0u);
  EXPECT_EQ(a0.getMemoryType(), DEVICE);
  EXPECT_EQ(a0.getDevice().getDeviceType(), GPU);
}

TEST(ArenaTest, CustomConstructorUnifiedTest) {
  Arena a0 = Arena(256_KiB, UNIFIED, Device(GPU));

  EXPECT_NE(a0.getData(), nullptr);
  EXPECT_EQ(a0.getSize(), 256_KiB);
  EXPECT_EQ(a0.getMemoryType(), UNIFIED);
  EXPECT_EQ(a0.getDevice().getDeviceType(), GPU);
}

TEST(ArenaTest, SetSizeTest) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));

  EXPECT_EQ(a0.getSize(), 1_MiB);

  a0.setSize(2_MiB);

  EXPECT_EQ(a0.getSize(), 2_MiB);

  a0.setSize(0u);

  EXPECT_EQ(a0.getSize(), 0u);
}

TEST(ArenaTest, MakeDataAndFreeDataCPUTest) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));

  EXPECT_EQ(a0.getOffset(), 0u);

  void *p0 = a0.makeData(64u);

  EXPECT_NE(p0, nullptr);
  EXPECT_EQ(a0.getOffset(), 64u);

  void *p1 = a0.makeData(128u);

  EXPECT_NE(p1, nullptr);
  EXPECT_EQ(a0.getOffset(), 192u);

  a0.freeData(128u);

  EXPECT_EQ(a0.getOffset(), 64u);

  a0.freeData(64u);

  EXPECT_EQ(a0.getOffset(), 0u);
}

TEST(ArenaTest, MakeDataAndFreeDataGPUTest) {
  Arena a0 = Arena(1_MiB, DEVICE, Device(GPU));

  EXPECT_EQ(a0.getOffset(), 0u);

  void *p0 = a0.makeData(64u);

  EXPECT_NE(p0, nullptr);
  EXPECT_EQ(a0.getOffset(), 64u);

  void *p1 = a0.makeData(256u);

  EXPECT_NE(p1, nullptr);
  EXPECT_EQ(a0.getOffset(), 320u);

  a0.freeData(256u);

  EXPECT_EQ(a0.getOffset(), 64u);

  a0.freeData(64u);

  EXPECT_EQ(a0.getOffset(), 0u);
}

TEST(ArenaTest, WipeDataCPUTest) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));
  a0.makeData(512u);

  EXPECT_EQ(a0.getOffset(), 512u);

  a0.wipeData();

  EXPECT_EQ(a0.getOffset(), 0u);
}

TEST(ArenaTest, WipeDataGPUTest) {
  Arena a0 = Arena(1_MiB, DEVICE, Device(GPU));
  a0.makeData(256u);

  EXPECT_EQ(a0.getOffset(), 256u);

  a0.wipeData();

  EXPECT_EQ(a0.getOffset(), 0u);
}

TEST(ArenaTest, CopyDataCPUToCPUTest) {
  Arena a0 = Arena(1_KiB, PINNED, Device(CPU));
  Arena a1 = Arena(1_KiB, PINNED, Device(CPU));

  int *p0 = (int *)a0.makeData(4 * sizeof(int));
  p0[0] = 10;
  p0[1] = 20;
  p0[2] = 30;
  p0[3] = 40;

  int *p1 = (int *)a1.makeData(4 * sizeof(int));

  DevicePair dp0 = DevicePair(Device(CPU), Device(CPU));
  Arena::copyData(p1, p0, 4 * sizeof(int), dp0);

  EXPECT_EQ(p1[0], 10);
  EXPECT_EQ(p1[1], 20);
  EXPECT_EQ(p1[2], 30);
  EXPECT_EQ(p1[3], 40);
}

TEST(ArenaTest, CopyDataCPUToGPUTest) {
  Arena a0 = Arena(1_KiB, PINNED, Device(CPU));
  Arena a1 = Arena(1_KiB, DEVICE, Device(GPU));
  Arena a2 = Arena(1_KiB, PINNED, Device(CPU));

  int *p0 = (int *)a0.makeData(4 * sizeof(int));
  p0[0] = 1;
  p0[1] = 2;
  p0[2] = 3;
  p0[3] = 4;

  int *p1 = (int *)a1.makeData(4 * sizeof(int));
  int *p2 = (int *)a2.makeData(4 * sizeof(int));

  DevicePair dp0 = DevicePair(Device(CPU), Device(GPU));
  Arena::copyData(p1, p0, 4 * sizeof(int), dp0);

  DevicePair dp1 = DevicePair(Device(GPU), Device(CPU));
  Arena::copyData(p2, p1, 4 * sizeof(int), dp1);

  EXPECT_EQ(p2[0], 1);
  EXPECT_EQ(p2[1], 2);
  EXPECT_EQ(p2[2], 3);
  EXPECT_EQ(p2[3], 4);
}

TEST(ArenaTest, CopyDataGPUToGPUTest) {
  Arena a0 = Arena(1_KiB, PINNED, Device(CPU));
  Arena a1 = Arena(1_KiB, DEVICE, Device(GPU));
  Arena a2 = Arena(1_KiB, DEVICE, Device(GPU));
  Arena a3 = Arena(1_KiB, PINNED, Device(CPU));

  int *p0 = (int *)a0.makeData(3 * sizeof(int));
  p0[0] = 7;
  p0[1] = 8;
  p0[2] = 9;

  int *p1 = (int *)a1.makeData(3 * sizeof(int));
  int *p2 = (int *)a2.makeData(3 * sizeof(int));
  int *p3 = (int *)a3.makeData(3 * sizeof(int));

  DevicePair dp0 = DevicePair(Device(CPU), Device(GPU));
  Arena::copyData(p1, p0, 3 * sizeof(int), dp0);

  DevicePair dp1 = DevicePair(Device(GPU), Device(GPU));
  Arena::copyData(p2, p1, 3 * sizeof(int), dp1);

  DevicePair dp2 = DevicePair(Device(GPU), Device(CPU));
  Arena::copyData(p3, p2, 3 * sizeof(int), dp2);

  EXPECT_EQ(p3[0], 7);
  EXPECT_EQ(p3[1], 8);
  EXPECT_EQ(p3[2], 9);
}

TEST(StorageTest, DefaultConstructorTest) {
  Storage s0 = Storage();

  EXPECT_EQ(s0.getData(), nullptr);
  EXPECT_EQ(s0.getSize(), 0u);
  EXPECT_EQ(s0.getScalarType(), ScalarType::UNKNOWN_SCALAR);
  EXPECT_EQ(s0.getArena(), nullptr);
}

TEST(StorageTest, CustomConstructorCPUTest) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));
  Storage s0 = Storage(8u, INT_32, &a0);

  EXPECT_NE(s0.getData(), nullptr);
  EXPECT_EQ(s0.getSize(), 8u);
  EXPECT_EQ(s0.getScalarType(), INT_32);
  EXPECT_EQ(s0.getArena(), &a0);
}

TEST(StorageTest, CustomConstructorGPUTest) {
  Arena a0 = Arena(1_MiB, DEVICE, Device(GPU));
  Storage s0 = Storage(4u, FLOAT_32, &a0);

  EXPECT_NE(s0.getData(), nullptr);
  EXPECT_EQ(s0.getSize(), 4u);
  EXPECT_EQ(s0.getScalarType(), FLOAT_32);
  EXPECT_EQ(s0.getArena(), &a0);
}

TEST(StorageTest, InitializerListConstructorTest) {
  Storage s0 = Storage({Element(1), Element(2), Element(3)});

  EXPECT_NE(s0.getData(), nullptr);
  EXPECT_EQ(s0.getSize(), 3u);
  EXPECT_EQ(s0.getScalarType(), INT_32);

  Storage s1 = Storage({Element(1.0), Element(2.0), Element(3.0), Element(4.0)});

  EXPECT_NE(s1.getData(), nullptr);
  EXPECT_EQ(s1.getSize(), 4u);
  EXPECT_EQ(s1.getScalarType(), FLOAT_64);
}

TEST(StorageTest, InitializerListWithExplicitArenaTest) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));
  Storage s0 = Storage({Element(10), Element(20), Element(30)}, &a0);

  EXPECT_NE(s0.getData(), nullptr);
  EXPECT_EQ(s0.getSize(), 3u);
  EXPECT_EQ(s0.getArena(), &a0);
  EXPECT_EQ(s0.getScalarType(), INT_32);
}

TEST(StorageTest, CopyConstructorCPUTest) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));
  Storage s0 = Storage(8u, INT_64, &a0);
  Storage s1 = Storage(s0);

  EXPECT_NE(s1.getData(), nullptr);
  EXPECT_EQ(s1.getSize(), s0.getSize());
  EXPECT_EQ(s1.getScalarType(), s0.getScalarType());
  EXPECT_EQ(s1.getArena(), s0.getArena());
}

TEST(StorageTest, CopyConstructorGPUTest) {
  Arena a0 = Arena(1_MiB, DEVICE, Device(GPU));
  Storage s0 = Storage(4u, FLOAT_64, &a0);
  Storage s1 = Storage(s0);

  EXPECT_NE(s1.getData(), nullptr);
  EXPECT_EQ(s1.getSize(), s0.getSize());
  EXPECT_EQ(s1.getScalarType(), s0.getScalarType());
  EXPECT_EQ(s1.getArena(), s0.getArena());
}

TEST(StorageTest, MoveConstructorCPUTest) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));
  Storage s0 = Storage(6u, UNSIGNED_INT_32, &a0);
  uint64_t z0 = s0.getSize();
  Storage s1 = Storage(std::move(s0));

  EXPECT_NE(s1.getData(), nullptr);
  EXPECT_EQ(s1.getSize(), z0);
  EXPECT_EQ(s1.getScalarType(), UNSIGNED_INT_32);
}

TEST(StorageTest, MoveConstructorGPUTest) {
  Arena a0 = Arena(1_MiB, DEVICE, Device(GPU));
  Storage s0 = Storage(5u, INT_16, &a0);
  uint64_t z0 = s0.getSize();
  Storage s1 = Storage(std::move(s0));

  EXPECT_NE(s1.getData(), nullptr);
  EXPECT_EQ(s1.getSize(), z0);
  EXPECT_EQ(s1.getScalarType(), INT_16);
}

TEST(StorageTest, CopyAssignmentCPUTest) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));
  Storage s0 = Storage(10u, INT_32, &a0);
  Storage s1 = Storage();
  s1 = s0;

  EXPECT_NE(s1.getData(), nullptr);
  EXPECT_EQ(s1.getSize(), 10u);
  EXPECT_EQ(s1.getScalarType(), INT_32);
  EXPECT_EQ(s1.getArena(), &a0);
}

TEST(StorageTest, CopyAssignmentGPUTest) {
  Arena a0 = Arena(1_MiB, DEVICE, Device(GPU));
  Storage s0 = Storage(8u, FLOAT_32, &a0);
  Storage s1 = Storage();
  s1 = s0;

  EXPECT_NE(s1.getData(), nullptr);
  EXPECT_EQ(s1.getSize(), 8u);
  EXPECT_EQ(s1.getScalarType(), FLOAT_32);
}

TEST(StorageTest, MoveAssignmentCPUTest) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));
  Storage s0 = Storage(12u, INT_8, &a0);
  Storage s1 = Storage();
  s1 = std::move(s0);

  EXPECT_NE(s1.getData(), nullptr);
  EXPECT_EQ(s1.getSize(), 12u);
  EXPECT_EQ(s1.getScalarType(), INT_8);
}

TEST(StorageTest, MoveAssignmentGPUTest) {
  Arena a0 = Arena(1_MiB, DEVICE, Device(GPU));
  Storage s0 = Storage(7u, UNSIGNED_INT_64, &a0);
  Storage s1 = Storage();
  s1 = std::move(s0);

  EXPECT_NE(s1.getData(), nullptr);
  EXPECT_EQ(s1.getSize(), 7u);
  EXPECT_EQ(s1.getScalarType(), UNSIGNED_INT_64);
}

TEST(StorageTest, FillDataCPUInt32Test) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));
  Storage s0 = Storage(5u, INT_32, &a0);

  s0.fillData(Element((int32_t)42));

  int32_t *p0 = s0.getData<int32_t>();

  for (uint64_t i = 0; i < 5u; ++i)
    EXPECT_EQ(p0[i], 42);
}

TEST(StorageTest, FillDataCPUFloat32Test) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));
  Storage s0 = Storage(4u, FLOAT_32, &a0);

  s0.fillData(Element(3.14f));

  float *p0 = s0.getData<float>();

  for (uint64_t i = 0; i < 4u; ++i)
    EXPECT_FLOAT_EQ(p0[i], 3.14f);
}

TEST(StorageTest, FillIncreasedDataCPUInt32Test) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));
  Storage s0 = Storage(5u, INT_32, &a0);

  s0.fillIncreasedData(Element((int32_t)0), Element((int32_t)2));

  int32_t *p0 = s0.getData<int32_t>();

  EXPECT_EQ(p0[0], 0);
  EXPECT_EQ(p0[1], 2);
  EXPECT_EQ(p0[2], 4);
  EXPECT_EQ(p0[3], 6);
  EXPECT_EQ(p0[4], 8);
}

TEST(StorageTest, FillIncreasedDataCPUFloat32Test) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));
  Storage s0 = Storage(4u, FLOAT_32, &a0);

  s0.fillIncreasedData(Element(1.0f), Element(0.5f));

  float *p0 = s0.getData<float>();

  EXPECT_FLOAT_EQ(p0[0], 1.0f);
  EXPECT_FLOAT_EQ(p0[1], 1.5f);
  EXPECT_FLOAT_EQ(p0[2], 2.0f);
  EXPECT_FLOAT_EQ(p0[3], 2.5f);
}

TEST(StorageTest, FillDecreasedDataCPUInt32Test) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));
  Storage s0 = Storage(5u, INT_32, &a0);

  s0.fillDecreasedData(Element((int32_t)10), Element((int32_t)3));

  int32_t *p0 = s0.getData<int32_t>();

  EXPECT_EQ(p0[0], 10);
  EXPECT_EQ(p0[1], 7);
  EXPECT_EQ(p0[2], 4);
  EXPECT_EQ(p0[3], 1);
  EXPECT_EQ(p0[4], -2);
}

TEST(StorageTest, FillDecreasedDataCPUFloat64Test) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));
  Storage s0 = Storage(4u, FLOAT_64, &a0);

  s0.fillDecreasedData(Element(5.0), Element(1.5));

  double *p0 = s0.getData<double>();

  EXPECT_DOUBLE_EQ(p0[0], 5.0);
  EXPECT_DOUBLE_EQ(p0[1], 3.5);
  EXPECT_DOUBLE_EQ(p0[2], 2.0);
  EXPECT_DOUBLE_EQ(p0[3], 0.5);
}

TEST(StorageTest, ScalarTypeTest) {
  Arena a0 = Arena(1_MiB, PINNED, Device(CPU));
  Storage s0 = Storage(1u, INT_8, &a0);
  Storage s1 = Storage(1u, INT_16, &a0);
  Storage s2 = Storage(1u, INT_32, &a0);
  Storage s3 = Storage(1u, INT_64, &a0);
  Storage s4 = Storage(1u, FLOAT_32, &a0);
  Storage s5 = Storage(1u, FLOAT_64, &a0);
  Storage s6 = Storage(1u, UNSIGNED_INT_8, &a0);
  Storage s7 = Storage(1u, UNSIGNED_INT_16, &a0);
  Storage s8 = Storage(1u, UNSIGNED_INT_32, &a0);
  Storage s9 = Storage(1u, UNSIGNED_INT_64, &a0);

  EXPECT_EQ(s0.getScalarType(), INT_8);
  EXPECT_EQ(s1.getScalarType(), INT_16);
  EXPECT_EQ(s2.getScalarType(), INT_32);
  EXPECT_EQ(s3.getScalarType(), INT_64);
  EXPECT_EQ(s4.getScalarType(), FLOAT_32);
  EXPECT_EQ(s5.getScalarType(), FLOAT_64);
  EXPECT_EQ(s6.getScalarType(), UNSIGNED_INT_8);
  EXPECT_EQ(s7.getScalarType(), UNSIGNED_INT_16);
  EXPECT_EQ(s8.getScalarType(), UNSIGNED_INT_32);
  EXPECT_EQ(s9.getScalarType(), UNSIGNED_INT_64);
}
} // namespace startorch
