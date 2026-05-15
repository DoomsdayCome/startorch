#include <startorch/common.hpp>
#include <startorch/device.hpp>
#include <startorch/memory.hpp>

#include <cstdint>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

namespace startorch {
TEST(ArenaTest, ConstructorCPUTest) {
  Arena a0(1_MiB, MemoryType::HOST, Device(CPU));

  EXPECT_EQ(a0.getSize(), 1_MiB);
  EXPECT_EQ(a0.getMemoryType(), MemoryType::HOST);
  EXPECT_NE(a0.getData(), nullptr);
}

TEST(ArenaTest, ConstructorGPUTest) {
  Arena a0(1_MiB, MemoryType::DEVICE, Device(GPU));

  EXPECT_EQ(a0.getSize(), 1_MiB);
  EXPECT_EQ(a0.getMemoryType(), MemoryType::DEVICE);
  EXPECT_NE(a0.getData(), nullptr);
}

TEST(ArenaTest, MakeDataCPUTest) {
  Arena a0(1_MiB, MemoryType::HOST, Device(CPU));
  void *p0 = a0.makeData(512_KiB);

  EXPECT_NE(p0, nullptr);
  EXPECT_EQ(a0.getOffset(), 512_KiB);
}

TEST(ArenaTest, MakeDataGPUTest) {
  Arena a0(1_MiB, MemoryType::DEVICE, Device(GPU));
  void *p0 = a0.makeData(512_KiB);

  EXPECT_NE(p0, nullptr);
  EXPECT_EQ(a0.getOffset(), 512_KiB);
}

TEST(ArenaTest, SetSizeCPUTest) {
  Arena a0(100, MemoryType::HOST, Device(CPU));
  int32_t *p0 = (int32_t *)a0.makeData(sizeof(int32_t));
  *p0 = 1234;

  a0.setSize(1_MiB);

  EXPECT_EQ(a0.getSize(), 1_MiB);
  EXPECT_EQ(*(int32_t *)a0.getData(), 1234);
}

TEST(ArenaTest, SetSizeGPUTest) {
  Arena a0(100, MemoryType::DEVICE, Device(GPU));
  a0.makeData(100);

  a0.setSize(2_MiB);

  EXPECT_EQ(a0.getSize(), 2_MiB);
  EXPECT_EQ(a0.getOffset(), 100);
}

TEST(ArenaTest, DataManagementTest) {
  Arena a0(1_MiB, MemoryType::HOST, Device(CPU));
  a0.makeData(500_KiB);
  a0.freeData(200_KiB);
  EXPECT_EQ(a0.getOffset(), 300_KiB);
  a0.wipeData();
  EXPECT_EQ(a0.getOffset(), 0);

  float f0 = 3.14f;
  float f1 = 0.0f;

  Arena a1(1_MiB, MemoryType::DEVICE, Device(GPU));
  void *p0 = a1.makeData(sizeof(float));

  Arena::copyData(p0, &f0, sizeof(float), DevicePair(Device(CPU), Device(GPU)));
  Arena::copyData(&f1, p0, sizeof(float), DevicePair(Device(GPU), Device(CPU)));

  EXPECT_FLOAT_EQ(f1, 3.14f);
}

TEST(StorageTest, CustomConstructorCPUTest) {
  Storage s0(256, ScalarType::FLOAT_32, &GLOBAL_PINNED_ARENA);
  EXPECT_EQ(s0.getSize(), 256);
  EXPECT_EQ(s0.getScalarType(), ScalarType::FLOAT_32);
}

TEST(StorageTest, CustomConstructorGPU) {
  Storage s0(256, ScalarType::FLOAT_32, &GLOBAL_DEVICE_ARENA);
  EXPECT_EQ(s0.getSize(), 256);
}

TEST(StorageTest, InitializerListCustomConstructorCPUTest) {
  Storage s0({1.1f, 2.2f, 3.3f}, &GLOBAL_PINNED_ARENA);
  EXPECT_EQ(s0.getSize(), 3);
  EXPECT_FLOAT_EQ(s0.getData<float>()[2], 3.3f);
}

TEST(StorageTest, CopyAndMoveTest) {
  Storage s0({10, 20}, &GLOBAL_PINNED_ARENA);
  Storage s1 = s0;
  EXPECT_NE(s0.getData(), s1.getData());
  EXPECT_EQ(s1.getData<int>()[0], 10);

  void *old_ptr = s1.getData();
  Storage s2 = std::move(s1);
  EXPECT_EQ(s2.getData(), old_ptr);
  EXPECT_EQ(s1.getData(), nullptr);
}

TEST(StorageTest, fillDataCPUTest) {
  Storage s0(10, ScalarType::INT_32, &GLOBAL_PINNED_ARENA);
  s0.fillData(77);
  EXPECT_EQ(s0.getData<int32_t>()[9], 77);
}

TEST(StorageTest, fillDataGPUTest) {
  Storage s0(10, ScalarType::FLOAT_32, &GLOBAL_DEVICE_ARENA);
  s0.fillData(1.23f);

  float res;

  Arena::copyData(&res, s0.getData(), sizeof(float), DevicePair(GLOBAL_DEVICE_ARENA.getDevice(), GLOBAL_PINNED_ARENA.getDevice()));
  EXPECT_FLOAT_EQ(res, 1.23f);
}

TEST(StorageTest, fillSequencedCPUTest) {
  Storage s0(5, ScalarType::INT_32, &GLOBAL_PINNED_ARENA);

  s0.fillIncreasedData(0, 2);
  EXPECT_EQ(s0.getData<int32_t>()[4], 8);

  s0.fillDecreasedData(5, 1);
  EXPECT_EQ(s0.getData<int32_t>()[0], 5);
  EXPECT_EQ(s0.getData<int32_t>()[4], 1);
}

TEST(StorageTest, fillSequencedGPUTest) {
  Storage s0(5, ScalarType::INT_32, &GLOBAL_DEVICE_ARENA);

  s0.fillDecreasedData(10, 2);
  cudaDeviceSynchronize();
  int32_t res[5] = {0};

  Arena::copyData(res, s0.getData(), sizeof(int32_t) * 5, DevicePair(GLOBAL_DEVICE_ARENA.getDevice(), GLOBAL_PINNED_ARENA.getDevice()));

  EXPECT_EQ(res[0], 10);
  EXPECT_EQ(res[1], 8);
  EXPECT_EQ(res[2], 6);
  EXPECT_EQ(res[3], 4);
  EXPECT_EQ(res[4], 2);
}
} // namespace startorch
