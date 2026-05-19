#include <startorch/layout.hpp>

#include <gtest/gtest.h>

namespace startorch {
static Storage makeUint64Storage(std::initializer_list<uint64_t> values, Device *device) {
  Storage s = Storage(values.size(), ScalarType::UNSIGNED_INT_64, device);
  uint64_t idx = 0;

  for (uint64_t v : values)
    s.getData<uint64_t>()[idx++] = v;

  return s;
}

TEST(LayoutTest, DefaultConstructorTest) {
  Layout l0 = Layout();
  Storage s0 = l0.getShape();
  Storage s1 = l0.getOrder();
  Storage s2 = l0.getStrides();
  Storage s3 = l0.getOffsets();

  EXPECT_EQ(s0.getData(), nullptr);
  EXPECT_EQ(s0.getSize(), 0);
  EXPECT_EQ(s0.getScalarType(), ScalarType::UNKNOWN_SCALAR);
  EXPECT_EQ(s0.getDevice(), nullptr);

  EXPECT_EQ(s1.getData(), nullptr);
  EXPECT_EQ(s1.getSize(), 0);
  EXPECT_EQ(s1.getScalarType(), ScalarType::UNKNOWN_SCALAR);
  EXPECT_EQ(s1.getDevice(), nullptr);

  EXPECT_EQ(s2.getData(), nullptr);
  EXPECT_EQ(s2.getSize(), 0);
  EXPECT_EQ(s2.getScalarType(), ScalarType::UNKNOWN_SCALAR);
  EXPECT_EQ(s2.getDevice(), nullptr);

  EXPECT_EQ(s3.getData(), nullptr);
  EXPECT_EQ(s3.getSize(), 0);
  EXPECT_EQ(s3.getScalarType(), ScalarType::UNKNOWN_SCALAR);
  EXPECT_EQ(s3.getDevice(), nullptr);
}

TEST(LayoutTest, CustomStorageOrderConstructorTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);

  Storage s0 = makeUint64Storage({3, 4}, &d0);
  Storage s1 = makeUint64Storage({0, 1}, &d0);
  Storage s2 = makeUint64Storage({4, 1}, &d0);
  Storage s3 = makeUint64Storage({0, 0}, &d0);

  Layout l0 = Layout(s0, s1, s2, s3);

  EXPECT_EQ(l0.getShape().getSize(), 2);
  EXPECT_EQ(l0.getOrder().getSize(), 2);
  EXPECT_EQ(l0.getStrides().getSize(), 2);
  EXPECT_EQ(l0.getOffsets().getSize(), 2);

  EXPECT_EQ(l0.getShape().getData<uint64_t>()[0], 3);
  EXPECT_EQ(l0.getShape().getData<uint64_t>()[1], 4);
  EXPECT_EQ(l0.getOrder().getData<uint64_t>()[0], 0);
  EXPECT_EQ(l0.getOrder().getData<uint64_t>()[1], 1);

  EXPECT_EQ(l0.getStrides().getData<uint64_t>()[0], 4);
  EXPECT_EQ(l0.getStrides().getData<uint64_t>()[1], 1);
  EXPECT_EQ(l0.getOffsets().getData<uint64_t>()[0], 0);
  EXPECT_EQ(l0.getOffsets().getData<uint64_t>()[1], 0);
}

TEST(LayoutTest, CustomRowMajorConstructorTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);

  Storage s0 = makeUint64Storage({3, 4}, &d0);
  Storage s1 = makeUint64Storage({4, 1}, &d0);
  Storage s2 = makeUint64Storage({0, 0}, &d0);

  Layout l0 = Layout(s0, OrderType::ROW_MAJOR, s1, s2);

  EXPECT_EQ(l0.getShape().getSize(), 2);
  EXPECT_EQ(l0.getStrides().getSize(), 2);
  EXPECT_EQ(l0.getOffsets().getSize(), 2);

  EXPECT_EQ(l0.getShape().getData<uint64_t>()[0], 3);
  EXPECT_EQ(l0.getShape().getData<uint64_t>()[1], 4);
  EXPECT_EQ(l0.getStrides().getData<uint64_t>()[0], 4);
  EXPECT_EQ(l0.getStrides().getData<uint64_t>()[1], 1);
  EXPECT_EQ(l0.getOffsets().getData<uint64_t>()[0], 0);
  EXPECT_EQ(l0.getOffsets().getData<uint64_t>()[1], 0);
}

TEST(LayoutTest, CustomColumnMajorConstructorTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);

  Storage s0 = makeUint64Storage({3, 4}, &d0);
  Storage s1 = makeUint64Storage({1, 3}, &d0);
  Storage s2 = makeUint64Storage({0, 0}, &d0);

  Layout l0 = Layout(s0, OrderType::COLUMN_MAJOR, s1, s2);

  EXPECT_EQ(l0.getShape().getData<uint64_t>()[0], 3);
  EXPECT_EQ(l0.getShape().getData<uint64_t>()[1], 4);
  EXPECT_EQ(l0.getStrides().getData<uint64_t>()[0], 1);
  EXPECT_EQ(l0.getStrides().getData<uint64_t>()[1], 3);
}

TEST(LayoutTest, CopyConstructorTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);

  Storage s0 = makeUint64Storage({2, 3}, &d0);
  Storage s1 = makeUint64Storage({3, 1}, &d0);
  Storage s2 = makeUint64Storage({0, 0}, &d0);

  Layout l0 = Layout(s0, OrderType::ROW_MAJOR, s1, s2);
  Layout l1 = Layout(l0);

  EXPECT_EQ(l1.getShape().getSize(), 2);

  EXPECT_EQ(l1.getShape().getData<uint64_t>()[0], 2);
  EXPECT_EQ(l1.getShape().getData<uint64_t>()[1], 3);

  EXPECT_EQ(l1.getStrides().getData<uint64_t>()[0], 3);
  EXPECT_EQ(l1.getStrides().getData<uint64_t>()[1], 1);
}

TEST(LayoutTest, MoveConstructorTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);

  Storage s0 = makeUint64Storage({2, 3}, &d0);
  Storage s1 = makeUint64Storage({3, 1}, &d0);
  Storage s2 = makeUint64Storage({0, 0}, &d0);

  Layout l0 = Layout(s0, OrderType::ROW_MAJOR, s1, s2);
  Layout l1 = Layout(std::move(l0));

  EXPECT_EQ(l1.getShape().getSize(), 2);
  EXPECT_EQ(l1.getShape().getData<uint64_t>()[0], 2);
  EXPECT_EQ(l1.getShape().getData<uint64_t>()[1], 3);
  EXPECT_EQ(l0.getShape().getData(), nullptr);
}

TEST(LayoutTest, CopyAssignmentTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);

  Storage s0 = makeUint64Storage({5, 6}, &d0);
  Storage s1 = makeUint64Storage({6, 1}, &d0);
  Storage s2 = makeUint64Storage({0, 0}, &d0);

  Layout l0 = Layout(s0, OrderType::ROW_MAJOR, s1, s2);
  Layout l1;
  l1 = l0;

  EXPECT_EQ(l1.getShape().getSize(), 2);
  EXPECT_EQ(l1.getShape().getData<uint64_t>()[0], 5);
  EXPECT_EQ(l1.getShape().getData<uint64_t>()[1], 6);
}

TEST(LayoutTest, MoveAssignmentTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);

  Storage s0 = makeUint64Storage({5, 6}, &d0);
  Storage s1 = makeUint64Storage({6, 1}, &d0);
  Storage s2 = makeUint64Storage({0, 0}, &d0);

  Layout l0 = Layout(s0, OrderType::ROW_MAJOR, s1, s2);
  Layout l1;
  l1 = std::move(l0);

  EXPECT_EQ(l1.getShape().getSize(), 2);
  EXPECT_EQ(l1.getShape().getData<uint64_t>()[0], 5);
  EXPECT_EQ(l1.getShape().getData<uint64_t>()[1], 6);
  EXPECT_EQ(l0.getShape().getData(), nullptr);
}
} // namespace startorch
