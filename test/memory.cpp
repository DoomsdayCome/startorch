#include "startorch/common.hpp"
#include "startorch/device.hpp"
#include <startorch/memory.hpp>

#include <gtest/gtest.h>

namespace startorch {
TEST(ElementTest, DefaultConstructorTest) {
  Element e0 = Element();

  EXPECT_EQ(e0.getData(), nullptr);
  EXPECT_EQ(e0.getDevice(), nullptr);
  EXPECT_EQ(e0.getScalarType(), ScalarType::UNKNOWN_SCALAR);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::UNKNOWN_OWNER);
}

TEST(ElementTest, Float32ReferenceConstructorTest) {
  float f0 = 3.14f;
  Element e0 = Element(&f0, &AMD5625U);

  EXPECT_FLOAT_EQ(*e0.getData<float>(), 3.14f);
  EXPECT_EQ(e0.getScalarType(), ScalarType::FLOAT_32);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::REFERENCE);
}

TEST(ElementTest, Float64ReferenceConstructorTest) {
  double d0 = 2.718281828;
  Element e0 = Element(&d0, &AMD5625U);

  EXPECT_DOUBLE_EQ(*e0.getData<double>(), 2.718281828);
  EXPECT_EQ(e0.getScalarType(), ScalarType::FLOAT_64);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::REFERENCE);
}

TEST(ElementTest, Int8ReferenceConstructorTest) {
  int8_t i0 = -12;
  Element e0 = Element(&i0, &AMD5625U);

  EXPECT_EQ(*e0.getData<int8_t>(), -12);
  EXPECT_EQ(e0.getScalarType(), ScalarType::INT_8);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::REFERENCE);
}

TEST(ElementTest, Int16ReferenceConstructorTest) {
  int16_t i0 = -1000;
  Element e0 = Element(&i0, &AMD5625U);

  EXPECT_EQ(*e0.getData<int16_t>(), -1000);
  EXPECT_EQ(e0.getScalarType(), ScalarType::INT_16);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::REFERENCE);
}

TEST(ElementTest, Int64ReferenceConstructorTest) {
  int64_t i0 = -9000000000LL;
  Element e0 = Element(&i0, &AMD5625U);

  EXPECT_EQ(*e0.getData<int64_t>(), -9000000000LL);
  EXPECT_EQ(e0.getScalarType(), ScalarType::INT_64);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::REFERENCE);
}

TEST(ElementTest, Uint8ReferenceConstructorTest) {
  uint8_t u0 = 255;
  Element e0 = Element(&u0, &AMD5625U);

  EXPECT_EQ(*e0.getData<uint8_t>(), 255);
  EXPECT_EQ(e0.getScalarType(), ScalarType::UNSIGNED_INT_8);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::REFERENCE);
}

TEST(ElementTest, Uint16ReferenceConstructorTest) {
  uint16_t u0 = 65535;
  Element e0 = Element(&u0, &AMD5625U);

  EXPECT_EQ(*e0.getData<uint16_t>(), 65535);
  EXPECT_EQ(e0.getScalarType(), ScalarType::UNSIGNED_INT_16);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::REFERENCE);
}

TEST(ElementTest, Uint32ReferenceConstructorTest) {
  uint32_t u0 = 4294967295U;
  Element e0 = Element(&u0, &AMD5625U);

  EXPECT_EQ(*e0.getData<uint32_t>(), 4294967295U);
  EXPECT_EQ(e0.getScalarType(), ScalarType::UNSIGNED_INT_32);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::REFERENCE);
}

TEST(ElementTest, Uint64ReferenceConstructorTest) {
  uint64_t u0 = 18446744073709551615ULL;
  Element e0 = Element(&u0, &AMD5625U);

  EXPECT_EQ(*e0.getData<uint64_t>(), 18446744073709551615ULL);
  EXPECT_EQ(e0.getScalarType(), ScalarType::UNSIGNED_INT_64);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::REFERENCE);
}

TEST(ElementTest, FloatOwnedConstructorTest) {
  Element e0 = Element(1.5f, &AMD5625U);

  EXPECT_FLOAT_EQ(*e0.getData<float>(), 1.5f);
  EXPECT_EQ(e0.getScalarType(), ScalarType::FLOAT_32);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::OWNED);
}

TEST(ElementTest, DoubleOwnedConstructorTest) {
  Element e0 = Element(3.14159265358979, &AMD5625U);

  EXPECT_DOUBLE_EQ(*e0.getData<double>(), 3.14159265358979);
  EXPECT_EQ(e0.getScalarType(), ScalarType::FLOAT_64);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::OWNED);
}

TEST(ElementTest, Int8OwnedConstructorTest) {
  Element e0 = Element(static_cast<int8_t>(-5), &AMD5625U);

  EXPECT_EQ(*e0.getData<int8_t>(), -5);
  EXPECT_EQ(e0.getScalarType(), ScalarType::INT_8);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::OWNED);
}

TEST(ElementTest, Int16OwnedConstructorTest) {
  Element e0 = Element(static_cast<int16_t>(-500), &AMD5625U);

  EXPECT_EQ(*e0.getData<int16_t>(), -500);
  EXPECT_EQ(e0.getScalarType(), ScalarType::INT_16);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::OWNED);
}

TEST(ElementTest, Int64OwnedConstructorTest) {
  Element e0 = Element(static_cast<int64_t>(-123456789LL), &AMD5625U);

  EXPECT_EQ(*e0.getData<int64_t>(), -123456789LL);
  EXPECT_EQ(e0.getScalarType(), ScalarType::INT_64);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::OWNED);
}

TEST(ElementTest, Uint8OwnedConstructorTest) {
  Element e0 = Element(static_cast<uint8_t>(200), &AMD5625U);

  EXPECT_EQ(*e0.getData<uint8_t>(), 200);
  EXPECT_EQ(e0.getScalarType(), ScalarType::UNSIGNED_INT_8);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::OWNED);
}

TEST(ElementTest, Uint16OwnedConstructorTest) {
  Element e0 = Element(static_cast<uint16_t>(1000), &AMD5625U);

  EXPECT_EQ(*e0.getData<uint16_t>(), 1000);
  EXPECT_EQ(e0.getScalarType(), ScalarType::UNSIGNED_INT_16);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::OWNED);
}

TEST(ElementTest, Uint32OwnedConstructorTest) {
  Element e0 = Element(static_cast<uint32_t>(99999), &AMD5625U);

  EXPECT_EQ(*e0.getData<uint32_t>(), 99999U);
  EXPECT_EQ(e0.getScalarType(), ScalarType::UNSIGNED_INT_32);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::OWNED);
}

TEST(ElementTest, Uint64OwnedConstructorTest) {
  Element e0 = Element(static_cast<uint64_t>(1234567890123ULL), &AMD5625U);

  EXPECT_EQ(*e0.getData<uint64_t>(), 1234567890123ULL);
  EXPECT_EQ(e0.getScalarType(), ScalarType::UNSIGNED_INT_64);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::OWNED);
}

TEST(ElementTest, CopyConstructorTest) {
  int32_t i0 = 42;
  Element e0 = Element(&i0, &AMD5625U);
  Element e1 = Element(e0);

  EXPECT_EQ(*e1.getData<int32_t>(), 42);
  EXPECT_EQ(e1.getDevice(), &AMD5625U);
  EXPECT_EQ(e1.getScalarType(), ScalarType::INT_32);
  EXPECT_EQ(e1.getOwnerType(), OwnerType::REFERENCE);
}

TEST(ElementTest, MoveConstructorTest) {
  int32_t i0 = 55;
  Element e0 = Element(&i0, &AMD5625U);
  Element e1 = Element(std::move(e0));

  EXPECT_EQ(*e1.getData<int32_t>(), 55);
  EXPECT_EQ(e1.getDevice(), &AMD5625U);
  EXPECT_EQ(e1.getScalarType(), ScalarType::INT_32);
  EXPECT_EQ(e1.getOwnerType(), OwnerType::REFERENCE);
  EXPECT_EQ(e0.getData(), nullptr);
}

TEST(ElementTest, CopyAssignmentTest) {
  int32_t i0 = 99;
  Element e0 = Element(&i0, &AMD5625U);
  Element e1;
  e1 = e0;

  EXPECT_EQ(*e1.getData<int32_t>(), 99);
  EXPECT_EQ(e1.getDevice(), &AMD5625U);
  EXPECT_EQ(e1.getScalarType(), ScalarType::INT_32);
  EXPECT_EQ(e1.getOwnerType(), OwnerType::REFERENCE);
}

TEST(ElementTest, MoveAssignmentTest) {
  int32_t i0 = 77;
  Element e0 = Element(&i0, &AMD5625U);
  Element e1;
  e1 = std::move(e0);

  EXPECT_EQ(*e1.getData<int32_t>(), 77);
  EXPECT_EQ(e1.getDevice(), &AMD5625U);
  EXPECT_EQ(e1.getScalarType(), ScalarType::INT_32);
  EXPECT_EQ(e1.getOwnerType(), OwnerType::REFERENCE);
  EXPECT_EQ(e0.getData(), nullptr);
}

TEST(ElementTest, ElementCastTest) {
  int32_t i0 = 123;
  Element e0 = Element(&i0, &AMD5625U);
  Element e1 = element_cast<ScalarType::FLOAT_32>(e0, &AMD5625U);

  EXPECT_EQ(e1.getScalarType(), ScalarType::FLOAT_32);
  EXPECT_FLOAT_EQ(*e1.getData<float>(), 123.0f);
  EXPECT_EQ(e1.getOwnerType(), OwnerType::OWNED);
}

TEST(StorageTest, DefaultConstructorTest) {
  Storage s0 = Storage();

  EXPECT_EQ(s0.getData(), nullptr);
  EXPECT_EQ(s0.getSize(), 0);
  EXPECT_EQ(s0.getScalarType(), ScalarType::UNKNOWN_SCALAR);
  EXPECT_EQ(s0.getDevice(), nullptr);
}

TEST(StorageTest, CustomConstructorTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);
  Storage s0 = Storage(10, ScalarType::INT_32, &d0);

  EXPECT_NE(s0.getData(), nullptr);
  EXPECT_EQ(s0.getSize(), 10);
  EXPECT_EQ(s0.getScalarType(), ScalarType::INT_32);
  EXPECT_EQ(s0.getDevice(), &d0);
}

TEST(StorageTest, CopyConstructorTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);
  Storage s0 = Storage(5, ScalarType::FLOAT_32, &d0);
  Storage s1 = Storage(s0);

  EXPECT_NE(s1.getData(), nullptr);
  EXPECT_EQ(s1.getSize(), 5);
  EXPECT_EQ(s1.getScalarType(), ScalarType::FLOAT_32);
  EXPECT_EQ(s1.getDevice(), &d0);
}

TEST(StorageTest, MoveConstructorTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);
  Storage s0 = Storage(5, ScalarType::FLOAT_32, &d0);
  void *original_ptr = s0.getData();
  Storage s1 = Storage(std::move(s0));

  EXPECT_EQ(s1.getData(), original_ptr);
  EXPECT_EQ(s1.getSize(), 5);
  EXPECT_EQ(s1.getScalarType(), ScalarType::FLOAT_32);
  EXPECT_EQ(s1.getDevice(), &d0);
  EXPECT_EQ(s0.getData(), nullptr);
  EXPECT_EQ(s0.getSize(), 0);
}

TEST(StorageTest, CopyAssignmentTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);
  Storage s0 = Storage(8, ScalarType::INT_64, &d0);
  Storage s1;
  s1 = s0;

  EXPECT_NE(s1.getData(), nullptr);
  EXPECT_EQ(s1.getSize(), 8);
  EXPECT_EQ(s1.getScalarType(), ScalarType::INT_64);
  EXPECT_EQ(s1.getDevice(), &d0);
}

TEST(StorageTest, MoveAssignmentTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);
  Storage s0 = Storage(8, ScalarType::INT_64, &d0);
  void *original_ptr = s0.getData();
  Storage s1;
  s1 = std::move(s0);

  EXPECT_EQ(s1.getData(), original_ptr);
  EXPECT_EQ(s1.getSize(), 8);
  EXPECT_EQ(s1.getScalarType(), ScalarType::INT_64);
  EXPECT_EQ(s1.getDevice(), &d0);
  EXPECT_EQ(s0.getData(), nullptr);
  EXPECT_EQ(s0.getSize(), 0);
}

TEST(StorageTest, SubscriptOperatorTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);
  Storage s0 = Storage(5, ScalarType::INT_32, &d0);
  s0.getData<int32_t>()[2] = 42;
  Element e0 = s0[2];

  EXPECT_EQ(*e0.getData<int32_t>(), 42);
  EXPECT_EQ(e0.getScalarType(), ScalarType::INT_32);
}

TEST(StorageTest, ConstantSubscriptOperatorTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);
  Storage s0 = Storage(5, ScalarType::INT_32, &d0);
  s0.getData<int32_t>()[3] = 99;
  const Storage &s_ref = s0;
  const Element e0 = s_ref[3];

  EXPECT_EQ(*e0.getData<int32_t>(), 99);
  EXPECT_EQ(e0.getScalarType(), ScalarType::INT_32);
}

TEST(StorageTest, FillDataTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);
  Storage s0 = Storage(5, ScalarType::INT_32, &d0);
  Element fill_val = Element(static_cast<int32_t>(7), &d0);

  s0.fillData(fill_val);

  for (uint64_t i = 0; i < s0.getSize(); ++i)
    EXPECT_EQ(*s0[i].getData<int32_t>(), 7);
}

TEST(StorageTest, FillIncreasedDataTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);
  Storage s0 = Storage(5, ScalarType::INT_32, &d0);
  Element start = Element(static_cast<int32_t>(0), &d0);
  Element step = Element(static_cast<int32_t>(2), &d0);

  s0.fillIncreasedData(start, step);

  for (uint64_t i = 0; i < s0.getSize(); ++i)
    EXPECT_EQ(*s0[i].getData<int32_t>(), static_cast<int32_t>(i * 2));
}

TEST(StorageTest, FillDecreasedDataTest) {
  Device d0 = Device(1_KiB, MemoryType::HOST, DeviceType::CPU);
  Storage s0 = Storage(5, ScalarType::INT_32, &d0);
  Element start = Element(static_cast<int32_t>(8), &d0);
  Element step = Element(static_cast<int32_t>(2), &d0);

  s0.fillDecreasedData(start, step);

  for (uint64_t i = 0; i < s0.getSize(); ++i)
    EXPECT_EQ(*s0[i].getData<int32_t>(), static_cast<int32_t>(8 - i * 2));
}
} // namespace startorch
