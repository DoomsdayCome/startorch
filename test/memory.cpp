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

TEST(ElementTest, CustomReferenceConstructorTest) {
  int32_t i0 = 67;

  Element e0 = Element(&i0, &AMD5625U);

  EXPECT_EQ(*e0.getData<int32_t>(), 67);
  EXPECT_EQ(e0.getDevice(), &AMD5625U);
  EXPECT_EQ(e0.getScalarType(), ScalarType::INT_32);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::REFERENCE);
}

TEST(ElementTest, CustomOwnedConstructorTest) {
  int32_t i0 = 67;

  Element e0 = Element(i0, &AMD5625U);

  EXPECT_EQ(*e0.getData<int32_t>(), 67);
  EXPECT_EQ(e0.getDevice(), &AMD5625U);
  EXPECT_EQ(e0.getScalarType(), ScalarType::INT_32);
  EXPECT_EQ(e0.getOwnerType(), OwnerType::OWNED);
}
} // namespace startorch
