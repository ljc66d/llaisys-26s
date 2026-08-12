#include "types.hpp"
#include <cstring>

namespace llaisys::utils {

float _f16_to_f32(fp16_t val) {
    uint16_t h = val._v;
    uint32_t sign = (h & 0x8000) << 16;
    int exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f32;

    if (exp == 0x1F) {
        // Inf / NaN
        f32 = sign | 0x7F800000 | (mant << 13);
    } else if (exp == 0) {
        if (mant == 0) {
            // ±0
            f32 = sign;
        } else {
            // 非规格化数转规格化
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            exp += 127 - 15 + 1;
            f32 = sign | (static_cast<uint32_t>(exp) << 23) | (mant << 13);
        }
    } else {
        // 规格化数：指数偏置转换 15 → 127
        f32 = sign | (static_cast<uint32_t>(exp + 127 - 15) << 23) | (mant << 13);
    }

    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

fp16_t _f32_to_f16(float val) {
    uint32_t f;
    memcpy(&f, &val, sizeof(f));
    uint16_t sign = (f >> 16) & 0x8000;
    int exp = ((f >> 23) & 0xFF) - 127;
    uint32_t mant = f & 0x7FFFFF;

    uint16_t h;

    if (exp >= 16) {
        // 上溢：返回无穷大
        h = sign | 0x7C00;
    } else if (exp <= -15) {
        // 下溢：转非规格化或0
        mant |= 0x800000; // 补回隐含前导1
        int shift = -14 - exp;
        if (shift > 24) {
            shift = 24;
        }
        mant >>= shift;
        // 四舍五入
        mant += 0x1000;
        if (mant & 0x800000) {
            mant = 0;
            exp++;
        }
        h = sign | static_cast<uint16_t>(mant >> 13);
    } else {
        // 规格化数
        mant += 0x1000; // 四舍五入
        if (mant & 0x800000) {
            mant = 0;
            exp++;
        }
        h = sign | static_cast<uint16_t>((exp + 15) << 10) | static_cast<uint16_t>(mant >> 13);
    }

    return {h};
}

float _bf16_to_f32(bf16_t val) {
    uint32_t bits32 = static_cast<uint32_t>(val._v) << 16;
    float out;
    std::memcpy(&out, &bits32, sizeof(out));
    return out;
}

bf16_t _f32_to_bf16(float val) {
    uint32_t bits32;
    std::memcpy(&bits32, &val, sizeof(bits32));

    const uint32_t rounding_bias = 0x00007FFF + ((bits32 >> 16) & 1);
    uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);

    return bf16_t{bf16_bits};
}

} // namespace llaisys::utils