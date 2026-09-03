    #else
        #define AUTOVEC
    #endif
#elif defined(__aarch64__) || defined(__ARM_NEON)
    #define USE_NEON
    #include <arm_neon.h>

    using nativeVector = int32x4_t;
    using vepi8  = int8x16_t;
    using vepi16 = int16x8_t;
    using vepi32 = int32x4_t;
    using vps32  = float32x4_t;

    inline vepi16 set1_epi16(int16_t v) { return vdupq_n_s16(v); }
    inline vepi16 load_epi16(const vepi16* p) { return vld1q_s16((const int16_t*)p); }
    inline void   store_epi16(vepi16* p, vepi16 v) { vst1q_s16((int16_t*)p, v); }

    inline vepi16 min_epi16(vepi16 a, vepi16 b) { return vminq_s16(a, b); }
    inline vepi16 max_epi16(vepi16 a, vepi16 b) { return vmaxq_s16(a, b); }

    inline vepi16 mullo_epi16(vepi16 a, vepi16 b) { return vmulq_s16(a, b); }
    inline vepi16 add_epi16(vepi16 a, vepi16 b) { return vaddq_s16(a, b); }
    inline vepi16 sub_epi16(vepi16 a, vepi16 b) { return vsubq_s16(a, b); }

    inline vepi32 set1_epi32(int32_t v) { return vdupq_n_s32(v); }
    inline vepi32 add_epi32(vepi32 a, vepi32 b) { return vaddq_s32(a, b); }

    inline int reduce_epi32(vepi32 v) { return vaddvq_s32(v); }

    inline vps32 cvtepi32_ps(vepi32 v) { return vcvtq_f32_s32(v); }
    inline vps32 load_ps(const float* p) { return vld1q_f32(p); }
    inline void  store_ps(float* p, vps32 v) { vst1q_f32(p, v); }

    inline vps32 set1_ps(float v) { return vdupq_n_f32(v); }
    inline vps32 zero_ps() { return vdupq_n_f32(0.0f); }
    inline vps32 mul_ps(vps32 a, vps32 b) { return vmulq_f32(a, b); }
    inline vps32 mul_add_ps(vps32 a, vps32 b, vps32 c) { return vfmaq_f32(c, a, b); }

    // Fast NNUE Dot Product using ARM NEON DotProd / Multiply-Add
    inline vepi32 dpbusd_epi32(const vepi32 sum, const vepi8 vec0, const vepi8 vec1) {
#if defined(__ARM_FEATURE_DOTPROD)
        return vdotq_s32(sum, vec0, vec1);
#else
        int16x8_t low = vmull_s8(vget_low_s8(vec0), vget_low_s8(vec1));
        int16x8_t high = vmull_s8(vget_high_s8(vec0), vget_high_s8(vec1));
        int32x4_t p1 = vpaddlq_s16(low);
        int32x4_t p2 = vpaddlq_s16(high);
        return vaddq_s32(sum, vaddq_s32(p1, p2));
#endif
    }
#else
    #define AUTOVEC
#endif
