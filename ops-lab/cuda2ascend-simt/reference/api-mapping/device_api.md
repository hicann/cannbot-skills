# Device API Mapping

说明：
- 分类依据：华为昇腾 CANN Community Edition 9.0.0-beta.2 `SIMT API` 分类。
- 本文件章节名使用 `device_api.yaml` 中 `category` 的子类名，不重复显示 `device/` 前缀。
- 对应的 YAML `category` 形式为：`device/<subcategory-kebab-case>`。
- 参考文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_0427.html

## kernel-definition

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| threadIdx | threadIdx |  |
| blockDim | blockDim |  |
| blockIdx | blockIdx |  |

## synchronization

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| __syncthreads | asc_syncthreads | void asc_syncthreads() |
| __syncthreads_and |  |  |
| __syncthreads_count |  |  |
| __syncthreads_or |  |  |
| __threadfence | asc_threadfence | void asc_threadfence() |
| __threadfence_block | asc_threadfence_block | void asc_threadfence_block() |
| __threadfence_system |  |  |

## math

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| _Pow_int |  |  |
| __brev |  |  |
| __brevll |  |  |
| __byte_perm |  |  |
| __clz |  |  |
| __clzll |  |  |
| __ffs |  |  |
| __ffsll |  |  |
| __funnelshift_l |  |  |
| __funnelshift_lc |  |  |
| __funnelshift_r |  |  |
| __funnelshift_rc |  |  |
| __nv_bswap16 |  |  |
| __nv_bswap32 |  |  |
| __nv_bswap64 |  |  |
| __popc |  |  |
| __popcll |  |  |
| __cosf |  |  |
| __dadd_rd |  |  |
| __dadd_rn |  |  |
| __dadd_ru |  |  |
| __dadd_rz |  |  |
| __ddiv_rd |  |  |
| __ddiv_rn |  |  |
| __ddiv_ru |  |  |
| __ddiv_rz |  |  |
| __dmul_rd |  |  |
| __dmul_rn |  |  |
| __dmul_ru |  |  |
| __dmul_rz |  |  |
| __drcp_rd |  |  |
| __drcp_rn |  |  |
| __drcp_ru |  |  |
| __drcp_rz |  |  |
| __dsqrt_rd |  |  |
| __dsqrt_rn |  |  |
| __dsqrt_ru |  |  |
| __dsqrt_rz |  |  |
| __dsub_rd |  |  |
| __dsub_rn |  |  |
| __dsub_ru |  |  |
| __dsub_rz |  |  |
| __exp10f |  |  |
| __expf |  |  |
| __fadd_rd |  |  |
| __fadd_rn |  |  |
| __fadd_ru |  |  |
| __fadd_rz |  |  |
| __fdiv_rd |  |  |
| __fdiv_rn |  |  |
| __fdiv_ru |  |  |
| __fdiv_rz |  |  |
| __fdividef |  |  |
| __fma_rd |  |  |
| __fma_rn |  |  |
| __fma_ru |  |  |
| __fma_rz |  |  |
| __fmaf_rd |  |  |
| __fmaf_rn |  |  |
| __fmaf_ru |  |  |
| __fmaf_rz |  |  |
| __fmul_rd |  |  |
| __fmul_rn |  |  |
| __fmul_ru |  |  |
| __fmul_rz |  |  |
| __frcp_rd |  |  |
| __frcp_rn |  |  |
| __frcp_ru |  |  |
| __frcp_rz |  |  |
| __frsqrt_rn |  |  |
| __fsqrt_rd |  |  |
| __fsqrt_rn |  |  |
| __fsqrt_ru |  |  |
| __fsqrt_rz |  |  |
| __fsub_rd |  |  |
| __fsub_rn |  |  |
| __fsub_ru |  |  |
| __fsub_rz |  |  |
| __h2div |  |  |
| __habs | __habs | bfloat16_t __habs(bfloat16_t x) |
| __habs2 |  |  |
| __hadd |  |  |
| __hadd2 |  |  |
| __hadd2_rn |  |  |
| __hadd2_sat |  |  |
| __hadd_rn |  |  |
| __hadd_sat |  |  |
| __hcmadd |  |  |
| __hdiv |  |  |
| __hfma | __hfma | bfloat16_t __hfma(bfloat16_t x, bfloat16_t y, bfloat16_t z) |
| __hfma2 |  |  |
| __hfma2_relu |  |  |
| __hfma2_sat |  |  |
| __hfma_relu |  |  |
| __hfma_sat |  |  |
| __hmax | __hmax | bfloat16_t __hmax(bfloat16_t x, bfloat16_t y) |
| __hmax2 |  |  |
| __hmax2_nan |  |  |
| __hmax_nan |  |  |
| __hmin | __hmin | bfloat16_t __hmin(bfloat16_t x, bfloat16_t y) |
| __hmin2 |  |  |
| __hmin2_nan |  |  |
| __hmin_nan |  |  |
| __hmul |  |  |
| __hmul2 |  |  |
| __hmul2_rn |  |  |
| __hmul2_sat |  |  |
| __hmul_rn |  |  |
| __hmul_sat |  |  |
| __hneg |  |  |
| __hneg2 |  |  |
| __hsub |  |  |
| __hsub2 |  |  |
| __hsub2_rn |  |  |
| __hsub2_sat |  |  |
| __hsub_rn |  |  |
| __hsub_sat |  |  |
| __log10f |  |  |
| __log2f |  |  |
| __logf |  |  |
| __mul24 |  |  |
| __mul64hi |  |  |
| __mulhi |  |  |
| __nv_fp128_acos |  |  |
| __nv_fp128_acosh |  |  |
| __nv_fp128_add |  |  |
| __nv_fp128_asin |  |  |
| __nv_fp128_asinh |  |  |
| __nv_fp128_atan |  |  |
| __nv_fp128_atanh |  |  |
| __nv_fp128_ceil |  |  |
| __nv_fp128_copysign |  |  |
| __nv_fp128_cos |  |  |
| __nv_fp128_cosh |  |  |
| __nv_fp128_div |  |  |
| __nv_fp128_exp |  |  |
| __nv_fp128_exp10 |  |  |
| __nv_fp128_exp2 |  |  |
| __nv_fp128_expm1 |  |  |
| __nv_fp128_fabs |  |  |
| __nv_fp128_fdim |  |  |
| __nv_fp128_floor |  |  |
| __nv_fp128_fma |  |  |
| __nv_fp128_fmax |  |  |
| __nv_fp128_fmin |  |  |
| __nv_fp128_fmod |  |  |
| __nv_fp128_frexp |  |  |
| __nv_fp128_hypot |  |  |
| __nv_fp128_ilogb |  |  |
| __nv_fp128_isnan |  |  |
| __nv_fp128_isunordered |  |  |
| __nv_fp128_ldexp |  |  |
| __nv_fp128_log |  |  |
| __nv_fp128_log10 |  |  |
| __nv_fp128_log1p |  |  |
| __nv_fp128_log2 |  |  |
| __nv_fp128_modf |  |  |
| __nv_fp128_mul |  |  |
| __nv_fp128_pow |  |  |
| __nv_fp128_remainder |  |  |
| __nv_fp128_rint |  |  |
| __nv_fp128_round |  |  |
| __nv_fp128_sin |  |  |
| __nv_fp128_sinh |  |  |
| __nv_fp128_sqrt |  |  |
| __nv_fp128_sub |  |  |
| __nv_fp128_tan |  |  |
| __nv_fp128_tanh |  |  |
| __nv_fp128_trunc |  |  |
| __powf |  |  |
| __rhadd |  |  |
| __sad |  |  |
| __saturatef |  |  |
| __sincosf |  |  |
| __sinf |  |  |
| __tanf |  |  |
| __uhadd |  |  |
| __umul24 |  |  |
| __umul64hi |  |  |
| __umulhi |  |  |
| __urhadd |  |  |
| __usad |  |  |
| __vabs2 |  |  |
| __vabs4 |  |  |
| __vabsdiffs2 |  |  |
| __vabsdiffs4 |  |  |
| __vabsdiffu2 |  |  |
| __vabsdiffu4 |  |  |
| __vabsss2 |  |  |
| __vabsss4 |  |  |
| __vadd2 |  |  |
| __vadd4 |  |  |
| __vaddss2 |  |  |
| __vaddss4 |  |  |
| __vaddus2 |  |  |
| __vaddus4 |  |  |
| __vavgs2 |  |  |
| __vavgs4 |  |  |
| __vavgu2 |  |  |
| __vavgu4 |  |  |
| __vhaddu2 |  |  |
| __vhaddu4 |  |  |
| __vmaxs2 |  |  |
| __vmaxs4 |  |  |
| __vmaxu2 |  |  |
| __vmaxu4 |  |  |
| __vmins2 |  |  |
| __vmins4 |  |  |
| __vminu2 |  |  |
| __vminu4 |  |  |
| __vneg2 |  |  |
| __vneg4 |  |  |
| __vnegss2 |  |  |
| __vnegss4 |  |  |
| __vsads2 |  |  |
| __vsads4 |  |  |
| __vsadu2 |  |  |
| __vsadu4 |  |  |
| __vsub2 |  |  |
| __vsub4 |  |  |
| __vsubss2 |  |  |
| __vsubss4 |  |  |
| __vsubus2 |  |  |
| __vsubus4 |  |  |
| _fdsign |  |  |
| _ldsign |  |  |
| abs | abs |  |
| acos |  |  |
| acosf | acosf | float acosf(float x) |
| acosh |  |  |
| acoshf | acoshf | float acoshf(float x) |
| asin |  |  |
| asinf | asinf | float asinf(float x) |
| asinh |  |  |
| asinhf | asinhf | float asinhf(float x) |
| atan |  |  |
| atan2 |  |  |
| atan2f | atan2f | float atan2f(float y, float x) |
| atanf | atanf | float atanf(float x) |
| atanh |  |  |
| atanhf | atanhf | float atanhf(float x) |
| cbrt |  |  |
| cbrtf | cbrtf | float cbrtf(float x) |
| ceil |  |  |
| copysign |  |  |
| copysignf | copysignf | float copysignf(float x, float y) |
| cos |  |  |
| cosf | cosf | float cosf(float x) |
| cosh |  |  |
| coshf | coshf | float coshf(float x) |
| cospi |  |  |
| cospif | cospif | float cospif(float x) |
| cyl_bessel_i0 |  |  |
| cyl_bessel_i0f | cyl_bessel_i0f | float cyl_bessel_i0f(float x) |
| cyl_bessel_i1 |  |  |
| cyl_bessel_i1f | cyl_bessel_i1f | float cyl_bessel_i1f(float x) |
| erf |  |  |
| erfc |  |  |
| erfcf | erfcf | float erfcf(float x) |
| erfcinv |  |  |
| erfcinvf | erfcinvf | float erfcinvf(float x) |
| erfcx |  |  |
| erfcxf | erfcxf | float erfcxf(float x) |
| erff | erff | float erff(float x) |
| erfinv |  |  |
| erfinvf | erfinvf | float erfinvf(float x) |
| exp |  |  |
| exp10 |  |  |
| exp10f | exp10f | float exp10f(float x) |
| exp2 |  |  |
| exp2f | exp2f | float exp2f(float x) |
| expf | expf | float expf(float x) |
| expm1 |  |  |
| expm1f | expm1f | float expm1f(float x) |
| fabs |  |  |
| fabsf | fabsf | float fabsf(float x) |
| fdim |  |  |
| fdimf | fdimf | float fdimf(float x, float y) |
| fdivide |  |  |
| fdividef | fdividef | float fdividef(float x, float y) |
| floor |  |  |
| fma |  |  |
| fmaf | fmaf | float fmaf(float x, float y, float z) |
| fmax |  |  |
| fmaxf | fmaxf | float fmaxf(float x, float y) |
| fmin |  |  |
| fminf | fminf | float fminf(float x, float y) |
| fmod |  |  |
| fmodf | fmodf | float fmodf(float x, float y) |
| frexp |  |  |
| frexpf | frexpf | float frexpf(float x, __gm__ int *exp) |
| h2cos | h2cos | bfloat16x2_t h2cos(bfloat16x2_t x) |
| h2exp | h2exp | bfloat16x2_t h2exp(bfloat16x2_t x) |
| h2exp10 | h2exp10 | bfloat16x2_t h2exp10(bfloat16x2_t x) |
| h2exp2 | h2exp2 | bfloat16x2_t h2exp2(bfloat16x2_t x) |
| h2log | h2log | bfloat16x2_t h2log(bfloat16x2_t x) |
| h2log10 | h2log10 | bfloat16x2_t h2log10(bfloat16x2_t x) |
| h2log2 | h2log2 | bfloat16x2_t h2log2(bfloat16x2_t x) |
| h2rcp | h2rcp | bfloat16x2_t h2rcp(bfloat16x2_t x) |
| h2rsqrt | h2rsqrt | bfloat16x2_t h2rsqrt(bfloat16x2_t x) |
| h2sin | h2sin | bfloat16x2_t h2sin(bfloat16x2_t x) |
| h2sqrt | h2sqrt | bfloat16x2_t h2sqrt(bfloat16x2_t x) |
| h2tanh | h2tanh | bfloat16x2_t h2tanh(bfloat16x2_t x) |
| h2tanh_approx |  |  |
| hcos | hcos | bfloat16_t hcos(bfloat16_t x) |
| hexp | hexp | bfloat16_t hexp(bfloat16_t x) |
| hexp10 | hexp10 | bfloat16_t hexp10(bfloat16_t x) |
| hexp2 | hexp2 | bfloat16_t hexp2(bfloat16_t x) |
| hlog | hlog | bfloat16_t hlog(bfloat16_t x) |
| hlog10 | hlog10 | bfloat16_t hlog10(bfloat16_t x) |
| hlog2 | hlog2 | bfloat16_t hlog2(bfloat16_t x) |
| hrcp | hrcp | bfloat16_t hrcp(bfloat16_t x) |
| hrsqrt | hrsqrt | bfloat16_t hrsqrt(bfloat16_t x) |
| hsin | hsin | bfloat16_t hsin(bfloat16_t x) |
| hsqrt | hsqrt | bfloat16_t hsqrt(bfloat16_t x) |
| htanh | htanh | bfloat16_t htanh(bfloat16_t x) |
| htanh_approx |  |  |
| hypot |  |  |
| hypotf | hypotf | float hypotf(float x, float y) |
| ilogb |  |  |
| ilogbf | ilogbf | int ilogbf(float x) |
| j0 |  |  |
| j0f | j0f | float j0f(float x) |
| j1 |  |  |
| j1f | j1f | float j1f(float x) |
| jn |  |  |
| jnf | jnf | float jnf(int n, float x) |
| labs | labs | long int labs(long int x) |
| ldexp |  |  |
| ldexpf | ldexpf | float ldexpf(float x, int exp) |
| lgamma |  |  |
| lgammaf | lgammaf | float lgammaf(float x) |
| llabs | llabs | long long int llabs(long long int x) |
| llmax | llmax | long long int llmax(const long long int x, const long long int y) |
| llmin | llmin | long long int llmin(const long long int x, const long long int y) |
| llrint |  |  |
| llround |  |  |
| log |  |  |
| log10 |  |  |
| log10f | log10f | float log10f(float x) |
| log1p |  |  |
| log1pf | log1pf | float log1pf(float x) |
| log2 |  |  |
| log2f | log2f | float log2f(float x) |
| logb |  |  |
| logbf | logbf | float logbf(float x) |
| logf | logf | float logf(float x) |
| lrint |  |  |
| lround |  |  |
| max | max |  |
| min | min |  |
| modf |  |  |
| modff | modff | float modff(float x, __gm__ float *n) |
| mul24 |  |  |
| mul64hi |  |  |
| mulhi |  |  |
| nan |  |  |
| nanf |  |  |
| nearbyint |  |  |
| nextafter |  |  |
| nextafterf | nextafterf | float nextafterf(float x, float y) |
| norm |  |  |
| norm3d |  |  |
| norm3df | norm3df | float norm3df(float a, float b, float c) |
| norm4d |  |  |
| norm4df | norm4df | float norm4df(float a, float b, float c, float d) |
| normcdf |  |  |
| normcdff | normcdff | float normcdff(float x) |
| normcdfinv |  |  |
| normcdfinvf | normcdfinvf | float normcdfinvf(float x) |
| normf | normf | float normf(int n, __gm__ float* a) |
| pow |  |  |
| powf | powf | float powf(float x, float y) |
| rcbrt |  |  |
| rcbrtf | rcbrtf | float rcbrtf(float x) |
| remainder |  |  |
| remainderf | remainderf | float remainderf(float x, float y) |
| remquo |  |  |
| remquof | remquof | float remquof(float x, float y, __gm__ int *quo) |
| rhypot |  |  |
| rhypotf | rhypotf | float rhypotf(float x, float y) |
| rint |  |  |
| rnorm |  |  |
| rnorm3d |  |  |
| rnorm3df | rnorm3df | float rnorm3df(float a, float b, float c) |
| rnorm4d |  |  |
| rnorm4df | rnorm4df | float rnorm4df(float a, float b, float c, float d) |
| rnormf | rnormf | float rnormf(int n, __gm__ float* a) |
| round |  |  |
| rsqrt |  |  |
| rsqrtf | rsqrtf | float rsqrtf(float x) |
| saturate |  |  |
| scalbln |  |  |
| scalblnf | scalblnf | float scalblnf(float x, int64_t n) |
| scalbn |  |  |
| scalbnf | scalbnf | float scalbnf(float x, int32_t n) |
| sin |  |  |
| sincos |  |  |
| sincosf | sincosf | void sincosf(float x, __gm__ float *s, __gm__  float *c) |
| sincospi |  |  |
| sincospif | sincospif | void sincospif(float x, __gm__ float *s, __gm__ float *c) |
| sinf | sinf | float sinf(float x) |
| sinh |  |  |
| sinhf | sinhf | float sinhf(float x) |
| sinpi |  |  |
| sinpif | sinpif | float sinpif(float x) |
| sqrt | sqrt |  |
| sqrtf | sqrtf | float sqrtf(float x) |
| tan |  |  |
| tanf | tanf | float tanf(float x) |
| tanh |  |  |
| tanhf | tanhf | float tanhf(float x) |
| tgamma |  |  |
| tgammaf | tgammaf | float tgammaf(float x) |
| trunc |  |  |
| ullmax | ullmax | unsigned long long int ullmax(const unsigned long long int x, const unsigned long long int y) |
| ullmin | ullmin | unsigned long long int ullmin(const unsigned long long int x, const unsigned long long int y) |
| umax | umax | unsigned int umax(const unsigned int x, const unsigned int y) |
| umin | umin | unsigned int umin(const unsigned int x, const unsigned int y) |
| umul24 |  |  |
| y0 |  |  |
| y0f | y0f | float y0f(float x) |
| y1 |  |  |
| y1f | y1f | float y1f(float x) |
| yn |  |  |
| ynf | ynf | float ynf(int n, float x) |

## precision-conversion

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| ceilf | ceilf | float ceilf(float x) |
| floorf | floorf | float floorf(float x) |
| h2ceil | h2ceil | bfloat16x2_t h2ceil(bfloat16x2_t x) |
| h2floor | h2floor | bfloat16x2_t h2floor(bfloat16x2_t x) |
| h2rint | h2rint | bfloat16x2_t h2rint(bfloat16x2_t x) |
| h2trunc | h2trunc | bfloat16x2_t h2trunc(bfloat16x2_t x) |
| hceil | hceil | bfloat16_t hceil(bfloat16_t x) |
| hfloor | hfloor | bfloat16_t hfloor(bfloat16_t x) |
| hrint | hrint | bfloat16_t hrint(bfloat16_t x) |
| htrunc | htrunc | bfloat16_t htrunc(bfloat16_t x) |
| llrintf | llrintf | long long int llrintf(float x) |
| llroundf | llroundf | long long int llroundf(float x) |
| lrintf | lrintf | long int lrintf(float x) |
| lroundf | lroundf | long int lroundf(float x) |
| nearbyintf | nearbyintf | float nearbyintf(float x) |
| rintf | rintf | float rintf(float x) |
| roundf | roundf | float roundf(float x) |
| truncf | truncf | float truncf(float x) |

## comparison

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| __finite |  |  |
| __finitef |  |  |
| __finitel |  |  |
| __hbeq2 |  |  |
| __hbequ2 |  |  |
| __hbge2 |  |  |
| __hbgeu2 |  |  |
| __hbgt2 |  |  |
| __hbgtu2 |  |  |
| __hble2 |  |  |
| __hbleu2 |  |  |
| __hblt2 |  |  |
| __hbltu2 |  |  |
| __hbne2 |  |  |
| __hbneu2 |  |  |
| __heq |  |  |
| __heq2 |  |  |
| __heq2_mask |  |  |
| __hequ |  |  |
| __hequ2 |  |  |
| __hequ2_mask |  |  |
| __hge |  |  |
| __hge2 |  |  |
| __hge2_mask |  |  |
| __hgeu |  |  |
| __hgeu2 |  |  |
| __hgeu2_mask |  |  |
| __hgt |  |  |
| __hgt2 |  |  |
| __hgt2_mask |  |  |
| __hgtu |  |  |
| __hgtu2 |  |  |
| __hgtu2_mask |  |  |
| __hisinf | __hisinf | bool __hisinf(bfloat16_t x) |
| __hisnan | __hisnan | bool __hisnan(bfloat16_t x) |
| __hisnan2 |  |  |
| __hle |  |  |
| __hle2 |  |  |
| __hle2_mask |  |  |
| __hleu |  |  |
| __hleu2 |  |  |
| __hleu2_mask |  |  |
| __hlt |  |  |
| __hlt2 |  |  |
| __hlt2_mask |  |  |
| __hltu |  |  |
| __hltu2 |  |  |
| __hltu2_mask |  |  |
| __hne |  |  |
| __hne2 |  |  |
| __hne2_mask |  |  |
| __hneu |  |  |
| __hneu2 |  |  |
| __hneu2_mask |  |  |
| __isinf |  |  |
| __isinff |  |  |
| __isinfl |  |  |
| __isnan |  |  |
| __isnanf |  |  |
| __isnanl |  |  |
| __signbit |  |  |
| __signbitf |  |  |
| __signbitl |  |  |
| __vcmpeq2 |  |  |
| __vcmpeq4 |  |  |
| __vcmpges2 |  |  |
| __vcmpges4 |  |  |
| __vcmpgeu2 |  |  |
| __vcmpgeu4 |  |  |
| __vcmpgts2 |  |  |
| __vcmpgts4 |  |  |
| __vcmpgtu2 |  |  |
| __vcmpgtu4 |  |  |
| __vcmples2 |  |  |
| __vcmples4 |  |  |
| __vcmpleu4 |  |  |
| __vcmplts2 |  |  |
| __vcmplts4 |  |  |
| __vcmpltu2 |  |  |
| __vcmpltu4 |  |  |
| __vcmpne2 |  |  |
| __vcmpne4 |  |  |
| __vseteq2 |  |  |
| __vseteq4 |  |  |
| __vsetges2 |  |  |
| __vsetges4 |  |  |
| __vsetgeu2 |  |  |
| __vsetgeu4 |  |  |
| __vsetgts2 |  |  |
| __vsetgts4 |  |  |
| __vsetgtu4 |  |  |
| __vsetles2 |  |  |
| __vsetles4 |  |  |
| __vsetleu2 |  |  |
| __vsetleu4 |  |  |
| __vsetlts2 |  |  |
| __vsetlts4 |  |  |
| __vsetltu2 |  |  |
| __vsetltu4 |  |  |
| __vsetne2 |  |  |
| __vsetne4 |  |  |
| isfinite | isfinite | bool isfinite(float x) |
| isinf | isinf | bool isinf(float x) |
| isnan | isnan | bool isnan(float x) |
| signbit | signbit | int signbit(float x) |

## atomic

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| atomicAdd | asc_atomic_add |  |
| atomicAdd_system |  |  |
| atomicAnd | asc_atomic_and | inline int32_t asc_atomic_and(int32_t *address, int32_t val) |
| atomicAnd_system |  |  |
| atomicCAS | asc_atomic_cas | inline int32_t asc_atomic_cas(int32_t *address, int32_t compare, int32_t val) |
| atomicCAS_system |  |  |
| atomicDec | asc_atomic_dec | inline uint32_t asc_atomic_dec(uint32_t *address, uint32_t val) |
| atomicExch | asc_atomic_exch | inline int32_t asc_atomic_exch(int32_t *address, int32_t val) |
| atomicExch_system |  |  |
| atomicInc | asc_atomic_inc | inline uint32_t asc_atomic_inc(uint32_t *address, uint32_t val) |
| atomicMax | asc_atomic_max | inline float asc_atomic_max(float *address, float val) |
| atomicMax_system |  |  |
| atomicMin | asc_atomic_min | inline float asc_atomic_min(float *address, float val) |
| atomicMin_system |  |  |
| atomicOr | asc_atomic_or | inline int32_t asc_atomic_or(int32_t *address, int32_t val) |
| atomicOr_system |  |  |
| atomicSub | asc_atomic_sub | inline float asc_atomic_sub(float *address, float val) |
| atomicSub_system |  |  |
| atomicXor | asc_atomic_xor | inline int32_t asc_atomic_xor(int32_t *address, int32_t val) |
| atomicXor_system |  |  |

## warp

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| __activemask | asc_activemask | uint32_t asc_activemask() |
| __all | asc_all | int32_t asc_all(int32_t predicate) |
| __all_sync | asc_all | int32_t asc_all(int32_t predicate) |
| __any | asc_any | int32_t asc_any(int32_t predicate) |
| __any_sync | asc_any | int32_t asc_any(int32_t predicate) |
| __ballot | asc_ballot | uint32_t asc_ballot(int32_t predicate) |
| __ballot_sync | asc_ballot | uint32_t asc_ballot(int32_t predicate) |
| __match_all_sync |  |  |
| __match_any_sync |  |  |
| __reduce_add_sync | asc_reduce_add | inline int32_t asc_reduce_add(int32_t val) |
| __reduce_max_sync | asc_reduce_max | inline int32_t asc_reduce_max(int32_t val) |
| __reduce_min_sync | asc_reduce_min | inline int32_t asc_reduce_min(int32_t val) |
| __shfl | asc_shfl | half2 asc_shfl(half2 var, int32_t src_lane, int32_t width = warpSize) |
| __shfl_down | asc_shfl_down | half2 asc_shfl_down(half2 var, uint32_t delta, int32_t width = warpSize) |
| __shfl_down_sync | asc_shfl_down | inline int32_t asc_shfl_down(int32_t var, uint32_t delta, int32_t width = warpSize) |
| __shfl_sync | asc_shfl | inline int32_t asc_shfl(int32_t var, int32_t src_lane, int32_t width = warpSize) |
| __shfl_up | asc_shfl_up | half2 asc_shfl_up(half2 var, uint32_t delta, int32_t width = warpSize) |
| __shfl_up_sync | asc_shfl_up | inline int32_t asc_shfl_up(int32_t var, uint32_t delta, int32_t width = warpSize) |
| __shfl_xor | asc_shfl_xor | half2 asc_shfl_xor(half2 var, int32_t lane_mask, int32_t width = warpSize) |
| __shfl_xor_sync | asc_shfl_xor | inline int32_t asc_shfl_xor(int32_t var, int32_t lane_mask, int32_t width = warpSize) |

## type-conversion

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| __bfloat1622float2 | __bfloat1622float2 | float2 __bfloat1622float2(const bfloat16x2_t x) |
| __bfloat162bfloat162 | __bfloat162bfloat162 | bfloat16x2_t __bfloat162bfloat162(const bfloat16_t x) |
| __bfloat162char_rz |  |  |
| __bfloat162float | __bfloat162float | float __bfloat162float(const bfloat16_t x) |
| __bfloat162int_rd | __bfloat162int_rd | int __bfloat162int_rd(const bfloat16_t x) |
| __bfloat162int_rn | __bfloat162int_rn | int __bfloat162int_rn(const bfloat16_t x) |
| __bfloat162int_ru | __bfloat162int_ru | int __bfloat162int_ru(const bfloat16_t x) |
| __bfloat162int_rz | __bfloat162int_rz | int __bfloat162int_rz(const bfloat16_t x) |
| __bfloat162ll_rd | __bfloat162ll_rd | long long int __bfloat162ll_rd(const bfloat16_t x) |
| __bfloat162ll_rn | __bfloat162ll_rn | long long int __bfloat162ll_rn(const bfloat16_t x) |
| __bfloat162ll_ru | __bfloat162ll_ru | long long int __bfloat162ll_ru(const bfloat16_t x) |
| __bfloat162ll_rz | __bfloat162ll_rz | long long int __bfloat162ll_rz(const bfloat16_t x) |
| __bfloat162short_rd |  |  |
| __bfloat162short_rn |  |  |
| __bfloat162short_ru |  |  |
| __bfloat162short_rz |  |  |
| __bfloat162uchar_rz |  |  |
| __bfloat162uint_rd | __bfloat162uint_rd | unsigned int __bfloat162uint_rd(const bfloat16_t x) |
| __bfloat162uint_rn | __bfloat162uint_rn | unsigned int __bfloat162uint_rn(const bfloat16_t x) |
| __bfloat162uint_ru | __bfloat162uint_ru | unsigned int __bfloat162uint_ru(const bfloat16_t x) |
| __bfloat162uint_rz | __bfloat162uint_rz | unsigned int __bfloat162uint_rz(const bfloat16_t x) |
| __bfloat162ull_rd | __bfloat162ull_rd | unsigned long long int __bfloat162ull_rd(const bfloat16_t x) |
| __bfloat162ull_rn | __bfloat162ull_rn | unsigned long long int __bfloat162ull_rn(const bfloat16_t x) |
| __bfloat162ull_ru | __bfloat162ull_ru | unsigned long long int __bfloat162ull_ru(const bfloat16_t x) |
| __bfloat162ull_rz | __bfloat162ull_rz | unsigned long long int __bfloat162ull_rz(const bfloat16_t x) |
| __bfloat162ushort_rd |  |  |
| __bfloat162ushort_rn |  |  |
| __bfloat162ushort_ru |  |  |
| __bfloat162ushort_rz |  |  |
| __bfloat16_as_short |  |  |
| __bfloat16_as_ushort |  |  |
| __double2bfloat16 |  |  |
| __double2float_rd |  |  |
| __double2float_rn |  |  |
| __double2float_ru |  |  |
| __double2float_rz |  |  |
| __double2half |  |  |
| __double2hiint |  |  |
| __double2int_rd |  |  |
| __double2int_rn |  |  |
| __double2int_ru |  |  |
| __double2int_rz |  |  |
| __double2ll_rd |  |  |
| __double2ll_rn |  |  |
| __double2ll_ru |  |  |
| __double2ll_rz |  |  |
| __double2loint |  |  |
| __double2uint_rd |  |  |
| __double2uint_rn |  |  |
| __double2uint_ru |  |  |
| __double2uint_rz |  |  |
| __double2ull_rd |  |  |
| __double2ull_rn |  |  |
| __double2ull_ru |  |  |
| __double2ull_rz |  |  |
| __double_as_longlong |  |  |
| __float22bfloat162_rn | __float22bfloat162_rn | bfloat16x2_t __float22bfloat162_rn(const float2 x) |
| __float22half2_rn | __float22half2_rn | half2 __float22half2_rn(const float2 x) |
| __float2bfloat16 | __float2bfloat16 | bfloat16_t __float2bfloat16(const float x) |
| __float2bfloat162_rn | __float2bfloat162_rn | bfloat16x2_t __float2bfloat162_rn(const float x) |
| __float2bfloat16_rd | __float2bfloat16_rd | bfloat16_t __float2bfloat16_rd(const float x) |
| __float2bfloat16_rn | __float2bfloat16_rn | bfloat16_t __float2bfloat16_rn(const float x) |
| __float2bfloat16_ru | __float2bfloat16_ru | bfloat16_t __float2bfloat16_ru(const float x) |
| __float2bfloat16_rz | __float2bfloat16_rz | bfloat16_t __float2bfloat16_rz(const float x) |
| __float2half | __float2half | half __float2half(const float x) |
| __float2half2_rn |  |  |
| __float2half_rd | __float2half_rd | half __float2half_rd(const float x) |
| __float2half_rn | __float2half_rn | half __float2half_rn(const float x) |
| __float2half_ru | __float2half_ru | half __float2half_ru(const float x) |
| __float2half_rz | __float2half_rz | half __float2half_rz(const float x) |
| __float2int_rd | __float2int_rd | int __float2int_rd(const float x) |
| __float2int_rn | __float2int_rn | int __float2int_rn(const float x) |
| __float2int_ru | __float2int_ru | int __float2int_ru(const float x) |
| __float2int_rz | __float2int_rz | int __float2int_rz(const float x) |
| __float2ll_rd | __float2ll_rd | long long int __float2ll_rd(const float x) |
| __float2ll_rn | __float2ll_rn | long long int __float2ll_rn(const float x) |
| __float2ll_ru | __float2ll_ru | long long int __float2ll_ru(const float x) |
| __float2ll_rz | __float2ll_rz | long long int __float2ll_rz(const float x) |
| __float2uint_rd | __float2uint_rd | unsigned int __float2uint_rd(const float x) |
| __float2uint_rn | __float2uint_rn | unsigned int __float2uint_rn(const float x) |
| __float2uint_ru | __float2uint_ru | unsigned int __float2uint_ru(const float x) |
| __float2uint_rz | __float2uint_rz | unsigned int __float2uint_rz(const float x) |
| __float2ull_rd | __float2ull_rd | unsigned long long int __float2ull_rd(const float x) |
| __float2ull_rn | __float2ull_rn | unsigned long long int __float2ull_rn(const float x) |
| __float2ull_ru | __float2ull_ru | unsigned long long int __float2ull_ru(const float x) |
| __float2ull_rz | __float2ull_rz | unsigned long long int __float2ull_rz(const float x) |
| __float_as_int | __float_as_int | int __float_as_int(const float x) |
| __float_as_uint | __float_as_uint | unsigned int __float_as_uint(const float x) |
| __floats2bfloat162_rn | __floats2bfloat162_rn | bfloat16x2_t __floats2bfloat162_rn(const float x, const float y) |
| __floats2half2_rn | __floats2half2_rn | half2 __floats2half2_rn(const float x, const float y) |
| __half22float2 |  |  |
| __half2char_rz |  |  |
| __half2float | __half2float | float __half2float(const half x) |
| __half2half2 |  |  |
| __half2int_rd | __half2int_rd | int __half2int_rd(const half x) |
| __half2int_rn | __half2int_rn | int __half2int_rn(const half x) |
| __half2int_ru | __half2int_ru | int __half2int_ru(const half x) |
| __half2int_rz | __half2int_rz | int __half2int_rz(const half x) |
| __half2ll_rd | __half2ll_rd | long long int __half2ll_rd(const half x) |
| __half2ll_rn | __half2ll_rn | long long int __half2ll_rn(const half x) |
| __half2ll_ru | __half2ll_ru | long long int __half2ll_ru(const half x) |
| __half2ll_rz | __half2ll_rz | long long int __half2ll_rz(const half x) |
| __half2short_rd |  |  |
| __half2short_rn |  |  |
| __half2short_ru |  |  |
| __half2short_rz |  |  |
| __half2uchar_rz |  |  |
| __half2uint_rd | __half2uint_rd | unsigned int __half2uint_rd(const half x) |
| __half2uint_rn | __half2uint_rn | unsigned int __half2uint_rn(const half x) |
| __half2uint_ru | __half2uint_ru | unsigned int __half2uint_ru(const half x) |
| __half2uint_rz | __half2uint_rz | unsigned int __half2uint_rz(const half x) |
| __half2ull_rd | __half2ull_rd | unsigned long long int __half2ull_rd(const half x) |
| __half2ull_rn | __half2ull_rn | unsigned long long int __half2ull_rn(const half x) |
| __half2ull_ru | __half2ull_ru | unsigned long long int __half2ull_ru(const half x) |
| __half2ull_rz | __half2ull_rz | unsigned long long int __half2ull_rz(const half x) |
| __half2ushort_rd |  |  |
| __half2ushort_rn |  |  |
| __half2ushort_ru |  |  |
| __half2ushort_rz |  |  |
| __half_as_short |  |  |
| __half_as_ushort |  |  |
| __halves2bfloat162 | __halves2bfloat162 | bfloat16x2_t __halves2bfloat162(const bfloat16_t x, const bfloat16_t y) |
| __halves2half2 | __halves2half2 | half2 __halves2half2(const half x, const half y) |
| __high2bfloat16 | __high2bfloat16 | bfloat16_t __high2bfloat16(const bfloat16x2_t x) |
| __high2bfloat162 | __high2bfloat162 | bfloat16x2_t __high2bfloat162(const bfloat16x2_t x) |
| __high2float | __high2float | float __high2float(const bfloat16x2_t x) |
| __high2half | __high2half | half __high2half(const half2 x) |
| __high2half2 | __high2half2 | half2 __high2half2(const half2 x) |
| __highs2bfloat162 | __highs2bfloat162 | bfloat16x2_t __highs2bfloat162(const bfloat16x2_t x, const bfloat16x2_t y) |
| __highs2half2 | __highs2half2 | half2 __highs2half2(const half2 x, const half2 y) |
| __hiloint2double |  |  |
| __int2bfloat16_rd | __int2bfloat16_rd | bfloat16_t __int2bfloat16_rd(const int x) |
| __int2bfloat16_rn | __int2bfloat16_rn | bfloat16_t __int2bfloat16_rn(const int x) |
| __int2bfloat16_ru | __int2bfloat16_ru | bfloat16_t __int2bfloat16_ru(const int x) |
| __int2bfloat16_rz | __int2bfloat16_rz | bfloat16_t __int2bfloat16_rz(const int x) |
| __int2double_rn |  |  |
| __int2float_rd | __int2float_rd | float __int2float_rd(const int x) |
| __int2float_rn | __int2float_rn | float __int2float_rn(const int x) |
| __int2float_ru | __int2float_ru | float __int2float_ru(const int x) |
| __int2float_rz | __int2float_rz | float __int2float_rz(const int x) |
| __int2half_rd | __int2half_rd | half __int2half_rd(const int x) |
| __int2half_rn | __int2half_rn | half __int2half_rn(const int x) |
| __int2half_ru | __int2half_ru | half __int2half_ru(const int x) |
| __int2half_rz | __int2half_rz | half __int2half_rz(const int x) |
| __int_as_float | __int_as_float | float __int_as_float(const int x) |
| __ll2bfloat16_rd | __ll2bfloat16_rd | bfloat16_t __ll2bfloat16_rd(const long long int x) |
| __ll2bfloat16_rn | __ll2bfloat16_rn | bfloat16_t __ll2bfloat16_rn(const long long int x) |
| __ll2bfloat16_ru | __ll2bfloat16_ru | bfloat16_t __ll2bfloat16_ru(const long long int x) |
| __ll2bfloat16_rz | __ll2bfloat16_rz | bfloat16_t __ll2bfloat16_rz(const long long int x) |
| __ll2double_rd |  |  |
| __ll2double_rn |  |  |
| __ll2double_ru |  |  |
| __ll2double_rz |  |  |
| __ll2float_rd | __ll2float_rd | float __ll2float_rd(const long long int x) |
| __ll2float_rn | __ll2float_rn | float __ll2float_rn(const long long int x) |
| __ll2float_ru | __ll2float_ru | float __ll2float_ru(const long long int x) |
| __ll2float_rz | __ll2float_rz | float __ll2float_rz(const long long int x) |
| __ll2half_rd | __ll2half_rd | half __ll2half_rd(const long long int x) |
| __ll2half_rn | __ll2half_rn | half __ll2half_rn(const long long int x) |
| __ll2half_ru | __ll2half_ru | half __ll2half_ru(const long long int x) |
| __ll2half_rz | __ll2half_rz | half __ll2half_rz(const long long int x) |
| __longlong_as_double |  |  |
| __low2bfloat16 | __low2bfloat16 | bfloat16_t __low2bfloat16(const bfloat16x2_t x) |
| __low2bfloat162 | __low2bfloat162 | bfloat16x2_t __low2bfloat162(const bfloat16x2_t x) |
| __low2float | __low2float | float __low2float(const bfloat16x2_t x) |
| __low2half | __low2half | half __low2half(const half2 x) |
| __low2half2 | __low2half2 | half2 __low2half2(const half2 x) |
| __lowhigh2highlow | __lowhigh2highlow | bfloat16x2_t __lowhigh2highlow(const bfloat16x2_t x) |
| __lows2bfloat162 | __lows2bfloat162 | bfloat16x2_t __lows2bfloat162(const bfloat16x2_t x, const bfloat16x2_t y) |
| __lows2half2 | __lows2half2 | half2 __lows2half2(const half2 x, const half2 y) |
| __nv_cvt_bfloat162raw_to_e8m0x2 |  |  |
| __nv_cvt_bfloat16raw2_to_fp4x2 |  |  |
| __nv_cvt_bfloat16raw2_to_fp6x2 |  |  |
| __nv_cvt_bfloat16raw2_to_fp8x2 |  |  |
| __nv_cvt_bfloat16raw_to_e8m0 |  |  |
| __nv_cvt_bfloat16raw_to_fp4 |  |  |
| __nv_cvt_bfloat16raw_to_fp6 |  |  |
| __nv_cvt_bfloat16raw_to_fp8 |  |  |
| __nv_cvt_double2_to_e8m0x2 |  |  |
| __nv_cvt_double2_to_fp4x2 |  |  |
| __nv_cvt_double2_to_fp6x2 |  |  |
| __nv_cvt_double2_to_fp8x2 |  |  |
| __nv_cvt_double_to_e8m0 |  |  |
| __nv_cvt_double_to_fp4 |  |  |
| __nv_cvt_double_to_fp6 |  |  |
| __nv_cvt_double_to_fp8 |  |  |
| __nv_cvt_e8m0_to_bf16raw |  |  |
| __nv_cvt_e8m0x2_to_bf162raw |  |  |
| __nv_cvt_float2_to_e8m0x2 |  |  |
| __nv_cvt_float2_to_fp4x2 |  |  |
| __nv_cvt_float2_to_fp6x2 |  |  |
| __nv_cvt_float2_to_fp8x2 |  |  |
| __nv_cvt_float_to_e8m0 |  |  |
| __nv_cvt_float_to_fp4 |  |  |
| __nv_cvt_float_to_fp6 |  |  |
| __nv_cvt_float_to_fp8 |  |  |
| __nv_cvt_fp4_to_halfraw |  |  |
| __nv_cvt_fp4x2_to_halfraw2 |  |  |
| __nv_cvt_fp6_to_halfraw |  |  |
| __nv_cvt_fp6x2_to_halfraw2 |  |  |
| __nv_cvt_fp8_to_halfraw |  |  |
| __nv_cvt_fp8x2_to_halfraw2 |  |  |
| __nv_cvt_halfraw2_to_fp4x2 |  |  |
| __nv_cvt_halfraw2_to_fp6x2 |  |  |
| __nv_cvt_halfraw2_to_fp8x2 |  |  |
| __nv_cvt_halfraw_to_fp4 |  |  |
| __nv_cvt_halfraw_to_fp6 |  |  |
| __nv_cvt_halfraw_to_fp8 |  |  |
| __short2bfloat16_rd |  |  |
| __short2bfloat16_rn |  |  |
| __short2bfloat16_ru |  |  |
| __short2bfloat16_rz |  |  |
| __short2half_rd |  |  |
| __short2half_rn |  |  |
| __short2half_ru |  |  |
| __short2half_rz |  |  |
| __short_as_bfloat16 |  |  |
| __short_as_half |  |  |
| __uint2bfloat16_rd | __uint2bfloat16_rd | bfloat16_t __uint2bfloat16_rd(const unsigned int x) |
| __uint2bfloat16_rn | __uint2bfloat16_rn | bfloat16_t __uint2bfloat16_rn(const unsigned int x) |
| __uint2bfloat16_ru | __uint2bfloat16_ru | bfloat16_t __uint2bfloat16_ru(const unsigned int x) |
| __uint2bfloat16_rz | __uint2bfloat16_rz | bfloat16_t __uint2bfloat16_rz(const unsigned int x) |
| __uint2double_rn |  |  |
| __uint2float_rd | __uint2float_rd | float __uint2float_rd(const unsigned int x) |
| __uint2float_rn | __uint2float_rn | float __uint2float_rn(const unsigned int x) |
| __uint2float_ru | __uint2float_ru | float __uint2float_ru(const unsigned int x) |
| __uint2float_rz | __uint2float_rz | float __uint2float_rz(const unsigned int x) |
| __uint2half_rd | __uint2half_rd | half __uint2half_rd(const unsigned int x) |
| __uint2half_rn | __uint2half_rn | half __uint2half_rn(const unsigned int x) |
| __uint2half_ru | __uint2half_ru | half __uint2half_ru(const unsigned int x) |
| __uint2half_rz | __uint2half_rz | half __uint2half_rz(const unsigned int x) |
| __uint_as_float | __uint_as_float | float __uint_as_float(const unsigned int x) |
| __ull2bfloat16_rd | __ull2bfloat16_rd | bfloat16_t __ull2bfloat16_rd(const unsigned long long int x) |
| __ull2bfloat16_rn | __ull2bfloat16_rn | bfloat16_t __ull2bfloat16_rn(const unsigned long long int x) |
| __ull2bfloat16_ru | __ull2bfloat16_ru | bfloat16_t __ull2bfloat16_ru(const unsigned long long int x) |
| __ull2bfloat16_rz | __ull2bfloat16_rz | bfloat16_t __ull2bfloat16_rz(const unsigned long long int x) |
| __ull2double_rd |  |  |
| __ull2double_rn |  |  |
| __ull2double_ru |  |  |
| __ull2double_rz |  |  |
| __ull2float_rd | __ull2float_rd | float __ull2float_rd(const unsigned long long int x) |
| __ull2float_rn | __ull2float_rn | float __ull2float_rn(const unsigned long long int x) |
| __ull2float_ru | __ull2float_ru | float __ull2float_ru(const unsigned long long int x) |
| __ull2float_rz | __ull2float_rz | float __ull2float_rz(const unsigned long long int x) |
| __ull2half_rd | __ull2half_rd | half __ull2half_rd(const unsigned long long int x) |
| __ull2half_rn | __ull2half_rn | half __ull2half_rn(const unsigned long long int x) |
| __ull2half_ru | __ull2half_ru | half __ull2half_ru(const unsigned long long int x) |
| __ull2half_rz | __ull2half_rz | half __ull2half_rz(const unsigned long long int x) |
| __ushort2bfloat16_rd |  |  |
| __ushort2bfloat16_rn |  |  |
| __ushort2bfloat16_ru |  |  |
| __ushort2bfloat16_rz |  |  |
| __ushort2half_rd |  |  |
| __ushort2half_rn |  |  |
| __ushort2half_ru |  |  |
| __ushort2half_rz |  |  |
| __ushort_as_bfloat16 | __ushort_as_bfloat16 | bfloat16_t __ushort_as_bfloat16(const unsigned short int x) |
| __ushort_as_half | __ushort_as_half | half __ushort_as_half(const unsigned short int x) |
| float2int |  |  |
| float_as_int | __float_as_int | int __float_as_int(const float x) |
| float_as_uint | __float_as_uint | unsigned int __float_as_uint(const float x) |
| int2float |  |  |
| int_as_float | __int_as_float | float __int_as_float(const int x) |
| uint2float |  |  |
| uint_as_float | __uint_as_float | float __uint_as_float(const unsigned int x) |

## vector-type-constructor

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| make_bfloat162 | make_bfloat162 | bfloat16x2_t make_bfloat162(bfloat16_t x, bfloat16_t y) |
| make_char1 |  |  |
| make_char2 | make_char2 | char2 make_char2(signed char x, signed char y) |
| make_char3 | make_char3 | char3 make_char3(signed char x, signed char y, signed char z) |
| make_char4 | make_char4 | char4 make_char4(signed char x, signed char y, signed char z, signed char w) |
| make_double1 |  |  |
| make_double2 |  |  |
| make_double3 |  |  |
| make_double4 |  |  |
| make_double4_16a |  |  |
| make_double4_32a |  |  |
| make_float1 |  |  |
| make_float2 | make_float2 | float2 make_float2(float x, float y) |
| make_float3 | make_float3 | float3 make_float3(float x, float y, float z) |
| make_float4 | make_float4 | float4 make_float4(float x, float y, float z, float w) |
| make_half2 | make_half2 | half2 make_half2(half x, half y) |
| make_int1 |  |  |
| make_int2 | make_int2 | int2 make_int2(int x, int y) |
| make_int3 | make_int3 | int3 make_int3(int x, int y, int z) |
| make_int4 | make_int4 | int4 make_int4(int x, int y, int z, int w) |
| make_long1 |  |  |
| make_long2 | make_long2 | long2 make_long2(long int x, long int y) |
| make_long3 | make_long3 | long3 make_long3(long int x, long int y, long int z) |
| make_long4 | make_long4 | long4 make_long4(long int x, long int y, long int z, long int w) |
| make_long4_16a |  |  |
| make_long4_32a |  |  |
| make_longlong1 |  |  |
| make_longlong2 | make_longlong2 | longlong2 make_longlong2(long long int x, long long int y) |
| make_longlong3 | make_longlong3 | longlong3 make_longlong3(long long int x, long long int y, long long int z) |
| make_longlong4 |  |  |
| make_longlong4_16a |  |  |
| make_longlong4_32a |  |  |
| make_short1 |  |  |
| make_short2 | make_short2 | short2 make_short2(short x, short y) |
| make_short3 | make_short3 | short3 make_short3(short x, short y, short z) |
| make_short4 | make_short4 | short4 make_short4(short x, short y, short z, short w) |
| make_uchar1 |  |  |
| make_uchar2 | make_uchar2 | uchar2 make_uchar2(unsigned char x, unsigned char y) |
| make_uchar3 | make_uchar3 | uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) |
| make_uchar4 |  |  |
| make_uint1 |  |  |
| make_uint2 | make_uint2 | uint2 make_uint2(unsigned int x, unsigned int y) |
| make_uint3 | make_uint3 | uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) |
| make_uint4 | make_uint4 | uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) |
| make_ulong1 |  |  |
| make_ulong2 | make_ulong2 | ulong2 make_ulong2(unsigned long int x, unsigned long int y) |
| make_ulong3 | make_ulong3 | ulong3 make_ulong3(unsigned long int x, unsigned long int y, unsigned long int z) |
| make_ulong4 |  |  |
| make_ulong4_16a |  |  |
| make_ulong4_32a |  |  |
| make_ulonglong1 |  |  |
| make_ulonglong2 | make_ulonglong2 | ulonglong2 make_ulonglong2(unsigned long long int x, unsigned long long int y) |
| make_ulonglong3 |  |  |
| make_ulonglong4 |  |  |
| make_ulonglong4_16a |  |  |
| make_ulonglong4_32a |  |  |
| make_ushort1 |  |  |
| make_ushort2 | make_ushort2 | ushort2 make_ushort2(unsigned short x, unsigned short y) |
| make_ushort3 | make_ushort3 | ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) |
| make_ushort4 |  |  |

## cache-hint-load-store

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| __ldca | asc_ldca | float4 asc_ldca(__gm__ float4* address) |
| __ldcg | asc_ldcg | float4 asc_ldcg(__gm__ float4* address) |
| __ldcs |  |  |
| __ldcv |  |  |
| __ldg |  |  |
| __ldlu |  |  |
| __stcg | asc_stcg | void asc_stcg(__gm__ float4* address, float4 val) |
| __stcs |  |  |
| __stwb |  |  |
| __stwt | asc_stwt | void asc_stwt(__gm__ float4* address, float4 val) |

## debug

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| __assert_fail | assert |  |
| __assertfail | assert |  |
| __brkpt |  |  |
| __pm0 |  |  |
| __pm1 |  |  |
| __pm2 |  |  |
| __pm3 |  |  |
| __prof_trigger |  |  |
| __trap | __trap |  |
| clock | clock |  |
| clock64 |  |  |
