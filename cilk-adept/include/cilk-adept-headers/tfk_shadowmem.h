
#ifndef TFK_SHADOWMEM_H
#define TFK_SHADOWMEM_H


class TFKShadowMemory {

  TFKShadowMemory** table;
  int nbytes = 0;
  float* data;

  int64_t last_key = 0;

  TFKShadowMemory* root;

  TFKShadowMemory* finger;

  public:
  //extern "C" {
  TFKShadowMemory (int _nbytes, TFKShadowMemory* _root);
  ~TFKShadowMemory();
  float extract_value(uint64_t key);

  __attribute__((always_inline)) void accumulate_value(uint64_t key, float value);
};

#endif
