//#include <cilk-adept-headers/tfk_shadowmem.h>

class TFKShadowMemory {

  TFKShadowMemory** table;
  int nbytes = 0;
  float* data;

  int64_t last_key = 0;

  TFKShadowMemory* root;

  TFKShadowMemory* finger;

  public:
  //extern "C" {
  TFKShadowMemory (int _nbytes, TFKShadowMemory* _root) {
    nbytes = _nbytes;
    if (_nbytes == 8) {
      root = this;
    } else {
      root = _root;
    }

    finger = NULL;
    last_key = 0;
    if (nbytes > 1) {
      table = (TFKShadowMemory**) calloc(256, sizeof(TFKShadowMemory*));
      //for (int i = 0; i < 256; i++) {
      //  table[i] = NULL;
      //}
      data = NULL;
    } else {
      //for (int i = 0; i < 256; i++) {
      //  table[i] = NULL;
      //}
      //table = NULL;
      data = (float*) calloc(256, sizeof(float));
      for (int i = 0; i < 256; i++) {
        data[i] = 0.0;
      }
    }
  }


  ~TFKShadowMemory() {
    if (nbytes > 1) {
      for (int i = 0; i < 256; i++) {
        if (table[i] != 0) {
          delete table[i];
        }
      }
      free(table);
    } else if (nbytes == 1){
      free(data);
    }
  }

  float extract_value(uint64_t key) {
    //if (nbytes == 8) {
    //  printf("looking at in extract %llu\n", key);
    //}
    if (nbytes == 8) key = key/4;

    if (nbytes == 8 && ((key&(0xFFFFFFFFFFFFFF00)) == (last_key&(0xFFFFFFFFFFFFFF00))) &&
        finger != NULL) {
      //printf("hit on extract\n");
      return finger->data[key%256];
    } else if (nbytes == 8){
      last_key = key;
      finger = NULL;
    }
    uint8_t idx = (key & (0xFF00000000000000))>>(64-8);
    if (nbytes == 1) {
      float val = data[idx];
      data[idx] = 0.0;
      //last_key = key;
      root->finger = this;
      //finger = this;
      return val;
    }
    if (table[idx] == NULL) {
      return 0.0;
      //table[idx] = new TFKShadowMemory(nbytes-2);
    } else {
      return table[idx]->extract_value((key<<(8)));
    }
  }

  __attribute__((always_inline)) void accumulate_value(uint64_t key, float value) {
    //if (nbytes == 8) {
    //  printf("looking at key %llu, last key %llu\n", key, last_key);
    //}
    if (nbytes == 8) key = key/4;
    if (nbytes == 8 && ((key&(0xFFFFFFFFFFFFFF00)) == (last_key&(0xFFFFFFFFFFFFFF00))) &&
        finger != NULL) {
      //printf("hit on accumulate\n");
      finger->data[key%256] += value;
      return;
    } else if (nbytes == 8){
      last_key = key;
      finger = NULL;
    }
    uint8_t idx = (key & (0xFF00000000000000))>>(64-8);
    //printf("nbytes is %d\n", nbytes);
    if (nbytes == 1) {
      data[idx] += value;
      //last_key = key;
      root->finger = this;
      return;
    }

    //uint64_t _idx = (key & (0xFF00000000000000))>>(64-8);
    //printf("The idx is before %llu after %llu\n", key, _idx);
    if (table[idx] == NULL) {
      table[idx] = new TFKShadowMemory(nbytes-1, root);
    }

    table[idx]->accumulate_value((key<<(8)), value);
  }
  //}
};
