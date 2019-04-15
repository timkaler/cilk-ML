#include <csi/csi.h>

extern "C" {

void __csi_init() {}

    void __csi_unit_init(const char * const file_name,
        const instrumentation_counts_t counts) {}


void __csi_task(const csi_id_t task_id, const csi_id_t detach_id) {
  //std::cout << "Started task\n";
}

void __csi_task_exit(const csi_id_t task_exit_id,
                          const csi_id_t task_id,
                          const csi_id_t detach_id) {
  //std::cout << "Exiting task\n";
}

void __csi_detach_continue(const csi_id_t detach_continue_id,
                                const csi_id_t detach_id) {
  //std::cout << "Starting continuation\n";
}

void __csi_detach(const csi_id_t detach_id, const int32_t* has_spawned) {
  //std::cout << "Before spawn\n";
}

 void __csi_before_sync(const csi_id_t sync_id, const int32_t* has_spawned) {
   //std::cout << "Ending continuation\n";
   if (*has_spawned) {
   }
 }
 void __csi_after_sync(const csi_id_t sync_id, const int32_t* has_spawned) {
  //std::cout << "After sync\n";
  if (*has_spawned) {
  }
  // tfk_reducer.sp_tree.close_P_node();  
}

} // extern "C"
