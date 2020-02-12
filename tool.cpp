#include <csi/csi.h>
#include <adept.h>
#include <cilk-adept-headers/tfkparallel.h>

void tfk_init();

extern "C" {

void __csi_init() {/*tfk_reducer.sp_tree.recording = true;*/}

    void __csi_unit_init(const char * const file_name,
        const instrumentation_counts_t counts) {}


void __csi_task(const csi_id_t task_id, const csi_id_t detach_id) {
  //std::cout << "Started task\n";
  if (!tfk_reducer.sp_tree.recording) return;
  tfk_reducer.sp_tree.open_S_node();
  tfk_init();
}

void __csi_task_exit(const csi_id_t task_exit_id,
                          const csi_id_t task_id,
                          const csi_id_t detach_id) {
  if (!tfk_reducer.sp_tree.recording) return;
  //std::cout << "Exiting task\n";
  tfk_reducer.sp_tree.close_S_node();
  //tfk_init(); // new
}

void __csi_detach_continue(const csi_id_t detach_continue_id,
                                const csi_id_t detach_id) {

  if (!tfk_reducer.sp_tree.recording) return;
  //std::cout << "Starting continuation\n";
  tfk_reducer.sp_tree.open_S_node();
  tfk_init();
}

void __csi_detach(const csi_id_t detach_id, const int32_t* has_spawned) {
  if (!tfk_reducer.sp_tree.recording) return;
  //std::cout << "Before spawn\n";
  tfk_reducer.sp_tree.open_P_node((void*) has_spawned);
  //tfk_init(); // new
}

 void __csi_before_sync(const csi_id_t sync_id, const int32_t* has_spawned) {
  if (!tfk_reducer.sp_tree.recording) return;
   //std::cout << "Ending continuation\n";
   if (*has_spawned) {
     tfk_reducer.sp_tree.close_S_node();
     //tfk_init(); // new
   }
 }
 void __csi_after_sync(const csi_id_t sync_id, const int32_t* has_spawned) {
  if (!tfk_reducer.sp_tree.recording) return;
  //std::cout << "After sync\n";
  if (*has_spawned) {
    tfk_reducer.sp_tree.sync_P_nodes((void*) has_spawned);
    //tfk_init();
    tfk_init();
  }
  // tfk_reducer.sp_tree.close_P_node();  
}

} // extern "C"
