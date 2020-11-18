//
// Created by biba_bo on 2020-11-16.
//

#include <zconf.h>
#include "process_camera_stream/gst_pipeline_processor.h"

int main(int argv, char *argc[]) {
    std::cout << "Prepare for initing!\n";
    sleep(2);
    std::cout << "End of preparing!\n";
    return start_gst_loop();
}