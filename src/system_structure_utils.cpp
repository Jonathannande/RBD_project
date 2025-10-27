//
// Created by Jonathan on 10/26/2025.
//
#import "system_of_bodies.h"


void SystemOfBodies::set_BWA() {
    // Based on the structure, which this should call first, a BWA is set up, which in this case could merely be a normal adjacency matrix


}

void SystemOfBodies::set_system_structure() {
    // This function should do a depth-first search and arrange body ids to match a strictly canonical structure

    for (int i = 0; i < n; ++i) {
        if (bodies[i]->parent_ID == 0) {
            bodies[i]->body_ID = n;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (bodies[i]->parent_ID == i) {}
    }

}


